# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import tomli
from collections import defaultdict
import torch.nn as nn


from nn_module import PlainMLP,DAN_NoBN,DomainDiscriminator
from mmd_utils import *

log = defaultdict(list)
class_names = ["-1", "B", "IR", "OR"]


# =========================
# Data: loaders + scaler
# =========================
def get_domain_loaders(
    **kwargs,
):
    shuffle_src = True
    drop_last_src = False
    shuffle_tgt = False
    drop_last_tgt = False

    src_path = kwargs.get("SRC_CSV")
    tgt_path = kwargs.get("TGT_CSV")
    batch_size = kwargs.get("BATCH_SIZE")
    label_col = kwargs.get("LABEL_COL")

    df_src = pd.read_csv(src_path)
    df_tgt = pd.read_csv(tgt_path)

    # 标签映射（统一）
    y_raw = df_src.iloc[:, label_col].astype(str).values
    cls2id = {c: i for i, c in enumerate(class_names)}
    ys_mapped = pd.Series(y_raw).map(cls2id)
    ys_np = ys_mapped.to_numpy(dtype=np.int64)
    ys = torch.tensor(ys_np, dtype=torch.long)

    # 特征（源/目标一致切片）
    Xs_df = df_src.iloc[:, 4:]
    Xt_df = df_tgt

    Xs_np = Xs_df.apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy(dtype=np.float32)
    Xt_np = Xt_df.apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy(dtype=np.float32)

    # 统一标准化（源+目标一起 fit）
    scaler = StandardScaler().fit(np.vstack([Xs_np, Xt_np]))
    Xs_np = scaler.transform(Xs_np)
    Xt_np = scaler.transform(Xt_np)

    Xs = torch.tensor(Xs_np, dtype=torch.float32)
    Xt = torch.tensor(Xt_np, dtype=torch.float32)

    src_ds = TensorDataset(Xs, ys)
    tgt_ds = TensorDataset(Xt, torch.zeros(len(Xt), dtype=torch.long))  # dummy label

    src_loader = DataLoader(src_ds, batch_size=batch_size, shuffle=shuffle_src, drop_last=drop_last_src)
    tgt_loader = DataLoader(tgt_ds, batch_size=batch_size, shuffle=shuffle_tgt, drop_last=drop_last_tgt)

    in_dim = Xs.shape[1]
    n_classes = len(class_names)

    return src_loader, tgt_loader, in_dim, n_classes, scaler, cls2id


# =========================
# Baseline MLP
# =========================
def train_mlp(
    src_loader: DataLoader,
    in_dim: int,
    n_classes: int,
    **kwargs
):
    """
    train mlp model
    """
    epochs = kwargs.get("EPOCHS_BASE")
    lr = kwargs.get("LEARNING_RATE")
    device = kwargs.get("DEVICE")
    hidden = kwargs.get("HIDDEN_DIM")
    weight_decay = kwargs.get("WEIGHT_DECAY")

    ys_all = []
    for _, yb in src_loader:
        ys_all.append(yb.numpy())
    ys_all = np.concatenate(ys_all)
    weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=ys_all)
    w_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    model = PlainMLP(in_dim, n_classes, hidden=hidden).to(device)
    criterion = nn.CrossEntropyLoss(weight=w_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for ep in range(1, epochs + 1):
        model.train()
        ce_vals = []
        for X, y in src_loader:
            X = X.to(device).float()
            y = y.to(device).long()
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            ce_vals.append(loss.item())
        sched.step()
        if ep % 5 == 0 or ep == 1:
            print(f"[MLP Epoch {ep:02d}] CE: {np.mean(ce_vals):.4f}")

    eval_mlp(model, src_loader, device=device)
    return model,w_tensor

def eval_mlp(model, loader, device="cpu"):
    "calculate accuracy of mlp on eval set"
    model.eval()
    with torch.no_grad():
        total_correct = 0
        for X, y in loader:
            X = X.to(device).float()
            logits = model(X)
            pred = logits.argmax(1)
            total_correct += (pred == y).sum().item()
        acc = total_correct / len(loader.dataset)
        print(f"MLP Accuracy on eval set: {acc:.4f}")

# =========================
# DAN (No BN) + MMD + A-distance
# =========================
@torch.no_grad()
def init_dan_from_plain_mlp(dan_model: DAN_NoBN, mlp_model: PlainMLP):
    linear_layers = [m for m in mlp_model.net if isinstance(m, nn.Linear)]
    L_backbone = len(dan_model.backbone.layers)
    assert len(linear_layers) == L_backbone + 1, \
        f"期望 baseline 有 {L_backbone+1} 个 Linear（含分类层），实际 {len(linear_layers)}"

    for i in range(L_backbone):
        dst = dan_model.backbone.layers[i]
        src = linear_layers[i]
        dst.weight.copy_(src.weight)
        dst.bias.copy_(src.bias)

    dan_model.classifier.fc.weight.copy_(linear_layers[-1].weight)
    dan_model.classifier.fc.bias.copy_(linear_layers[-1].bias)
    print("[Init] DAN initialized from PlainMLP weights.")


# =========================
# Training (supports external model)
# =========================
def train_dan(
    src_loader: DataLoader,
    tgt_loader: DataLoader,
    in_dim: int,
    n_classes: int,
    mlp_model,
    weights,
    **kwargs,
):
    hid_dims = kwargs.get("HID_DIMS")
    mmd_layers = kwargs.get("MMD_LAYERS")
    mmd_weight = kwargs.get("MMD_WEIGHT")
    epochs = kwargs.get("EPOCHS")
    warmup_epochs = kwargs.get("WARMUP")
    num_scales = kwargs.get("NUM_SCALES")
    lr = kwargs.get("LEARNING_RATE")
    grad_clip = kwargs.get("GRAD_CLIP")
    repeat_target_to_match = True
    device = "cpu"

    model = DAN_NoBN(in_dim, hid_dims, n_classes, mmd_layers).to(device)
    init_dan_from_plain_mlp(model, mlp_model)  # 用 baseline 权重热启动

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    mmd_history, clf_loss_history = [], []

    # 目标域全集
    Xt_all = []
    for xb_t, _ in tgt_loader:
        Xt_all.append(xb_t)
    Xt_all = torch.cat(Xt_all, dim=0).to(device).float()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_mmd_vals, epoch_clf_vals = [], []
        iter_src = iter(src_loader)

        for _ in range(len(src_loader)):
            x_s, y_s = next(iter_src)
            x_s = x_s.to(device).float()
            y_s = y_s.to(device).long()

            if repeat_target_to_match and Xt_all.size(0) < x_s.size(0):
                rep = max(1, x_s.size(0) // Xt_all.size(0))
                x_t = Xt_all.repeat(rep, 1)[:x_s.size(0)]
            else:
                x_t = Xt_all

            optimizer.zero_grad()
            logits_s, feats_s, feats_t_full = model(x_s, x_t)
            clf_loss = criterion(logits_s, y_s)

            mmd_terms = []
            for li in mmd_layers:
                sigmas = mk_sigmas_median(feats_s[li].detach(), feats_t_full[li].detach(), num_scales=num_scales)
                mmd_terms.append(mmd_rbf(feats_s[li], feats_t_full[li], sigmas=sigmas))
            mmd_total = torch.stack(mmd_terms).sum()

            cur_w = mmd_weight * min(1.0, epoch / max(1, warmup_epochs))
            loss = clf_loss + cur_w * mmd_total

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_clf_vals.append(clf_loss.detach().item())
            epoch_mmd_vals.append(mmd_total.detach().item())


        avg_mmd = float(np.mean(epoch_mmd_vals))
        avg_clf = float(np.mean(epoch_clf_vals))
        mmd_history.append(avg_mmd)
        clf_loss_history.append(avg_clf)
        sched.step()

        print(f"[Epoch {epoch:02d}] CE: {avg_clf:.4f} | MMD: {avg_mmd:.4f} | total(w={cur_w:.2f}): {(avg_clf + cur_w*avg_mmd):.4f}")

    # MMD 曲线
    plt.figure()
    plt.plot(range(1, epochs + 1), mmd_history, marker='o')
    plt.xlabel("Epoch"); plt.ylabel("MMD"); plt.title("DAN Training: MMD per Epoch (with warmup)")
    plt.grid(True); plt.tight_layout(); plt.show()

    return model, mmd_history, clf_loss_history


# =========================
# Evaluation & Prediction
# =========================
@torch.no_grad()
def eval_on_loader(model, loader, device="cpu", class_names=None, title_prefix="Eval", has_label=True):
    model.eval()
    all_y_true, all_y_pred = [], []
    all_probs = []

    for xb, yb in loader:
        xb = xb.to(device).float()
        feats = model.backbone(xb)
        logits = model.classifier(feats[-1])
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(1)

        all_probs.append(probs)
        all_y_pred.append(preds)
        if has_label:
            all_y_true.append(yb.numpy())

    all_probs = np.vstack(all_probs)
    all_preds = np.concatenate(all_y_pred)

    if has_label:
        y_true = np.concatenate(all_y_true)
        acc = accuracy_score(y_true, all_preds)
        print(f"\n[{title_prefix}] Accuracy: {acc:.4f}")
        print(classification_report(y_true, all_preds, digits=4,
                                    target_names=(class_names if class_names else None)))

        cm = confusion_matrix(y_true, all_preds)
        plt.figure()
        plt.imshow(cm, interpolation='nearest')
        plt.title(f'{title_prefix} Confusion Matrix')
        plt.colorbar()
        ticks = np.arange(cm.shape[0])
        plt.xticks(ticks, class_names if class_names else ticks, rotation=45)
        plt.yticks(ticks, class_names if class_names else ticks)
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('True label'); plt.xlabel('Predicted label')
        plt.tight_layout(); plt.show()
    else:
        print(f"\n[{title_prefix}] No labels available. Showing confidence histogram...")
        prob_max = all_probs.max(axis=1)
        plt.figure()
        plt.hist(prob_max, bins=20)
        plt.xlabel('Max class probability'); plt.ylabel('Count')
        plt.title(f'{title_prefix} confidence histogram')
        plt.tight_layout(); plt.show()

    return all_preds, all_probs

@torch.no_grad()
def infer_on_loader(model, loader, device="cpu"):
    """返回 y_true, y_pred, prob_max（若 loader 没有真标签，y_true 返回 None）"""
    model.eval()
    all_y_true, all_y_pred, all_prob_max = [], [], []
    for xb, yb in loader:
        xb = xb.to(device).float()
        feats = model.backbone(xb)
        logits = model.classifier(feats[-1])
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(1)
        all_y_pred.append(preds)
        all_prob_max.append(probs.max(axis=1))
        # 若 yb 是 dummy 也会有值，但你知道目标域是无标签
        all_y_true.append(yb.numpy())
    y_pred = np.concatenate(all_y_pred)
    prob_max = np.concatenate(all_prob_max)
    y_true = np.concatenate(all_y_true) if hasattr(loader.dataset, 'tensors') else None
    return y_true, y_pred, prob_max

@torch.no_grad()
def predict_target_csv(model, tgt_csv_path: str, device: str = "cpu",
                       out_csv: str = "predictions_target.csv",
                       header_has_label: bool = False,
                       scaler: StandardScaler = None,
                       class_names=None, low_conf_thresh: float = 0.6):
    df = pd.read_csv(tgt_csv_path)
    if header_has_label:
        y_true = df.iloc[:, 0].to_numpy()
        X = df.to_numpy(dtype=np.float32)
    else:
        y_true = None
        X = df.to_numpy(dtype=np.float32)

    if scaler is not None:
        X = scaler.transform(X)

    X_tensor = torch.from_numpy(X).to(device).float()
    model.eval()
    feats = model.backbone(X_tensor)
    logits = model.classifier(feats[-1])
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = probs.argmax(axis=1); prob_max = probs.max(axis=1)

    out_df = pd.DataFrame({"pred": preds, "prob_max": prob_max})
    for k in range(probs.shape[1]):
        out_df[f"p_{k}"] = probs[:, k]
    if y_true is not None:
        out_df.insert(0, "label_true", y_true)

    out_df.to_csv(out_csv, index=False)
    print(f"[Target] Saved predictions to: {out_csv}")

    if y_true is None:
        low_idx = np.where(prob_max < low_conf_thresh)[0]
        print(f"[Target] Low-confidence (<{low_conf_thresh}): {len(low_idx)} (first 20 idx: {low_idx[:20].tolist()})")
    return out_df

# =========================
# Main
# =========================
def main():
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    with open("params.toml",'rb') as f:
        param = tomli.load(f)

    # 1) 数据
    src_loader, tgt_loader, in_dim, n_classes, scaler, cls2id = get_domain_loaders(**param["data_parameter"])

    # 2) 基线 PlainMLP,验证源域特征可分
    mlp_model,weight = train_mlp(src_loader=src_loader, in_dim=in_dim, n_classes=n_classes,**param["MLP"])

    # 3)
    model_dan, mmd_hist1, ce_hist1= train_dan(
        src_loader=src_loader, tgt_loader=tgt_loader,
        in_dim=in_dim, n_classes=n_classes,
        mlp_model=mlp_model, weights=weight,
        **param["dan_parameter"]
    )

if __name__ == "__main__":
    main()

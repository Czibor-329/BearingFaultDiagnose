# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit


# ==================== 基础/度量函数 ====================
def _check_X(X, name="X"):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"{name} 必须是二维数组，当前形状={X.shape}")
    if not np.isfinite(X).all():
        raise ValueError(f"{name} 含 NaN/Inf，请先清洗。")
    return X

def _cov_power(X0, power, eps):
    d = X0.shape[1]
    C = np.cov(X0, rowvar=False) + eps * np.eye(d)
    evals, evecs = np.linalg.eigh(C)
    evals = np.clip(evals, eps, None)
    return evecs @ np.diag(evals ** power) @ evecs.T

def coral_fit(Xs, Xt, eps=1e-6):
    Xs = _check_X(Xs, "Xs"); Xt = _check_X(Xt, "Xt")
    if Xs.shape[1] != Xt.shape[1]:
        raise ValueError("Xs 与 Xt 特征维度不一致。")
    ms = Xs.mean(0, keepdims=True); mt = Xt.mean(0, keepdims=True)
    Xs0 = Xs - ms; Xt0 = Xt - mt
    Ws = _cov_power(Xs0, power=-0.5, eps=eps)
    Wt = _cov_power(Xt0, power=+0.5, eps=eps)
    return {"ms": ms, "mt": mt, "Ws": Ws, "Wt": Wt}

def coral_transform(X, params):
    X = _check_X(X, "X")
    Z = (X - params["ms"]) @ params["Ws"] @ params["Wt"]
    return Z + params["mt"]

def coral_distance(Xa, Xb, eps=1e-6):
    Xa = _check_X(Xa, "Xa"); Xb = _check_X(Xb, "Xb")
    d = Xa.shape[1]
    Ca = np.cov(Xa - Xa.mean(0), rowvar=False) + eps*np.eye(d)
    Cb = np.cov(Xb - Xb.mean(0), rowvar=False) + eps*np.eye(d)
    return np.linalg.norm(Ca - Cb, ord="fro")

def _pairwise_sq_dists(A, B):
    AA = np.sum(A*A, axis=1, keepdims=True)
    BB = np.sum(B*B, axis=1, keepdims=True).T
    AB = A @ B.T
    return AA + BB - 2*AB

def mmd_rbf(Xa, Xb, gamma=None):
    Xa = _check_X(Xa, "Xa"); Xb = _check_X(Xb, "Xb")
    if gamma is None:
        Z = np.vstack([Xa, Xb])
        D = _pairwise_sq_dists(Z, Z)
        med = np.median(D[D > 0])
        if not np.isfinite(med) or med <= 0:
            med = 1.0
        gamma = 1.0 / (2.0 * med)
    Kaa = np.exp(-_pairwise_sq_dists(Xa, Xa) * gamma)
    Kbb = np.exp(-_pairwise_sq_dists(Xb, Xb) * gamma)
    Kab = np.exp(-_pairwise_sq_dists(Xa, Xb) * gamma)
    m = Xa.shape[0]; n = Xb.shape[0]
    mmd2 = (Kaa.sum() - np.trace(Kaa)) / (m*(m-1)) \
         + (Kbb.sum() - np.trace(Kbb)) / (n*(n-1)) \
         - 2.0 * Kab.mean()
    return max(mmd2, 0.0)

def proxy_A_distance(Xa, Xb, test_size=0.3, random_state=0):
    Xa = _check_X(Xa, "Xa"); Xb = _check_X(Xb, "Xb")
    X = np.vstack([Xa, Xb])
    y = np.hstack([np.zeros(len(Xa), dtype=int), np.ones(len(Xb), dtype=int)])
    rng = np.random.RandomState(random_state)
    idx = np.arange(len(X)); rng.shuffle(idx)
    split = int(len(X)*(1 - test_size))
    tr, te = idx[:split], idx[split:]
    clf = LogisticRegression(max_iter=500)
    clf.fit(X[tr], y[tr])
    e = (clf.predict(X[te]) != y[te]).mean()
    A = abs(2 * (1 - 2*e))
    return A, e


# ==================== 主流程（调用独立函数） ====================
def evaluate_and_plot(
    Xs, ys, Xt,
    eps=1e-5,
    clf=None,
    fig_prefix="figs/phase2",
    random_state=0,
    class_names=None,
    source_val_ratio=0.2
):
 
    # 检查 4 类
    classes = np.unique(ys)
    if len(classes) != 4:
        raise ValueError(f"ys 应该恰好包含 4 类，但检测到 {len(classes)} 类：{classes}")

    # 预处理（源域拟合 → 两域应用）
    pre = Pipeline([
        ("vth", VarianceThreshold(threshold=1e-8)),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])
    Xs_p = pre.fit_transform(Xs)
    Xt_p = pre.transform(Xt)

    # 源域分层划分（train/val）
    sss = StratifiedShuffleSplit(n_splits=1, test_size=source_val_ratio, random_state=random_state)
    (train_idx, val_idx) = next(sss.split(Xs_p, ys))
    Xs_tr, Xs_va = Xs_p[train_idx], Xs_p[val_idx]
    ys_tr, ys_va = ys[train_idx], ys[val_idx]

    # 分类器
    if clf is None:
        clf = RandomForestClassifier(n_estimators=400, random_state=random_state)

    # —— 源域：对齐前 —— #
    clf.fit(Xs_tr, ys_tr)
    yhat_src_before = clf.predict(Xs_va)
    acc_src_before = accuracy_score(ys_va, yhat_src_before)
    

    # —— CORAL 对齐：源域训练集 + 全量目标域 —— #
    params = coral_fit(Xs_tr, Xt_p, eps=eps)
    Xs_tr_aln = coral_transform(Xs_tr, params)
    Xs_va_aln = coral_transform(Xs_va, params)

    clf2 = RandomForestClassifier(n_estimators=400, random_state=random_state)
    clf2.fit(Xs_tr_aln, ys_tr)
    yhat_src_after = clf2.predict(Xs_va_aln)
    acc_src_after = accuracy_score(ys_va, yhat_src_after)
    

    # —— 指标（整域） —— #
    cov_before = coral_distance(Xs_p, Xt_p)
    cov_after  = coral_distance(coral_transform(Xs_p, coral_fit(Xs_p, Xt_p, eps=eps)), Xt_p)
    A_before, e_before = proxy_A_distance(Xs_p, Xt_p, random_state=random_state)
    A_after,  e_after  = proxy_A_distance(coral_transform(Xs_p, coral_fit(Xs_p, Xt_p, eps=eps)), Xt_p,
                                          random_state=random_state)
    mmd_before = mmd_rbf(Xs_p, Xt_p)
    mmd_after  = mmd_rbf(coral_transform(Xs_p, coral_fit(Xs_p, Xt_p, eps=eps)), Xt_p)

    yt_pred = clf2.predict(Xt_p)
    tes = pd.DataFrame(yt_pred)
    tes.to_csv('coral_predict')
    Xs_p_aln = coral_transform(Xs_p, coral_fit(Xs_p, Xt_p, eps=eps))

    metrics = {
        "源域正确率_对齐前": acc_src_before,
        "源域正确率_对齐后": acc_src_after,
        "源域正确率_变化(后-前)": acc_src_after - acc_src_before,
        "协方差距离_对齐前": cov_before,
        "协方差距离_对齐后": cov_after,
        "A距离_对齐前": A_before, "域分类测试误差_对齐前": e_before,
        "A距离_对齐后": A_after,  "域分类测试误差_对齐后": e_after,
        "MMD(RBF)_对齐前": mmd_before,
        "MMD(RBF)_对齐后": mmd_after,
    }

    return metrics

# ==================== CLI 示例 ====================
if __name__ == "__main__":
    # 例：Xs.csv（第3列为标签，5列起为特征），Xt.csv（全是特征）
    df_s = pd.read_csv("Xs.csv")
    df_t = pd.read_csv("Xt.csv")
    ys = df_s.iloc[:, 1].astype("category").cat.codes.to_numpy()
    Xs = df_s.iloc[:, 4:].to_numpy()
    Xt = df_t.to_numpy()

    # （可选）中文类名，按编码顺序
    class_names = ["正常", "内圈故障", "外圈故障", "滚动体故障"]

    out = evaluate_and_plot(
        Xs, ys, Xt,
        eps=1e-5,
        fig_prefix="figs/phase2",
        class_names=class_names,
        source_val_ratio=0.2,
        random_state=0
    )
    print(out)

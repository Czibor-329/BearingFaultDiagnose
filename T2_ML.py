# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, ConfusionMatrixDisplay
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.kernel_approximation import RBFSampler


def build_models():
    """Return dict[name->Pipeline] for the classifiers shown in the image."""
    return {
        # KNN（注意标准化）
        "KNeighborsClassifier": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5))
        ]),
        # 线性SVM
        "LinearSVC": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(random_state=42, max_iter=5000))
        ]),
        # 线性SGD（SVM hinge 损失）
        "SGDClassifier": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SGDClassifier(loss="hinge", max_iter=3000, tol=1e-3, random_state=42))
        ]),
        # 非线性SVM（RBF核，开启概率以便计算ROC-AUC）
        "SVC (RBF, prob=True)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=42))
        ]),
        # 集成：随机森林
        "RandomForest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=300, random_state=42))
        ]),
        # 集成：梯度提升
        "GradientBoosting": Pipeline([
            ("clf", GradientBoostingClassifier(random_state=42))
        ]),
        # Kernel Approximation + 线性分类（SGD hinge）
        "RBFSampler + SGD (hinge)": Pipeline([
            ("scaler", StandardScaler()),
            ("rbf", RBFSampler(gamma=0.02, random_state=42, n_components=1000)),
            ("clf", SGDClassifier(loss="hinge", max_iter=3000, tol=1e-3, random_state=42))
        ]),
    }


def evaluate(models, X_train, y_train, X_test, y_test):
    """Train/eval all models; return dataframe, predictions of best model, and best-model name."""
    records = []
    preds_by_model = {}

    for name, pipe in models.items():
        t0 = time.time()
        pipe.fit(X_train, y_train)
        train_time = time.time() - t0

        t1 = time.time()
        y_pred = pipe.predict(X_test)
        pred_time = time.time() - t1
        preds_by_model[name] = y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        # ROC-AUC (macro-ovr)
        roc = np.nan
        clf = pipe.named_steps[list(pipe.named_steps.keys())[-1]]
        try:
            if hasattr(clf, "predict_proba"):
                proba = pipe.predict_proba(X_test)
                roc = roc_auc_score(y_test, proba, multi_class="ovr")
            elif hasattr(clf, "decision_function"):
                dec = pipe.decision_function(X_test)
                roc = roc_auc_score(y_test, dec, multi_class="ovr")
        except Exception:
            pass

        records.append({
            "model": name,
            "accuracy": acc,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_macro": f1,
            "roc_auc_ovr": roc,
            "train_time_s": train_time,
            "predict_time_s": pred_time,
        })

    df_scores = pd.DataFrame(records).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    best_name = df_scores.iloc[0]["model"]
    return df_scores, preds_by_model[best_name], best_name


from viz import save_confusion_matrix_pdf

def main(output_dir="./figs/res2"):
    plt.rcParams['font.sans-serif'] = ['SimHei']   # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题
    
    # 1) 数据
    data = pd.read_csv('Xs.csv')
    X, y = data.iloc[:,4:], data.iloc[:,1].astype("category").cat.codes.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 2) 模型
    models = build_models()

    # 3) 训练与评估
    df_scores, y_pred_best, best_name = evaluate(models, X_train, y_train, X_test, y_test)

    # 4) 导出CSV
    csv_path = f"{output_dir}/classifier_metrics_digits.csv"
    df_scores.to_csv(csv_path, index=False)

    # 6) 最佳模型的混淆矩阵
    best_model = models[best_name]
    y_pred_best = best_model.predict(X_test)
    plt.figure(figsize=(6, 6))
    target_names = ['N','B','IR','OR']
    save_confusion_matrix_pdf(y_test,y_pred_best,target_names,f"{output_dir}/cm")
    
    return {
        "csv": csv_path,
        "best_model": best_name,
        "figs": [
            f"{output_dir}/accuracy_bar.png",
            f"{output_dir}/precision_macro_bar.png",
            f"{output_dir}/recall_macro_bar.png",
            f"{output_dir}/f1_macro_bar.png",
            f"{output_dir}/roc_auc_ovr_bar.png",
            f"{output_dir}/confusion_matrix_best.png",
        ]
    }


if __name__ == "__main__":
    out = main()
    print(out)

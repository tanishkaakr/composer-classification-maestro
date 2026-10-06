"""Step 2: train, tune (on validation only), evaluate (on test once), and analyse.

Usage:  python src/train.py
Reads:  data/features.csv
Writes: results/metrics.csv, results/confusion_matrix.png, results/ablation.csv,
        results/ablation.png, results/feature_importance.csv/.png, models/best_model.joblib
"""
import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
GROUPS = ["pitch_", "mel_", "vert_", "tex_", "dyn_", "rhy_"]

# name -> (estimator factory, hyper-parameter grid searched on the VALIDATION set)
MODELS = {
    "Logistic Regression": (lambda **p: LogisticRegression(max_iter=3000, class_weight="balanced", **p),
                            {"C": [0.01, 0.1, 1, 10]}),
    "SVM (RBF)": (lambda **p: SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED, **p),
                  {"C": [1, 10, 100], "gamma": ["scale", 0.003]}),
    "Gradient Boosting": (lambda **p: HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED, **p),
                          {"learning_rate": [0.05, 0.1], "max_depth": [3, None], "max_iter": [200]}),
    "MLP": (lambda **p: MLPClassifier(max_iter=800, random_state=SEED, **p),
            {"hidden_layer_sizes": [(64,), (128,)], "alpha": [1e-3, 1e-1]}),
}


def scores(y_true, y_pred):
    return {"acc": accuracy_score(y_true, y_pred),
            "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features.csv")
    args = ap.parse_args()
    for d in ("results", "models"):
        Path(d).mkdir(exist_ok=True)

    df = pd.read_csv(args.features)
    feat_cols = [c for c in df.columns if c.startswith(tuple(GROUPS))]
    tr, va, te = (df[df["split"] == s] for s in ("train", "validation", "test"))
    Xtr, ytr, Xva, yva, Xte, yte = tr[feat_cols], tr["composer"], va[feat_cols], va["composer"], te[feat_cols], te["composer"]
    classes = sorted(df["composer"].unique())
    print(f"{len(feat_cols)} features | train {len(tr)}, validation {len(va)}, test {len(te)} | {len(classes)} composers")

    # ---------- 1. tune each model on validation, then evaluate on test ----------
    rows, fitted = [], {}
    for name, (make, grid) in MODELS.items():
        best_f1, best_params, best_pipe = -1, None, None
        for params in ParameterGrid(grid):
            pipe = make_pipeline(StandardScaler(), make(**params)).fit(Xtr, ytr)
            f1 = f1_score(yva, pipe.predict(Xva), average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_params, best_pipe = f1, params, pipe
        s = scores(yte, best_pipe.predict(Xte))
        rows.append({"model": name, "val_macro_f1": best_f1, "train_acc": accuracy_score(ytr, best_pipe.predict(Xtr)),
                     **{f"test_{k}": v for k, v in s.items()}, "best_params": str(best_params)})
        fitted[name] = best_pipe
        print(f"{name:20s} val F1 {best_f1:.3f} | test acc {s['acc']:.3f}, macro-F1 {s['macro_f1']:.3f} | {best_params}")
    res = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    res.to_csv("results/metrics.csv", index=False)

    best_name = res.iloc[0]["model"]  # chosen by VALIDATION score, not test
    best = fitted[best_name]
    print(f"\nBest model (by validation macro-F1): {best_name}")

    # ---------- 2. confusion matrix on test ----------
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(yte, best.predict(Xte), labels=classes, normalize="true",
                                            xticks_rotation=45, ax=ax, colorbar=False, values_format=".2f")
    ax.set_title(f"{best_name}: test confusion matrix (row-normalised)")
    fig.tight_layout()
    fig.savefig("results/confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ---------- 3. ablation by feature group (default gradient boosting, test set) ----------
    abl = []
    for g in GROUPS + ["ALL"]:
        cols = feat_cols if g == "ALL" else [c for c in feat_cols if c.startswith(g)]
        m = make_pipeline(StandardScaler(), HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED))
        m.fit(tr[cols], ytr)
        s = scores(yte, m.predict(te[cols]))
        abl.append({"group": g.strip("_"), "n_features": len(cols), "test_acc": s["acc"], "test_macro_f1": s["macro_f1"]})
    abl = pd.DataFrame(abl)
    abl.to_csv("results/ablation.csv", index=False)
    print("\nAblation (feature group alone):\n", abl.round(3).to_string(index=False))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(abl["group"], abl["test_macro_f1"])
    ax.set_ylabel("test macro-F1")
    ax.set_title("Feature-group ablation (gradient boosting)")
    fig.tight_layout()
    fig.savefig("results/ablation.png", dpi=150)
    plt.close(fig)

    # ---------- 4. permutation feature importance (validation set, best model) ----------
    pi = permutation_importance(best, Xva, yva, scoring="f1_macro", n_repeats=5, random_state=SEED, n_jobs=-1)
    imp = pd.DataFrame({"feature": feat_cols, "importance": pi.importances_mean, "std": pi.importances_std})
    imp = imp.sort_values("importance", ascending=False)
    imp.to_csv("results/feature_importance.csv", index=False)
    top = imp.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top["feature"], top["importance"], xerr=top["std"])
    ax.set_xlabel("drop in validation macro-F1 when shuffled")
    ax.set_title(f"Top 20 features ({best_name})")
    fig.tight_layout()
    fig.savefig("results/feature_importance.png", dpi=150)
    plt.close(fig)
    print("\nTop 10 features:\n", imp.head(10).round(4).to_string(index=False))

    joblib.dump({"pipeline": best, "features": feat_cols, "classes": list(best.classes_), "name": best_name},
                "models/best_model.joblib")
    print("\nSaved model to models/best_model.joblib and plots/tables to results/")


if __name__ == "__main__":
    main()

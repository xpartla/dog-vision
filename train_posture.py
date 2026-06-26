"""Train and evaluate the posture classifier on a dataset from build_dataset.py.

Splits by *clip* (group), never by frame, so the reported scores reflect
generalization to unseen clips/views rather than memorized frames. Reports
grouped cross-validation, a held-out test confusion matrix, and feature
importances, then saves a model bundle that `posture.LearnedPostureClassifier`
loads at inference time.

Example:
    python train_posture.py dataset.npz --out posture_model.joblib
    python train_posture.py dataset.npz --model mlp
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(kind: str):
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=0,
        )
    if kind == "mlp":
        return Pipeline([
            ("scale", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=600,
                                  random_state=0)),
        ])
    raise ValueError(f"unknown model kind: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, help=".npz from build_dataset.py")
    parser.add_argument("--out", type=Path, default=Path("posture_model.joblib"))
    parser.add_argument("--model", choices=["rf", "mlp"], default="rf")
    parser.add_argument("--test-frac", type=float, default=0.25,
                        help="Fraction of *clips* held out for the test report")
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=True)
    X, y, groups = data["X"], data["y"], data["groups"]
    feature_names = list(data["feature_names"])
    conf_thr = float(data["confidence_threshold"]) if "confidence_threshold" in data else 0.5

    classes, counts = np.unique(y, return_counts=True)
    n_groups = len(np.unique(groups))
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {n_groups} clips")
    for c, n in zip(classes, counts):
        print(f"  {c:12} {n}")
    if len(classes) < 2:
        raise SystemExit("Need at least 2 posture classes to train.")
    if n_groups < args.cv_folds + 1:
        raise SystemExit(f"Only {n_groups} clips; need more for a meaningful split. "
                         f"Record more clips per posture (and vary the viewpoint).")

    # --- Grouped cross-validation (generalization across clips) ---
    folds = min(args.cv_folds, n_groups)
    gkf = GroupKFold(n_splits=folds)
    accs = []
    for tr, va in gkf.split(X, y, groups):
        m = build_model(args.model)
        m.fit(X[tr], y[tr])
        accs.append(m.score(X[va], y[va]))
    print(f"\nGrouped {folds}-fold CV accuracy: "
          f"{np.mean(accs):.3f} +/- {np.std(accs):.3f}")

    # --- Held-out test report (split by clip) ---
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=0)
    tr, te = next(gss.split(X, y, groups))
    model = build_model(args.model)
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te])
    print(f"\nHeld-out test ({len(np.unique(groups[te]))} clips, {len(te)} frames):")
    print(classification_report(y[te], pred, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    labels_sorted = sorted(classes)
    cm = confusion_matrix(y[te], pred, labels=labels_sorted)
    print("        " + "  ".join(f"{c[:7]:>7}" for c in labels_sorted))
    for c, rowv in zip(labels_sorted, cm):
        print(f"{c[:7]:>7} " + "  ".join(f"{v:>7}" for v in rowv))

    if args.model == "rf":
        importances = model.feature_importances_
        top = np.argsort(importances)[::-1][:15]
        print("\nTop 15 features:")
        for i in top:
            print(f"  {feature_names[i]:28} {importances[i]:.4f}")

    # --- Refit on ALL data and save ---
    final = build_model(args.model)
    final.fit(X, y)
    bundle = {
        "model": final,
        "classes": list(final.classes_),
        "feature_names": feature_names,
        "confidence_threshold": conf_thr,
        "model_kind": args.model,
        "feature_version": len(feature_names),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out)
    print(f"\nSaved model to {args.out.resolve()}")
    print("Use it with:  python classify_video.py <video> --posture-model "
          f"{args.out}")


if __name__ == "__main__":
    main()

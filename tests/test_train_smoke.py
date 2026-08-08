"""End-to-end training smoke test on a tiny synthetic dataset.

Three Gaussian class clusters split across 8 fake clips each — enough groups
for the grouped CV protocol — trained through the real train() entry point,
then loaded back through the real inference wrapper.
"""

from __future__ import annotations

import joblib
import numpy as np
import pytest
from conftest import make_frame

from dogvision import pose_features as pf
from dogvision.posture import POSTURE_LABELS, LearnedPostureClassifier
from dogvision.tools.train_posture import train

CLASSES = ("lying", "sitting", "standing")


def _write_synthetic_dataset(path, clips_per_class: int = 8, frames_per_clip: int = 12):
    rng = np.random.default_rng(0)
    X, y, groups = [], [], []
    for label in CLASSES:
        center = rng.normal(0.0, 1.0, pf.N_FEATURES)
        for clip in range(clips_per_class):
            clip_offset = rng.normal(0.0, 0.1, pf.N_FEATURES)
            for _ in range(frames_per_clip):
                X.append(center + clip_offset + rng.normal(0.0, 0.3, pf.N_FEATURES))
                y.append(label)
                groups.append(f"{label}-{clip:02d}")
    np.savez(
        path,
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y),
        groups=np.asarray(groups),
        feature_names=np.asarray(pf.FEATURE_NAMES),
        confidence_threshold=0.5,
    )


@pytest.mark.parametrize("model_kind", ["rf", "mlp"])
def test_train_writes_loadable_bundle(tmp_path, model_kind):
    dataset = tmp_path / "dataset.npz"
    _write_synthetic_dataset(dataset)
    out = tmp_path / f"model_{model_kind}.joblib"

    result = train(dataset, model_kind=model_kind, out_path=out)

    assert result == out
    assert out.exists()
    bundle = joblib.load(out)
    assert list(bundle["feature_names"]) == list(pf.FEATURE_NAMES)
    assert set(bundle["classes"]) == set(CLASSES)
    assert bundle["model_kind"] == model_kind
    assert bundle["confidence_threshold"] == 0.5

    # The freshly trained bundle must plug straight into the inference wrapper.
    clf = LearnedPostureClassifier(out)
    label, probability = clf.classify(make_frame("standing"))
    assert label in POSTURE_LABELS
    assert 0.0 <= probability <= 1.0


def test_train_rejects_single_class_dataset(tmp_path):
    rng = np.random.default_rng(1)
    dataset = tmp_path / "one_class.npz"
    n = 40
    np.savez(
        dataset,
        X=rng.normal(size=(n, pf.N_FEATURES)).astype(np.float32),
        y=np.asarray(["sitting"] * n),
        groups=np.asarray([f"clip-{i % 8}" for i in range(n)]),
        feature_names=np.asarray(pf.FEATURE_NAMES),
        confidence_threshold=0.5,
    )
    with pytest.raises(SystemExit):
        train(dataset, out_path=tmp_path / "m.joblib")


def test_train_rejects_too_few_clips(tmp_path):
    rng = np.random.default_rng(2)
    dataset = tmp_path / "few_clips.npz"
    n = 40
    np.savez(
        dataset,
        X=rng.normal(size=(n, pf.N_FEATURES)).astype(np.float32),
        y=np.asarray(["sitting", "standing"] * (n // 2)),
        groups=np.asarray([f"clip-{i % 3}" for i in range(n)]),
        feature_names=np.asarray(pf.FEATURE_NAMES),
        confidence_threshold=0.5,
    )
    with pytest.raises(SystemExit):
        train(dataset, out_path=tmp_path / "m.joblib", cv_folds=5)

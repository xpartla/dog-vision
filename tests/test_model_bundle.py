"""Executable guards for the shipped model bundles.

Turns the frozen-feature-order invariant and the "repo runs out of the box"
promise into red CI runs: every committed .joblib must load under the pinned
dependency set and agree with the current pose_features definition.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import pytest
from conftest import make_frame
from sklearn.exceptions import InconsistentVersionWarning

from dogvision import pose_features as pf
from dogvision.posture import POSTURE_LABELS, LearnedPostureClassifier

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATHS = sorted(MODELS_DIR.glob("*.joblib"))


def test_shipped_models_are_present():
    assert MODEL_PATHS, f"no .joblib bundles found in {MODELS_DIR}"


@pytest.mark.parametrize("path", MODEL_PATHS, ids=lambda p: p.name)
def test_bundle_loads_cleanly_and_matches_feature_names(path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bundle = joblib.load(path)
    # An sklearn version-mismatch warning here means the pin in pyproject.toml
    # and the environment that trained the bundle have drifted apart. (Other
    # warning categories — e.g. third-party deprecations — are not drift.)
    mismatches = [
        str(w.message) for w in caught if issubclass(w.category, InconsistentVersionWarning)
    ]
    assert not mismatches, f"loading {path.name} warned: {mismatches[:2]}"

    assert list(bundle["feature_names"]) == list(pf.FEATURE_NAMES)
    assert bundle["feature_version"] == pf.N_FEATURES
    assert set(bundle["classes"]) <= set(POSTURE_LABELS)


@pytest.mark.parametrize("path", MODEL_PATHS, ids=lambda p: p.name)
@pytest.mark.parametrize("pose", ("standing", "sitting", "lying"))
def test_bundle_classifies_factory_frames(path, pose):
    clf = LearnedPostureClassifier(path)
    label, probability = clf.classify(make_frame(pose))
    assert label in POSTURE_LABELS
    assert 0.0 <= probability <= 1.0


@pytest.mark.parametrize("path", MODEL_PATHS, ids=lambda p: p.name)
def test_bundle_returns_unknown_on_degenerate_frame(path):
    clf = LearnedPostureClassifier(path)
    keep = set(pf.COORD_KEYPOINTS[:2])
    frame = make_frame("standing", occlude=set(pf.COORD_KEYPOINTS) - keep)
    label, probability = clf.classify(frame)
    assert label == "unknown"
    assert probability == 0.0

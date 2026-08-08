"""Rule-based posture classifier on synthetic geometry.

The rules assume a roughly horizontal camera (tripod near ribcage height) —
the factory skeletons are built for exactly that assumption, so each pose must
classify correctly. Degraded frames must fall back to "unknown" rather than
guess.
"""

from __future__ import annotations

import pytest
from conftest import make_frame

from dogvision import pose_features as pf
from dogvision.posture import (
    POSTURE_LABELS,
    Frame,
    classify_posture,
    compute_posture_features,
)

POSES = ("standing", "sitting", "lying")


@pytest.mark.parametrize("pose", POSES)
def test_factory_poses_classify_correctly(pose):
    label, confidence = classify_posture(compute_posture_features(make_frame(pose)))
    assert label == pose
    assert confidence > 0.5


@pytest.mark.parametrize("pose", POSES)
def test_labels_are_from_the_declared_set(pose):
    label, _ = classify_posture(compute_posture_features(make_frame(pose)))
    assert label in POSTURE_LABELS


def test_near_empty_frame_is_unknown():
    # Three keypoints that feed none of the sit/stand/lie evidence (no hip,
    # no trunk pair, no hind-leg triplet).
    keep = {"nose", "left_eye", "tail_end"}
    f = make_frame("standing", occlude=set(pf.COORD_KEYPOINTS) - keep)
    label, confidence = classify_posture(compute_posture_features(f))
    assert label == "unknown"
    assert confidence == 0.0


def test_empty_frame_is_unknown():
    label, _ = classify_posture(compute_posture_features(Frame(keypoints={})))
    assert label == "unknown"


def test_low_confidence_frame_is_unknown():
    # All keypoints present but below the visibility threshold.
    f = make_frame("standing", confidence=0.1)
    label, _ = classify_posture(compute_posture_features(f))
    assert label == "unknown"

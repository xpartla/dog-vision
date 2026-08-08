"""Temporal smoothing: LabelSmoother, merge_short_segments, and the 1€ filter."""

from __future__ import annotations

import numpy as np

from dogvision.posture import (
    Frame,
    Keypoint,
    KeypointSmoother,
    LabelSmoother,
    _OneEuroFilter,
    merge_short_segments,
)

# --- LabelSmoother ----------------------------------------------------------


def test_majority_vote_over_window():
    s = LabelSmoother(window=3, suppress_unknown=False)
    assert s.push("sitting") == "sitting"
    s.push("sitting")
    assert s.push("standing") == "sitting"  # 2 sitting vs 1 standing
    assert s.push("standing") == "standing"  # window now [sit, stand, stand]


def test_suppress_unknown_replaces_with_last_stable():
    s = LabelSmoother(window=3, suppress_unknown=True)
    assert s.push("sitting") == "sitting"
    # A transient detection failure must not dilute the vote.
    assert s.push("unknown") == "sitting"
    assert s.push("unknown") == "sitting"


def test_unknown_passes_through_before_any_stable_label():
    s = LabelSmoother(window=3, suppress_unknown=True)
    assert s.push("unknown") == "unknown"


def test_no_suppression_lets_unknown_win_the_vote():
    s = LabelSmoother(window=3, suppress_unknown=False)
    s.push("sitting")
    s.push("unknown")
    assert s.push("unknown") == "unknown"


def test_window_one_is_passthrough():
    s = LabelSmoother(window=1)
    for label in ("sitting", "standing", "lying", "sitting"):
        assert s.push(label) == label


def test_reset_clears_state():
    s = LabelSmoother(window=3, suppress_unknown=True)
    s.push("sitting")
    s.reset()
    assert s.push("unknown") == "unknown"


# --- merge_short_segments ---------------------------------------------------


def test_unknown_gaps_are_forward_filled():
    labels = ["sitting", "unknown", "unknown", "sitting"]
    assert merge_short_segments(labels, min_length=1) == ["sitting"] * 4


def test_leading_unknowns_are_back_filled():
    labels = ["unknown", "unknown", "standing", "standing"]
    assert merge_short_segments(labels, min_length=1) == ["standing"] * 4


def test_short_segment_absorbed_by_larger_neighbour():
    labels = ["sitting"] * 10 + ["standing"] * 2 + ["lying"] * 20
    merged = merge_short_segments(labels, min_length=3)
    assert merged == ["sitting"] * 10 + ["lying"] * 22


def test_tie_prefers_left_neighbour():
    labels = ["sitting"] * 10 + ["standing"] * 2 + ["lying"] * 10
    merged = merge_short_segments(labels, min_length=3)
    assert merged == ["sitting"] * 12 + ["lying"] * 10


def test_short_edge_segment_absorbed():
    labels = ["standing"] * 2 + ["sitting"] * 30
    assert merge_short_segments(labels, min_length=5) == ["sitting"] * 32


def test_empty_input():
    assert merge_short_segments([], min_length=25) == []


def test_uniform_input_unchanged():
    assert merge_short_segments(["lying"] * 40, min_length=25) == ["lying"] * 40


def test_clip_shorter_than_min_length_terminates():
    # Regression: a single segment below min_length used to loop forever.
    assert merge_short_segments(["sitting"] * 3, min_length=25) == ["sitting"] * 3


def test_all_unknown_terminates():
    assert merge_short_segments(["unknown"] * 5, min_length=25) == ["unknown"] * 5


# --- 1€ filter / KeypointSmoother -------------------------------------------


def test_one_euro_first_sample_passes_through():
    f = _OneEuroFilter(fps=30.0)
    assert f.filter(123.4) == 123.4


def test_one_euro_reduces_jitter_variance():
    rng = np.random.default_rng(0)
    raw = 100.0 + rng.normal(0.0, 2.0, size=300)
    f = _OneEuroFilter(fps=30.0)
    smoothed = np.array([f.filter(v) for v in raw])
    # Skip the settle-in region, then the filtered signal must be calmer.
    assert smoothed[50:].var() < raw[50:].var()


def test_keypoint_smoother_first_frame_passes_through():
    ks = KeypointSmoother(fps=30.0)
    out = ks.smooth(Frame({"nose": Keypoint(100.0, 200.0, 0.9)}))
    assert out.keypoints["nose"].x == 100.0
    assert out.keypoints["nose"].y == 200.0


def test_low_confidence_keypoint_does_not_update_filter_state():
    ks = KeypointSmoother(fps=30.0)
    ks.smooth(Frame({"nose": Keypoint(100.0, 200.0, 0.9)}))

    # A wild low-confidence detection: emit last smoothed position, keep the
    # (low) confidence, and do not anchor the filter to the bad value.
    out2 = ks.smooth(Frame({"nose": Keypoint(999.0, 999.0, 0.1)}))
    assert out2.keypoints["nose"].x == 100.0
    assert out2.keypoints["nose"].y == 200.0
    assert out2.keypoints["nose"].confidence == 0.1

    # Recovery continues from the last good position, not from the outlier.
    out3 = ks.smooth(Frame({"nose": Keypoint(102.0, 202.0, 0.9)}))
    assert 100.0 <= out3.keypoints["nose"].x <= 102.0
    assert 200.0 <= out3.keypoints["nose"].y <= 202.0


def test_keypoint_smoother_reset():
    ks = KeypointSmoother(fps=30.0)
    ks.smooth(Frame({"nose": Keypoint(100.0, 200.0, 0.9)}))
    ks.reset()
    out = ks.smooth(Frame({"nose": Keypoint(500.0, 500.0, 0.9)}))
    assert out.keypoints["nose"].x == 500.0  # fresh passthrough, no memory

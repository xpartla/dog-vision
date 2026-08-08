"""DLC prediction-file round-trip: single- and multi-animal column layouts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dogvision.posture import list_keypoint_names, load_keypoint_frames

BODYPARTS = ["nose", "back_middle", "tail_base"]
COORDS = ["x", "y", "likelihood"]


def _single_animal_df(n_frames: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cols = pd.MultiIndex.from_product(
        [["test_scorer"], BODYPARTS, COORDS],
        names=["scorer", "bodyparts", "coords"],
    )
    data = np.empty((n_frames, len(cols)))
    for j, (_s, _bp, coord) in enumerate(cols):
        if coord == "likelihood":
            data[:, j] = rng.uniform(0.6, 1.0, n_frames)
        else:
            data[:, j] = rng.uniform(0.0, 640.0, n_frames)
    return pd.DataFrame(data, columns=cols)


def test_single_animal_roundtrip(tmp_path):
    df = _single_animal_df()
    path = tmp_path / "preds.h5"
    df.to_hdf(path, key="df_with_missing")

    frames = load_keypoint_frames(path, confidence_threshold=0.5)
    assert len(frames) == len(df)
    for i, frame in enumerate(frames):
        assert set(frame.keypoints) == set(BODYPARTS)
        for bp in BODYPARTS:
            kp = frame.keypoints[bp]
            assert kp.x == pytest.approx(df.iloc[i][("test_scorer", bp, "x")])
            assert kp.y == pytest.approx(df.iloc[i][("test_scorer", bp, "y")])
            assert kp.confidence == pytest.approx(df.iloc[i][("test_scorer", bp, "likelihood")])


def test_list_keypoint_names(tmp_path):
    path = tmp_path / "preds.h5"
    _single_animal_df().to_hdf(path, key="df_with_missing")
    assert list_keypoint_names(path) == BODYPARTS


def _multi_animal_df() -> pd.DataFrame:
    """Two individuals, two frames: dog_a is the confident detection in frame
    0, dog_b in frame 1. Positions are distinct so the choice is observable."""
    cols = pd.MultiIndex.from_product(
        [["test_scorer"], ["dog_a", "dog_b"], BODYPARTS, COORDS],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    df = pd.DataFrame(0.0, index=range(2), columns=cols)
    for bp_i, bp in enumerate(BODYPARTS):
        for frame in range(2):
            df.loc[frame, ("test_scorer", "dog_a", bp, "x")] = 100.0 + bp_i
            df.loc[frame, ("test_scorer", "dog_a", bp, "y")] = 110.0 + bp_i
            df.loc[frame, ("test_scorer", "dog_b", bp, "x")] = 500.0 + bp_i
            df.loc[frame, ("test_scorer", "dog_b", bp, "y")] = 510.0 + bp_i
    for bp in BODYPARTS:
        df.loc[0, ("test_scorer", "dog_a", bp, "likelihood")] = 0.95
        df.loc[0, ("test_scorer", "dog_b", bp, "likelihood")] = 0.60
        df.loc[1, ("test_scorer", "dog_a", bp, "likelihood")] = 0.55
        df.loc[1, ("test_scorer", "dog_b", bp, "likelihood")] = 0.90
    return df


def test_multi_animal_picks_highest_mean_likelihood_per_frame(tmp_path):
    path = tmp_path / "preds.h5"
    _multi_animal_df().to_hdf(path, key="df_with_missing")

    frames = load_keypoint_frames(path, confidence_threshold=0.5)
    assert len(frames) == 2
    # Frame 0 → dog_a's coordinates; frame 1 → dog_b's.
    assert frames[0].keypoints["nose"].x == pytest.approx(100.0)
    assert frames[0].keypoints["nose"].confidence == pytest.approx(0.95)
    assert frames[1].keypoints["nose"].x == pytest.approx(500.0)
    assert frames[1].keypoints["nose"].confidence == pytest.approx(0.90)


def test_nan_keypoints_are_dropped(tmp_path):
    df = _single_animal_df()
    df.iloc[0, df.columns.get_loc(("test_scorer", "nose", "x"))] = np.nan
    path = tmp_path / "preds.h5"
    df.to_hdf(path, key="df_with_missing")

    frames = load_keypoint_frames(path)
    assert "nose" not in frames[0].keypoints
    assert "nose" in frames[1].keypoints

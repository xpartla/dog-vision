"""Property tests for the viewpoint-robust feature vector.

These turn the README's invariance claims into executable guarantees:
translation/scale invariance, rotation invariance of every angle feature,
flip consistency (validating _FLIP_INDEX), and occlusion semantics.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_frame, mirror, rescale, rotate, translate
from hypothesis import given, settings
from hypothesis import strategies as st

from dogvision import pose_features as pf

POSES = ("standing", "sitting", "lying")

# --- Feature-vector layout, derived from the declared tables ----------------
# FEATURE_NAMES contains one duplicated name (two angle triplets share the
# back_middle vertex), so tests address slots positionally, mirroring the
# frozen block order: coords | angles | dists | belly | spine bow | globals.

_COORD0 = 0
_ANGLE0 = _COORD0 + 3 * len(pf.COORD_KEYPOINTS)
_DIST0 = _ANGLE0 + 2 * len(pf.ANGLE_TRIPLETS)
_BELLY0 = _DIST0 + 2 * len(pf.DIST_PAIRS)
_SPINE_BOW0 = _BELLY0 + 4
_BBOX_ASPECT = _SPINE_BOW0 + 2
_N_VISIBLE = _BBOX_ASPECT + 1


def coord_slots(kp: str) -> tuple[int, int, int]:
    k = pf.COORD_KEYPOINTS.index(kp)
    base = _COORD0 + 3 * k
    return base, base + 1, base + 2  # .x, .y, .vis


def angle_slots(triplet_index: int) -> tuple[int, int]:
    base = _ANGLE0 + 2 * triplet_index
    return base, base + 1  # value, .valid


def dist_slots(pair_index: int) -> tuple[int, int]:
    base = _DIST0 + 2 * pair_index
    return base, base + 1


def test_feature_layout_is_frozen():
    """The name list mirrors the declared block order exactly, and the vector
    is the 128 features the README and the shipped models are built on."""
    names = pf.FEATURE_NAMES
    i = 0
    for kp in pf.COORD_KEYPOINTS:
        assert names[i : i + 3] == [f"{kp}.x", f"{kp}.y", f"{kp}.vis"]
        i += 3
    for _a, b, _c in pf.ANGLE_TRIPLETS:
        assert names[i : i + 2] == [f"ang_{b}", f"ang_{b}.valid"]
        i += 2
    for name, _k1, _k2 in pf.DIST_PAIRS:
        assert names[i : i + 2] == [name, f"{name}.valid"]
        i += 2
    assert names[i : i + 4] == [
        "belly_front_paw_dist",
        "belly_front_paw_dist.valid",
        "belly_back_paw_dist",
        "belly_back_paw_dist.valid",
    ]
    i += 4
    assert names[i : i + 2] == ["spine_bow", "spine_bow.valid"]
    i += 2
    assert names[i : i + 2] == ["bbox_aspect_hw", "n_visible_frac"]
    i += 2
    assert i == len(names) == pf.N_FEATURES == 128


@pytest.mark.parametrize("pose", POSES)
def test_vector_length_matches_names(pose):
    vec = pf.feature_vector(make_frame(pose))
    assert vec is not None
    assert len(vec) == pf.N_FEATURES == len(pf.FEATURE_NAMES)


@pytest.mark.parametrize("pose", POSES)
def test_full_frame_has_all_valid_flags_set(pose):
    vec = pf.feature_vector(make_frame(pose))
    for kp in pf.COORD_KEYPOINTS:
        assert vec[coord_slots(kp)[2]] == 1.0
    for t in range(len(pf.ANGLE_TRIPLETS)):
        assert vec[angle_slots(t)[1]] == 1.0
    for d in range(len(pf.DIST_PAIRS)):
        assert vec[dist_slots(d)[1]] == 1.0
    assert vec[_BELLY0 + 1] == 1.0 and vec[_BELLY0 + 3] == 1.0
    assert vec[_SPINE_BOW0 + 1] == 1.0
    assert vec[_N_VISIBLE] == pytest.approx(1.0)


# --- Invariances ------------------------------------------------------------


@pytest.mark.parametrize("pose", POSES)
@given(
    dx=st.floats(-2000, 2000, allow_nan=False),
    dy=st.floats(-2000, 2000, allow_nan=False),
)
@settings(deadline=None)
def test_translation_invariance(pose, dx, dy):
    f = make_frame(pose)
    v0 = pf.feature_vector(f)
    v1 = pf.feature_vector(translate(f, dx, dy))
    np.testing.assert_allclose(v1, v0, atol=1e-4)


@pytest.mark.parametrize("pose", POSES)
@given(s=st.floats(0.05, 20.0, allow_nan=False))
@settings(deadline=None)
def test_scale_invariance(pose, s):
    f = make_frame(pose)
    v0 = pf.feature_vector(f)
    v1 = pf.feature_vector(rescale(f, s))
    np.testing.assert_allclose(v1, v0, atol=1e-4)


# Everything except the normalized coordinates and the axis-aligned bbox
# aspect is rotation-invariant by design.
_ROTATION_INVARIANT = np.array(
    [
        i
        for i, n in enumerate(pf.FEATURE_NAMES)
        if not (n.endswith(".x") or n.endswith(".y") or n == "bbox_aspect_hw")
    ],
    dtype=int,
)


@pytest.mark.parametrize("pose", POSES)
@given(deg=st.floats(0.0, 360.0, allow_nan=False))
@settings(deadline=None)
def test_rotation_invariance_of_angles_and_distances(pose, deg):
    f = make_frame(pose)
    v0 = pf.feature_vector(f)
    v1 = pf.feature_vector(rotate(f, deg))
    np.testing.assert_allclose(v1[_ROTATION_INVARIANT], v0[_ROTATION_INVARIANT], atol=1e-3)


def test_rotation_changes_normalized_coordinates():
    """Guards against the invariance test above passing vacuously: the
    coordinate features are *supposed* to change under rotation."""
    f = make_frame("standing")
    v0 = pf.feature_vector(f)
    v90 = pf.feature_vector(rotate(f, 90.0))
    coord_idx = np.array([i for i, n in enumerate(pf.FEATURE_NAMES) if n.endswith((".x", ".y"))])
    assert np.abs(v90[coord_idx] - v0[coord_idx]).max() > 0.5


@pytest.mark.parametrize("pose", POSES)
@given(occ=st.sets(st.sampled_from(pf.COORD_KEYPOINTS), max_size=8))
@settings(deadline=None)
def test_flip_consistency(pose, occ):
    """feature_vector(mirror(f)) == flip_feature_vector(feature_vector(f)) —
    the direct validation of _FLIP_INDEX, including under asymmetric
    occlusion, where a wrong swap table would be most visible."""
    f = make_frame(pose, occlude=occ)
    v = pf.feature_vector(f)
    vm = pf.feature_vector(mirror(f))
    assert v is not None and vm is not None
    np.testing.assert_allclose(vm, pf.flip_feature_vector(v), atol=1e-4)


def test_flip_is_involution():
    v = pf.feature_vector(make_frame("sitting"))
    np.testing.assert_allclose(pf.flip_feature_vector(pf.flip_feature_vector(v)), v, atol=0)


# --- Occlusion semantics ----------------------------------------------------


@pytest.mark.parametrize("kp", pf.COORD_KEYPOINTS)
def test_occlusion_zeroes_exactly_the_dependent_slots(kp):
    vec = pf.feature_vector(make_frame("standing", occlude={kp}))
    assert vec is not None

    x_i, y_i, vis_i = coord_slots(kp)
    assert vec[x_i] == 0.0 and vec[y_i] == 0.0 and vec[vis_i] == 0.0

    for t, (a, b, c) in enumerate(pf.ANGLE_TRIPLETS):
        val_i, valid_i = angle_slots(t)
        if kp in (a, b, c):
            assert vec[val_i] == 0.0 and vec[valid_i] == 0.0
        else:
            assert vec[valid_i] == 1.0

    for d, (_name, k1, k2) in enumerate(pf.DIST_PAIRS):
        _val_i, valid_i = dist_slots(d)
        assert vec[valid_i] == (0.0 if kp in (k1, k2) else 1.0)

    # Belly-to-paw averages stay valid while the belly and at least one paw of
    # the pair are visible; only losing the belly kills both.
    belly_front_valid = vec[_BELLY0 + 1]
    belly_back_valid = vec[_BELLY0 + 3]
    if kp == "belly_bottom":
        assert belly_front_valid == 0.0 and belly_back_valid == 0.0
    else:
        assert belly_front_valid == 1.0 and belly_back_valid == 1.0

    spine_bow_valid = vec[_SPINE_BOW0 + 1]
    assert spine_bow_valid == (0.0 if kp in pf._SPINE_BOW_KPS else 1.0)

    expected_frac = (len(pf.COORD_KEYPOINTS) - 1) / len(pf.COORD_KEYPOINTS)
    assert vec[_N_VISIBLE] == pytest.approx(expected_frac, abs=1e-6)


def test_too_few_visible_keypoints_returns_none():
    keep = set(pf.COORD_KEYPOINTS[: pf.MIN_VISIBLE_KEYPOINTS - 1])  # 3 visible
    f = make_frame("standing", occlude=set(pf.COORD_KEYPOINTS) - keep)
    assert pf.feature_vector(f) is None


def test_min_visible_boundary_produces_vector():
    keep = set(pf.COORD_KEYPOINTS[: pf.MIN_VISIBLE_KEYPOINTS])  # exactly 4
    f = make_frame("standing", occlude=set(pf.COORD_KEYPOINTS) - keep)
    assert pf.feature_vector(f) is not None


def test_coincident_keypoints_return_none():
    f = make_frame(
        "standing",
        overrides={kp: (100.0, 200.0) for kp in pf.COORD_KEYPOINTS},
    )
    assert pf.feature_vector(f) is None

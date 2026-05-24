"""Viewpoint-robust feature extraction from SuperAnimal-Quadruped keypoints.

This is the shared bridge between training (`build_dataset.py`) and inference
(`LearnedPostureClassifier` in `posture.py`). Both must produce the *identical*
feature vector, so the logic lives here once.

Design goals (learned from the rule-based classifier's failure on an elevated
front-on camera, where image-y stopped meaning "height"):

* **Translation + scale invariant.** Keypoints are centered on the visible-point
  centroid and divided by their RMS radius, so the dog's position in frame and
  distance from the camera don't matter.
* **No hard-coded "up" axis.** We feed normalized (x, y) coordinates and a rich
  set of joint angles, and let the model learn which cues separate the postures
  across whatever camera angles appear in the training data — rather than
  asserting that higher-in-frame == higher-off-the-ground.
* **Occlusion-aware.** Every keypoint and every angle carries a visibility/valid
  flag, and missing values are zero-filled. Which keypoints drop out is itself
  informative (a lying dog hides its paws differently than a standing one).
* **Left/right symmetric** when combined with horizontal-flip augmentation in the
  dataset builder (see `flip_feature_vector`).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from posture import (
    Frame,
    KP_NOSE, KP_LEFT_EYE, KP_RIGHT_EYE, KP_LEFT_EAR_BASE, KP_RIGHT_EAR_BASE,
    KP_NECK_BASE, KP_BACK_BASE, KP_BACK_MIDDLE, KP_BACK_END,
    KP_TAIL_BASE, KP_TAIL_END,
    KP_FRONT_LEFT_THIGH, KP_FRONT_RIGHT_THIGH,
    KP_FRONT_LEFT_KNEE, KP_FRONT_RIGHT_KNEE,
    KP_FRONT_LEFT_PAW, KP_FRONT_RIGHT_PAW,
    KP_BACK_LEFT_THIGH, KP_BACK_RIGHT_THIGH,
    KP_BACK_LEFT_KNEE, KP_BACK_RIGHT_KNEE,
    KP_BACK_LEFT_PAW, KP_BACK_RIGHT_PAW,
    _angle_at,
)

# Extra SuperAnimal bodyparts not promoted to constants in posture.py but useful
# as model inputs (the dog-relevant ones; antlers are skipped).
KP_UPPER_JAW = "upper_jaw"
KP_LOWER_JAW = "lower_jaw"
KP_NECK_END = "neck_end"
KP_THROAT_BASE = "throat_base"
KP_BELLY_BOTTOM = "belly_bottom"
KP_BODY_MIDDLE_LEFT = "body_middle_left"
KP_BODY_MIDDLE_RIGHT = "body_middle_right"

# Ordered keypoints whose normalized (x, y, visible) become features. Order is
# frozen — changing it invalidates saved models, so append, never reorder.
COORD_KEYPOINTS: tuple[str, ...] = (
    KP_NOSE, KP_UPPER_JAW, KP_LOWER_JAW, KP_LEFT_EYE, KP_RIGHT_EYE,
    KP_LEFT_EAR_BASE, KP_RIGHT_EAR_BASE,
    KP_NECK_BASE, KP_NECK_END, KP_THROAT_BASE,
    KP_BACK_BASE, KP_BACK_MIDDLE, KP_BACK_END, KP_TAIL_BASE, KP_TAIL_END,
    KP_BELLY_BOTTOM, KP_BODY_MIDDLE_LEFT, KP_BODY_MIDDLE_RIGHT,
    KP_FRONT_LEFT_THIGH, KP_FRONT_LEFT_KNEE, KP_FRONT_LEFT_PAW,
    KP_FRONT_RIGHT_THIGH, KP_FRONT_RIGHT_KNEE, KP_FRONT_RIGHT_PAW,
    KP_BACK_LEFT_THIGH, KP_BACK_LEFT_KNEE, KP_BACK_LEFT_PAW,
    KP_BACK_RIGHT_THIGH, KP_BACK_RIGHT_KNEE, KP_BACK_RIGHT_PAW,
)

# Joint-angle triplets (vertex is the middle point). Angles are translation,
# scale and rotation invariant — the most viewpoint-robust signal available.
ANGLE_TRIPLETS: tuple[tuple[str, str, str], ...] = (
    (KP_BACK_LEFT_THIGH, KP_BACK_LEFT_KNEE, KP_BACK_LEFT_PAW),     # L hind stifle
    (KP_BACK_RIGHT_THIGH, KP_BACK_RIGHT_KNEE, KP_BACK_RIGHT_PAW),  # R hind stifle
    (KP_FRONT_LEFT_THIGH, KP_FRONT_LEFT_KNEE, KP_FRONT_LEFT_PAW),  # L fore
    (KP_FRONT_RIGHT_THIGH, KP_FRONT_RIGHT_KNEE, KP_FRONT_RIGHT_PAW),  # R fore
    (KP_NECK_BASE, KP_BACK_BASE, KP_BACK_MIDDLE),                  # spine bend front
    (KP_BACK_BASE, KP_BACK_MIDDLE, KP_BACK_END),                  # spine bend rear
    (KP_NECK_BASE, KP_BACK_MIDDLE, KP_TAIL_BASE),                 # spine overall
    (KP_NOSE, KP_NECK_BASE, KP_BACK_BASE),                        # neck/head angle
)

# Left/right keypoint pairs, used to mirror a feature vector for flip augmentation.
_LR_PAIRS: tuple[tuple[str, str], ...] = (
    (KP_LEFT_EYE, KP_RIGHT_EYE),
    (KP_LEFT_EAR_BASE, KP_RIGHT_EAR_BASE),
    (KP_BODY_MIDDLE_LEFT, KP_BODY_MIDDLE_RIGHT),
    (KP_FRONT_LEFT_THIGH, KP_FRONT_RIGHT_THIGH),
    (KP_FRONT_LEFT_KNEE, KP_FRONT_RIGHT_KNEE),
    (KP_FRONT_LEFT_PAW, KP_FRONT_RIGHT_PAW),
    (KP_BACK_LEFT_THIGH, KP_BACK_RIGHT_THIGH),
    (KP_BACK_LEFT_KNEE, KP_BACK_RIGHT_KNEE),
    (KP_BACK_LEFT_PAW, KP_BACK_RIGHT_PAW),
)


def _build_feature_names() -> list[str]:
    names: list[str] = []
    for kp in COORD_KEYPOINTS:
        names += [f"{kp}.x", f"{kp}.y", f"{kp}.vis"]
    for a, b, c in ANGLE_TRIPLETS:
        names += [f"ang_{b}", f"ang_{b}.valid"]
    names += ["bbox_aspect_hw", "n_visible_frac"]
    return names


FEATURE_NAMES: list[str] = _build_feature_names()
N_FEATURES: int = len(FEATURE_NAMES)

MIN_VISIBLE_KEYPOINTS = 4


def feature_vector(frame: Frame) -> Optional[np.ndarray]:
    """Return the fixed-length feature vector for a frame, or None if too few
    keypoints are visible to normalize reliably."""
    visible = frame.visible()
    if len(visible) < MIN_VISIBLE_KEYPOINTS:
        return None

    xs = np.array([kp.x for kp in visible.values()], dtype=float)
    ys = np.array([kp.y for kp in visible.values()], dtype=float)
    cx, cy = float(xs.mean()), float(ys.mean())
    # RMS radius about the centroid — a rotation-invariant, outlier-tolerant scale.
    scale = float(np.sqrt(((xs - cx) ** 2 + (ys - cy) ** 2).mean()))
    if scale < 1e-3:
        return None

    feats: list[float] = []
    for kp_name in COORD_KEYPOINTS:
        kp = frame.get(kp_name)
        if kp is None:
            feats += [0.0, 0.0, 0.0]
        else:
            feats += [(kp.x - cx) / scale, (kp.y - cy) / scale, 1.0]

    for a_name, b_name, c_name in ANGLE_TRIPLETS:
        a, b, c = frame.get(a_name), frame.get(b_name), frame.get(c_name)
        if a and b and c:
            ang = _angle_at(b, a, c)
            if math.isnan(ang):
                feats += [0.0, 0.0]
            else:
                feats += [ang / 180.0, 1.0]
        else:
            feats += [0.0, 0.0]

    w = float(xs.max() - xs.min())
    h = float(ys.max() - ys.min())
    feats.append(h / w if w > 1e-3 else 0.0)
    feats.append(len(visible) / float(len(COORD_KEYPOINTS)))

    return np.asarray(feats, dtype=np.float32)


# Precomputed index map for flip augmentation, built once.
def _build_flip_index() -> np.ndarray:
    name_to_idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    swap = {}
    for l, r in _LR_PAIRS:
        for suf in (".x", ".y", ".vis"):
            swap[name_to_idx[f"{l}{suf}"]] = name_to_idx[f"{r}{suf}"]
            swap[name_to_idx[f"{r}{suf}"]] = name_to_idx[f"{l}{suf}"]
    # Angle features: swap the left/right limb angles too.
    angle_lr = {
        f"ang_{KP_BACK_LEFT_KNEE}": f"ang_{KP_BACK_RIGHT_KNEE}",
        f"ang_{KP_FRONT_LEFT_KNEE}": f"ang_{KP_FRONT_RIGHT_KNEE}",
    }
    for ln, rn in angle_lr.items():
        for suf in ("", ".valid"):
            swap[name_to_idx[ln + suf]] = name_to_idx[rn + suf]
            swap[name_to_idx[rn + suf]] = name_to_idx[ln + suf]
    idx = np.arange(N_FEATURES)
    for src, dst in swap.items():
        idx[src] = dst
    return idx


_FLIP_INDEX = _build_flip_index()
# x-coordinate feature positions (negated on horizontal flip).
_X_INDICES = np.array(
    [i for i, n in enumerate(FEATURE_NAMES) if n.endswith(".x")], dtype=int
)


def flip_feature_vector(vec: np.ndarray) -> np.ndarray:
    """Horizontally-mirrored copy of a feature vector (data augmentation).

    Swaps left/right keypoints & limb angles and negates normalized x. A dog
    facing left vs. right is the same posture, so training on both halves the
    orientation the model has to learn."""
    flipped = vec[_FLIP_INDEX].copy()
    flipped[_X_INDICES] *= -1.0
    return flipped

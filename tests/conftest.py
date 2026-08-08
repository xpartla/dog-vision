"""Synthetic Frame factory and geometry helpers for the test suite.

The factory produces plausible *side-view* dog skeletons (facing left, image y
grows downward, ground at the bottom) covering every keypoint in
`pose_features.COORD_KEYPOINTS`. Coordinates are hand-tuned so the rule-based
classifier — which assumes a roughly horizontal camera — labels each pose
correctly; the invariance tests then transform these frames freely.
"""

from __future__ import annotations

import math

from dogvision import pose_features
from dogvision.posture import Frame, Keypoint

DEFAULT_CONFIDENCE = 0.9

# name -> (x, y) in pixels; side view facing left, ground near y=490.
_STANDING: dict[str, tuple[float, float]] = {
    "nose": (140, 330),
    "upper_jaw": (150, 345),
    "lower_jaw": (155, 360),
    "left_eye": (170, 300),
    "right_eye": (185, 302),
    "left_earbase": (200, 268),
    "right_earbase": (215, 272),
    "neck_base": (250, 300),
    "neck_end": (240, 315),
    "throat_base": (225, 350),
    "back_base": (300, 292),
    "back_middle": (390, 288),
    "back_end": (470, 292),
    "tail_base": (500, 300),
    "tail_end": (560, 255),
    "belly_bottom": (385, 405),
    "body_middle_left": (388, 345),
    "body_middle_right": (395, 348),
    "front_left_thai": (245, 355),
    "front_left_knee": (240, 425),
    "front_left_paw": (235, 488),
    "front_right_thai": (260, 358),
    "front_right_knee": (255, 428),
    "front_right_paw": (252, 490),
    "back_left_thai": (495, 355),
    "back_left_knee": (500, 432),
    "back_left_paw": (520, 489),
    "back_right_thai": (508, 358),
    "back_right_knee": (512, 434),
    "back_right_paw": (532, 490),
}

# Rear lowered onto the ground, spine sloping up to a raised head, hind legs
# folded sharply, forelegs straight.
_SITTING: dict[str, tuple[float, float]] = {
    "nose": (255, 235),
    "upper_jaw": (262, 250),
    "lower_jaw": (266, 262),
    "left_eye": (283, 228),
    "right_eye": (295, 231),
    "left_earbase": (305, 222),
    "right_earbase": (318, 226),
    "neck_base": (300, 285),
    "neck_end": (310, 300),
    "throat_base": (285, 300),
    "back_base": (330, 320),
    "back_middle": (400, 372),
    "back_end": (462, 432),
    "tail_base": (478, 448),
    "tail_end": (540, 470),
    "belly_bottom": (360, 420),
    "body_middle_left": (395, 390),
    "body_middle_right": (405, 393),
    "front_left_thai": (300, 360),
    "front_left_knee": (295, 425),
    "front_left_paw": (292, 488),
    "front_right_thai": (315, 362),
    "front_right_knee": (310, 427),
    "front_right_paw": (308, 490),
    "back_left_thai": (470, 430),
    "back_left_knee": (505, 462),
    "back_left_paw": (468, 486),
    "back_right_thai": (482, 433),
    "back_right_knee": (515, 465),
    "back_right_paw": (480, 488),
}

# Sternal lie: spine flat just above the ground, head lowered, legs folded.
_LYING: dict[str, tuple[float, float]] = {
    "nose": (160, 458),
    "upper_jaw": (165, 465),
    "lower_jaw": (168, 470),
    "left_eye": (182, 440),
    "right_eye": (194, 443),
    "left_earbase": (205, 425),
    "right_earbase": (218, 428),
    "neck_base": (250, 428),
    "neck_end": (240, 438),
    "throat_base": (225, 455),
    "back_base": (300, 432),
    "back_middle": (380, 436),
    "back_end": (455, 433),
    "tail_base": (485, 438),
    "tail_end": (550, 445),
    "belly_bottom": (370, 468),
    "body_middle_left": (378, 448),
    "body_middle_right": (388, 450),
    "front_left_thai": (280, 450),
    "front_left_knee": (262, 468),
    "front_left_paw": (285, 477),
    "front_right_thai": (295, 452),
    "front_right_knee": (275, 470),
    "front_right_paw": (298, 479),
    "back_left_thai": (460, 452),
    "back_left_knee": (492, 466),
    "back_left_paw": (452, 478),
    "back_right_thai": (472, 455),
    "back_right_knee": (502, 469),
    "back_right_paw": (462, 480),
}

_POSES = {"standing": _STANDING, "sitting": _SITTING, "lying": _LYING}

assert all(set(p) == set(pose_features.COORD_KEYPOINTS) for p in _POSES.values()), (
    "factory poses must cover exactly the COORD_KEYPOINTS set"
)


def make_frame(
    posture: str = "standing",
    occlude: set[str] | frozenset[str] = frozenset(),
    overrides: dict[str, tuple[float, float]] | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
    confidence_threshold: float = 0.5,
) -> Frame:
    """Build a synthetic Frame for one of the factory postures.

    `occlude` names get confidence 0.0 (below any sane threshold), matching how
    a real detection failure looks to the pipeline. `overrides` replaces
    individual keypoint positions.
    """
    coords = dict(_POSES[posture])
    if overrides:
        coords.update(overrides)
    keypoints = {
        name: Keypoint(x=float(x), y=float(y), confidence=0.0 if name in occlude else confidence)
        for name, (x, y) in coords.items()
    }
    return Frame(keypoints=keypoints, confidence_threshold=confidence_threshold)


# --- Pure geometric transforms on Frame ------------------------------------


def _map_frame(frame: Frame, fn) -> Frame:
    return Frame(
        keypoints={
            name: Keypoint(*fn(kp.x, kp.y), kp.confidence) for name, kp in frame.keypoints.items()
        },
        confidence_threshold=frame.confidence_threshold,
    )


def translate(frame: Frame, dx: float, dy: float) -> Frame:
    return _map_frame(frame, lambda x, y: (x + dx, y + dy))


def rescale(frame: Frame, s: float) -> Frame:
    return _map_frame(frame, lambda x, y: (x * s, y * s))


def _centroid(frame: Frame) -> tuple[float, float]:
    xs = [kp.x for kp in frame.keypoints.values()]
    ys = [kp.y for kp in frame.keypoints.values()]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def rotate(frame: Frame, deg: float) -> Frame:
    """Rotate all keypoints about the frame centroid."""
    cx, cy = _centroid(frame)
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))

    def rot(x: float, y: float) -> tuple[float, float]:
        px, py = x - cx, y - cy
        return cx + c * px - s * py, cy + s * px + c * py

    return _map_frame(frame, rot)


def mirror(frame: Frame) -> Frame:
    """Horizontal mirror: negate x about the centroid AND swap left/right
    keypoint names (per pose_features._LR_PAIRS) — the physically-consistent
    flip a camera would see."""
    swap: dict[str, str] = {}
    for left, right in pose_features._LR_PAIRS:
        swap[left] = right
        swap[right] = left
    cx, _ = _centroid(frame)
    return Frame(
        keypoints={
            swap.get(name, name): Keypoint(2.0 * cx - kp.x, kp.y, kp.confidence)
            for name, kp in frame.keypoints.items()
        },
        confidence_threshold=frame.confidence_threshold,
    )

"""Camera-relative orientation estimate from 2-D projected keypoints.

Two orthogonal signals:
  spine direction   — angle of the tail→neck vector in image space
  bilateral spread  — distance between ear/eye pairs relative to spine length;
                      large when the dog faces the camera, small in profile.

OrientationResult drives the compass widget in overlay.py and gates the
head-tilt estimate in posture.py (tilt is only meaningful when the dog is
at least partially face-on).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from posture import (
    Frame,
    KP_BACK_BASE, KP_BACK_MIDDLE, KP_BACK_END,
    KP_NECK_BASE,
    KP_LEFT_EAR_BASE, KP_RIGHT_EAR_BASE,
    KP_LEFT_EYE, KP_RIGHT_EYE,
)


@dataclass(frozen=True)
class OrientationResult:
    """Camera-relative orientation estimate for one frame.

    spine_angle_deg : direction of tail→neck in image space
                      (0° = right, 90° = downward in image, ±180° = left)
    bilateral_conf  : 0→1; 1 = fully face-on (ears spread, spine foreshortened)
                           0 = pure profile  (ears overlap, spine at full length)
    spine_len_ratio : complement — how much of the spine projects in-plane
                      (1 = pure profile, 0 = face-on / fully foreshortened)
    confidence      : overall signal quality (0 = too few keypoints to estimate)
    """
    spine_angle_deg: float
    bilateral_conf: float
    spine_len_ratio: float
    confidence: float


_UNKNOWN = OrientationResult(
    spine_angle_deg=0.0, bilateral_conf=0.0, spine_len_ratio=0.0, confidence=0.0
)


def estimate_orientation(frame: Frame) -> OrientationResult:
    """Estimate camera-relative orientation from 2-D projected keypoints."""
    neck = frame.get(KP_NECK_BASE)
    tail = (
        frame.get(KP_BACK_END)
        or frame.get(KP_BACK_MIDDLE)
        or frame.get(KP_BACK_BASE)
    )

    spine_angle_deg = 0.0
    spine_len_2d = 0.0
    spine_ok = False

    if neck and tail:
        dx = neck.x - tail.x
        dy = neck.y - tail.y
        spine_len_2d = math.hypot(dx, dy)
        if spine_len_2d > 5.0:
            spine_angle_deg = math.degrees(math.atan2(dy, dx))
            spine_ok = True

    le = frame.get(KP_LEFT_EAR_BASE)
    re = frame.get(KP_RIGHT_EAR_BASE)
    if not (le and re):
        le = frame.get(KP_LEFT_EYE)
        re = frame.get(KP_RIGHT_EYE)

    ear_sep = 0.0
    bilateral_ok = False
    if le and re:
        ear_sep = math.hypot(le.x - re.x, le.y - re.y)
        bilateral_ok = True

    if not spine_ok and not bilateral_ok:
        return _UNKNOWN

    total = spine_len_2d + ear_sep + 1e-6
    bilateral_conf = ear_sep / total
    spine_len_ratio = spine_len_2d / total

    if spine_ok and bilateral_ok:
        conf = 1.0
    elif spine_ok:
        conf = 0.5
    else:
        conf = 0.2

    return OrientationResult(
        spine_angle_deg=spine_angle_deg,
        bilateral_conf=bilateral_conf,
        spine_len_ratio=spine_len_ratio,
        confidence=conf,
    )

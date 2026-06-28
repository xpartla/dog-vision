"""Camera-relative orientation estimate from 2-D projected keypoints.

Two orthogonal signals:
  spine direction   — principal axis of all visible spine keypoints (PCA)
  bilateral spread  — distance between ear/eye pairs relative to spine length;
                      large when the dog faces the camera, small in profile.

OrientationResult drives the compass widget in overlay.py and gates the
head-tilt estimate in posture.py (tilt is only meaningful when the dog is
at least partially face-on).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from posture import (
    Frame,
    KP_NOSE, KP_NECK_BASE,
    KP_BACK_BASE, KP_BACK_MIDDLE, KP_BACK_END,
    KP_TAIL_BASE, KP_TAIL_END,
    KP_LEFT_EAR_BASE, KP_RIGHT_EAR_BASE,
    KP_LEFT_EYE, KP_RIGHT_EYE,
)

# Spine keypoints in anatomical order (head end → tail end).
# PCA over all visible members is far more stable than a single 2-point vector.
_SPINE_KPS = [
    KP_NOSE, KP_NECK_BASE, KP_BACK_BASE, KP_BACK_MIDDLE,
    KP_BACK_END, KP_TAIL_BASE, KP_TAIL_END,
]


@dataclass(frozen=True)
class OrientationResult:
    """Camera-relative orientation estimate for one frame.

    spine_angle_deg : direction of tail→head in image space
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
    """Estimate camera-relative orientation from 2-D projected keypoints.

    Uses PCA over all available spine keypoints for a more robust direction
    estimate than a single two-point vector.  The principal axis is signed so
    it always points from tail toward head.
    """
    spine_pts = [frame.get(n) for n in _SPINE_KPS]
    spine_pts = [p for p in spine_pts if p is not None]

    spine_angle_deg = 0.0
    spine_len_2d = 0.0
    spine_ok = False

    if len(spine_pts) >= 2:
        coords = np.array([[p.x, p.y] for p in spine_pts], dtype=float)

        # PCA: principal axis via SVD of the mean-centred point cloud.
        centroid = coords.mean(axis=0)
        centered = coords - centroid
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        direction = vt[0]  # unit vector along principal axis; sign is ambiguous

        # Sign convention: direction must point from tail toward head.
        # coords[0] is the head-most detected point; coords[-1] is tail-most
        # (because _SPINE_KPS is ordered head→tail).  ref_vec points head→tail,
        # so we negate direction when it aligns with ref_vec.
        ref_vec = coords[-1] - coords[0]
        if np.dot(direction, ref_vec) > 0:
            direction = -direction

        spine_angle_deg = math.degrees(math.atan2(direction[1], direction[0]))

        # Spine length: span of spine points projected onto the principal axis.
        proj = centered @ direction
        spine_len_2d = float(proj.max() - proj.min())
        spine_ok = spine_len_2d > 5.0

    # Bilateral spread: ear bases, or eyes if ear bases are missing.
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


class OrientationSmoother:
    """Temporally smooth orientation estimates using circular EMA.

    Angles are smoothed in unit-circle space (cos/sin components) so that the
    ±180° wrap-around never causes discontinuous jumps.  bilateral_conf and
    spine_len_ratio are smoothed with a separate EMA.
    """

    def __init__(self, angle_alpha: float = 0.1, conf_alpha: float = 0.2):
        """
        angle_alpha : EMA weight given to each new angle sample (lower = smoother).
        conf_alpha  : EMA weight for bilateral_conf and spine_len_ratio.
        """
        self.angle_alpha = angle_alpha
        self.conf_alpha = conf_alpha
        self._cos: Optional[float] = None
        self._sin: Optional[float] = None
        self._bilateral: Optional[float] = None
        self._spine_ratio: Optional[float] = None

    def smooth(self, result: OrientationResult) -> OrientationResult:
        if result.confidence < 0.1:
            # No valid signal — return last known smoothed value if available.
            if self._cos is not None:
                angle = math.degrees(math.atan2(self._sin, self._cos))
                return OrientationResult(
                    spine_angle_deg=angle,
                    bilateral_conf=self._bilateral,
                    spine_len_ratio=self._spine_ratio,
                    confidence=result.confidence,
                )
            return result

        angle_rad = math.radians(result.spine_angle_deg)
        c, s = math.cos(angle_rad), math.sin(angle_rad)

        if self._cos is None:
            self._cos, self._sin = c, s
            self._bilateral = result.bilateral_conf
            self._spine_ratio = result.spine_len_ratio
        else:
            a, b = self.angle_alpha, self.conf_alpha
            self._cos = a * c + (1.0 - a) * self._cos
            self._sin = a * s + (1.0 - a) * self._sin
            self._bilateral = b * result.bilateral_conf + (1.0 - b) * self._bilateral
            self._spine_ratio = b * result.spine_len_ratio + (1.0 - b) * self._spine_ratio

        mag = math.hypot(self._cos, self._sin)
        smooth_angle = (
            math.degrees(math.atan2(self._sin / mag, self._cos / mag))
            if mag > 1e-6
            else result.spine_angle_deg
        )

        return OrientationResult(
            spine_angle_deg=smooth_angle,
            bilateral_conf=self._bilateral,
            spine_len_ratio=self._spine_ratio,
            confidence=result.confidence,
        )

    def reset(self) -> None:
        self._cos = self._sin = self._bilateral = self._spine_ratio = None

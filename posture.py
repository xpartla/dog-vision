"""Phase-2 posture classification on top of SuperAnimal-Quadruped keypoints.

Rule-based classifier: sitting / standing / lying / head tilt. Each feature is
computed independently and gracefully returns None when its keypoints are
missing or low-confidence; the classifier ensembles whatever survived. Sliding
window majority voting smooths the per-frame labels.

Camera assumption: roughly horizontal (tripod ~ribcage height, slight downward
tilt). Dog can be at any orientation.

Threshold values are starting guesses tuned to Australian Shepherd proportions;
expect to tune once you can actually see the classifier running on real footage.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# === Keypoint names ==========================================================
# Best-effort names for SuperAnimal-Quadruped. Run `--list-keypoints` against
# a generated .h5 file to verify; if any name differs, edit here.

KP_NOSE = "nose"
KP_LEFT_EYE = "left_eye"
KP_RIGHT_EYE = "right_eye"
KP_LEFT_EAR_BASE = "left_earbase"
KP_RIGHT_EAR_BASE = "right_earbase"
KP_NECK_BASE = "neck_base"
KP_BACK_BASE = "back_base"        # near hips end
KP_BACK_END = "back_end"          # near withers / shoulders
KP_TAIL_BASE = "tail_base"
KP_FRONT_LEFT_PAW = "front_left_paw"
KP_FRONT_RIGHT_PAW = "front_right_paw"
KP_BACK_LEFT_PAW = "back_left_paw"
KP_BACK_RIGHT_PAW = "back_right_paw"
KP_BACK_LEFT_KNEE = "back_left_knee"
KP_BACK_RIGHT_KNEE = "back_right_knee"
KP_BACK_LEFT_THIGH = "back_left_thai"   # SuperAnimal labels these "thai"
KP_BACK_RIGHT_THIGH = "back_right_thai"

DEFAULT_CONFIDENCE_THRESHOLD = 0.5


# === Per-frame keypoint container ===========================================

@dataclass(frozen=True)
class Keypoint:
    x: float
    y: float
    confidence: float


@dataclass
class Frame:
    keypoints: dict[str, Keypoint]
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD

    def get(self, name: str) -> Optional[Keypoint]:
        kp = self.keypoints.get(name)
        if kp is None or kp.confidence < self.confidence_threshold:
            return None
        return kp

    def visible(self) -> dict[str, Keypoint]:
        return {n: kp for n, kp in self.keypoints.items() if kp.confidence >= self.confidence_threshold}


# === DLC .h5 loader =========================================================

def list_keypoint_names(h5_path: Path) -> list[str]:
    """Return all bodypart names found in a DLC predictions .h5 file."""
    df = pd.read_hdf(h5_path)
    if not hasattr(df.columns, "get_level_values"):
        raise ValueError(f"{h5_path} does not have the expected DLC column structure")
    for level_name in ("bodyparts", "bodypart"):
        if level_name in df.columns.names:
            return list(df.columns.get_level_values(level_name).unique())
    # Fallback: assume bodypart is at level -2 (last level is x/y/likelihood)
    return list(df.columns.get_level_values(-2).unique())


def load_keypoint_frames(
    h5_path: Path,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> list[Frame]:
    """Read a DLC predictions .h5 file into a list of Frame objects (one per video frame).

    Handles both single-animal and multi-animal column layouts. For multi-animal,
    picks the individual with the highest mean confidence per frame (typically
    the actual subject when only one dog is in view).
    """
    df = pd.read_hdf(h5_path)

    # Find the bodypart and coords levels in the multi-index columns
    bodypart_level = None
    coords_level = None
    for name in df.columns.names:
        if name in ("bodyparts", "bodypart"):
            bodypart_level = name
        if name in ("coords", "coord"):
            coords_level = name
    if bodypart_level is None:
        bodypart_level = df.columns.names[-2]
    if coords_level is None:
        coords_level = df.columns.names[-1]

    bodyparts = list(df.columns.get_level_values(bodypart_level).unique())

    # Identify whether there are multiple individuals
    individual_level = None
    for name in df.columns.names:
        if name in ("individuals", "individual"):
            individual_level = name
            break

    frames: list[Frame] = []
    for frame_idx in range(len(df)):
        row = df.iloc[frame_idx]

        if individual_level is not None:
            individuals = list(df.columns.get_level_values(individual_level).unique())
            best_indiv, best_score = None, -1.0
            for indiv in individuals:
                mask = df.columns.get_level_values(individual_level) == indiv
                lik_mask = mask & (df.columns.get_level_values(coords_level) == "likelihood")
                if not lik_mask.any():
                    continue
                mean_lik = float(np.nanmean(row[lik_mask].values))
                if mean_lik > best_score:
                    best_indiv, best_score = indiv, mean_lik
            indiv_filter = best_indiv
        else:
            indiv_filter = None

        keypoints: dict[str, Keypoint] = {}
        for bp in bodyparts:
            try:
                if indiv_filter is not None:
                    sub = row.xs(bp, level=bodypart_level).xs(indiv_filter, level=individual_level)
                else:
                    sub = row.xs(bp, level=bodypart_level)
                x = float(sub.xs("x", level=coords_level).iloc[0] if hasattr(sub.xs("x", level=coords_level), "iloc") else sub.xs("x", level=coords_level))
                y = float(sub.xs("y", level=coords_level).iloc[0] if hasattr(sub.xs("y", level=coords_level), "iloc") else sub.xs("y", level=coords_level))
                lik = float(sub.xs("likelihood", level=coords_level).iloc[0] if hasattr(sub.xs("likelihood", level=coords_level), "iloc") else sub.xs("likelihood", level=coords_level))
            except (KeyError, ValueError, IndexError):
                continue
            if math.isnan(x) or math.isnan(y) or math.isnan(lik):
                continue
            keypoints[bp] = Keypoint(x=x, y=y, confidence=lik)

        frames.append(Frame(keypoints=keypoints, confidence_threshold=confidence_threshold))

    return frames


# === Geometric helpers ======================================================

def _vec(a: Keypoint, b: Keypoint) -> np.ndarray:
    return np.array([b.x - a.x, b.y - a.y], dtype=float)


def _angle_at(b: Keypoint, a: Keypoint, c: Keypoint) -> float:
    """Angle ABC in degrees, measured at vertex B."""
    v1 = _vec(b, a)
    v2 = _vec(b, c)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cos))


# === Posture features =======================================================

@dataclass
class PostureFeatures:
    body_aspect_h_over_w: Optional[float]   # height / width of visible-keypoint bbox
    back_knee_angle_deg: Optional[float]    # ~180 straight, ~90 sharp bend
    hip_height_ratio: Optional[float]       # (paw_y - hip_y) / spine_length
    spine_pitch_deg: Optional[float]        # angle of spine vs horizontal; +ve = front higher

    # Per-feature confidence (min likelihood of contributing keypoints)
    body_aspect_conf: float = 0.0
    back_knee_conf: float = 0.0
    hip_height_conf: float = 0.0
    spine_pitch_conf: float = 0.0


def compute_posture_features(frame: Frame) -> PostureFeatures:
    visible = frame.visible()

    # Body aspect (H/W) from all visible keypoints
    body_aspect: Optional[float] = None
    body_aspect_conf = 0.0
    if len(visible) >= 4:
        xs = np.array([kp.x for kp in visible.values()])
        ys = np.array([kp.y for kp in visible.values()])
        w = float(xs.max() - xs.min())
        h = float(ys.max() - ys.min())
        if w > 1.0:
            body_aspect = h / w
            body_aspect_conf = float(np.median([kp.confidence for kp in visible.values()]))

    # Back-knee angle: pick the leg with highest min-confidence
    back_knee: Optional[float] = None
    back_knee_conf = 0.0
    for thigh_n, knee_n, paw_n in (
        (KP_BACK_LEFT_THIGH, KP_BACK_LEFT_KNEE, KP_BACK_LEFT_PAW),
        (KP_BACK_RIGHT_THIGH, KP_BACK_RIGHT_KNEE, KP_BACK_RIGHT_PAW),
    ):
        t, k, p = frame.get(thigh_n), frame.get(knee_n), frame.get(paw_n)
        if t and k and p:
            angle = _angle_at(k, t, p)
            conf = min(t.confidence, k.confidence, p.confidence)
            if not math.isnan(angle) and conf > back_knee_conf:
                back_knee, back_knee_conf = angle, conf

    # Hip height above paws, normalized by spine length
    hip = frame.get(KP_BACK_BASE) or frame.get(KP_TAIL_BASE)
    shoulder = frame.get(KP_BACK_END) or frame.get(KP_NECK_BASE)
    paw_candidates = [frame.get(n) for n in (KP_BACK_LEFT_PAW, KP_BACK_RIGHT_PAW,
                                              KP_FRONT_LEFT_PAW, KP_FRONT_RIGHT_PAW)]
    paws = [p for p in paw_candidates if p is not None]

    hip_height: Optional[float] = None
    hip_height_conf = 0.0
    if hip and shoulder and paws:
        spine_len = math.hypot(shoulder.x - hip.x, shoulder.y - hip.y)
        if spine_len > 1.0:
            paw_y = max(p.y for p in paws)  # bottom-most paw (largest y in image coords)
            hip_height = (paw_y - hip.y) / spine_len
            hip_height_conf = min(hip.confidence, shoulder.confidence,
                                  max(p.confidence for p in paws))

    # Spine pitch (front higher = positive)
    spine_pitch: Optional[float] = None
    spine_pitch_conf = 0.0
    if hip and shoulder:
        dy = hip.y - shoulder.y       # +ve when shoulder higher (smaller y)
        dx = abs(shoulder.x - hip.x)
        if dx > 1.0 or abs(dy) > 1.0:
            spine_pitch = math.degrees(math.atan2(dy, dx + 1e-6))
            spine_pitch_conf = min(hip.confidence, shoulder.confidence)

    return PostureFeatures(
        body_aspect_h_over_w=body_aspect,
        back_knee_angle_deg=back_knee,
        hip_height_ratio=hip_height,
        spine_pitch_deg=spine_pitch,
        body_aspect_conf=body_aspect_conf,
        back_knee_conf=back_knee_conf,
        hip_height_conf=hip_height_conf,
        spine_pitch_conf=spine_pitch_conf,
    )


# === Posture classifier =====================================================

POSTURE_LABELS = ("sitting", "standing", "lying", "unknown")


def classify_posture(features: PostureFeatures) -> tuple[str, float]:
    """Return (label, score-in-0-1).

    Each feature contributes weighted votes. Output is the highest-voted label,
    scaled by the fraction of features that successfully contributed. If too few
    features are available, returns 'unknown'.
    """
    scores = {"sitting": 0.0, "standing": 0.0, "lying": 0.0}
    available_weight = 0.0

    # Body H/W: a flat bbox (small H/W) is a robust 'lying' signal regardless of orientation.
    if features.body_aspect_h_over_w is not None:
        w = features.body_aspect_conf
        available_weight += w
        ar = features.body_aspect_h_over_w
        if ar < 0.5:
            scores["lying"] += 2.0 * w
        elif ar > 1.3:
            scores["standing"] += 0.6 * w   # tall+narrow = front-view standing
        else:
            scores["sitting"] += 0.4 * w
            scores["standing"] += 0.4 * w

    # Back-knee angle: the strongest sit/stand discriminator when visible.
    if features.back_knee_angle_deg is not None:
        w = features.back_knee_conf
        available_weight += w
        ang = features.back_knee_angle_deg
        if ang < 110:
            scores["sitting"] += 1.5 * w
        elif ang > 145:
            scores["standing"] += 1.5 * w
        else:
            scores["sitting"] += 0.4 * w
            scores["standing"] += 0.4 * w

    # Hip height above paws: standing has hips well above paw line.
    if features.hip_height_ratio is not None:
        w = features.hip_height_conf
        available_weight += w
        r = features.hip_height_ratio
        if r > 0.45:
            scores["standing"] += 1.0 * w
        elif r < 0.15:
            scores["lying"] += 1.0 * w
        else:
            scores["sitting"] += 0.8 * w

    # Spine pitch: sitting tilts the spine up at the front.
    if features.spine_pitch_deg is not None:
        w = features.spine_pitch_conf
        available_weight += w
        p = features.spine_pitch_deg
        if 15 < p < 60:
            scores["sitting"] += 0.8 * w
        elif abs(p) < 10:
            scores["standing"] += 0.3 * w
            scores["lying"] += 0.3 * w

    if available_weight < 0.5:
        return ("unknown", 0.0)

    best = max(scores, key=scores.get)
    total = sum(scores.values())
    if total < 1e-6:
        return ("unknown", 0.0)
    confidence = scores[best] / total
    if confidence < 0.4:
        return ("unknown", confidence)
    return (best, confidence)


# === Head tilt ==============================================================

HEAD_TILT_LABELS = ("upright", "tilt_left", "tilt_right", "unknown")


def classify_head_tilt(frame: Frame, tilt_threshold_deg: float = 15.0) -> tuple[str, float]:
    """Tilt of the eye-line relative to (perpendicular-to-head-axis).

    Returns (label, signed-tilt-deg). Sign convention: positive = right ear lower
    than left ear in image space (head tilted toward the dog's left from the
    viewer's perspective); negative = the other way.
    """
    nose = frame.get(KP_NOSE)
    neck = frame.get(KP_NECK_BASE)
    le = frame.get(KP_LEFT_EYE)
    re = frame.get(KP_RIGHT_EYE)
    # Fall back to ear bases if eyes are not visible
    if not le or not re:
        le = frame.get(KP_LEFT_EAR_BASE) or le
        re = frame.get(KP_RIGHT_EAR_BASE) or re

    if not (nose and neck and le and re):
        return ("unknown", 0.0)

    head_axis = _vec(nose, neck)
    head_len = float(np.linalg.norm(head_axis))
    if head_len < 5.0:
        # Dog is facing camera head-on (or away) — head axis collapses; tilt is unobservable.
        return ("unknown", 0.0)

    eye_axis = _vec(le, re)
    if float(np.linalg.norm(eye_axis)) < 1e-6:
        return ("unknown", 0.0)

    # Perpendicular to head axis (rotate 90° in image plane)
    head_perp = np.array([-head_axis[1], head_axis[0]])
    head_perp /= float(np.linalg.norm(head_perp))

    # Signed angle between eye_axis and head_perp
    eye_norm = eye_axis / float(np.linalg.norm(eye_axis))
    cos = float(np.clip(np.dot(eye_norm, head_perp), -1.0, 1.0))
    sin = float(eye_norm[0] * head_perp[1] - eye_norm[1] * head_perp[0])
    angle_deg = math.degrees(math.atan2(sin, cos))
    # Wrap to [-90, 90] (sign reflects tilt direction)
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180

    if abs(angle_deg) < tilt_threshold_deg:
        return ("upright", angle_deg)
    return (("tilt_left" if angle_deg > 0 else "tilt_right"), angle_deg)


# === Sliding-window smoothing ==============================================

class LabelSmoother:
    """Majority-vote label over a sliding window of recent predictions."""

    def __init__(self, window: int = 10):
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self._buffer: deque[str] = deque(maxlen=window)

    def push(self, label: str) -> str:
        self._buffer.append(label)
        return Counter(self._buffer).most_common(1)[0][0]

    def reset(self) -> None:
        self._buffer.clear()

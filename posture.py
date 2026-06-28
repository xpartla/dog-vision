"""Phase-2 posture classification on top of SuperAnimal-Quadruped keypoints.

Rule-based classifier: sitting / standing / lying / head tilt. Each feature is
computed independently and gracefully returns None when its keypoints are
missing or low-confidence; the classifier ensembles whatever survived. Sliding
window majority voting smooths the per-frame labels.

Camera assumption: roughly horizontal (tripod ~ribcage height, slight downward
tilt). Dog can be at any orientation.

Height features are measured against the lowest *visible* keypoint (the ground
proxy) rather than the paws specifically — paws are the first points to drop out
when a dog sits or lies, and anchoring to them used to collapse every feature to
None and wreck the classification. Ratios are normalized by the keypoint-bbox
diagonal so they are scale-independent.

Threshold values are starting guesses for Australian Shepherd proportions. Tune
them on real footage with `classify_video.py --dump-features out.csv`, which
writes the per-frame feature values alongside the predicted label.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from orientation import OrientationResult

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
KP_BACK_BASE = "back_base"        # FRONT of the back (near withers / shoulders)
KP_BACK_MIDDLE = "back_middle"
KP_BACK_END = "back_end"          # REAR of the back (near hips / sacrum)
KP_TAIL_BASE = "tail_base"
KP_TAIL_END = "tail_end"
KP_FRONT_LEFT_THIGH = "front_left_thai"   # SuperAnimal labels these "thai"
KP_FRONT_RIGHT_THIGH = "front_right_thai"
KP_FRONT_LEFT_KNEE = "front_left_knee"
KP_FRONT_RIGHT_KNEE = "front_right_knee"
KP_FRONT_LEFT_PAW = "front_left_paw"
KP_FRONT_RIGHT_PAW = "front_right_paw"
KP_BACK_LEFT_THIGH = "back_left_thai"
KP_BACK_RIGHT_THIGH = "back_right_thai"
KP_BACK_LEFT_KNEE = "back_left_knee"
KP_BACK_RIGHT_KNEE = "back_right_knee"
KP_BACK_LEFT_PAW = "back_left_paw"
KP_BACK_RIGHT_PAW = "back_right_paw"

TRUNK_KEYPOINTS = (KP_NECK_BASE, KP_BACK_END, KP_BACK_MIDDLE, KP_BACK_BASE)
ALL_PAW_KEYPOINTS = (KP_FRONT_LEFT_PAW, KP_FRONT_RIGHT_PAW,
                     KP_BACK_LEFT_PAW, KP_BACK_RIGHT_PAW)

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
    return list(df.columns.get_level_values(-2).unique())


def load_keypoint_frames(
    h5_path: Path,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> list[Frame]:
    """Read a DLC predictions .h5 file into a list of Frame objects.

    Handles both single-animal and multi-animal column layouts. For multi-animal,
    picks the individual with the highest mean confidence per frame.
    """
    df = pd.read_hdf(h5_path)

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
                xv = sub.xs("x", level=coords_level)
                yv = sub.xs("y", level=coords_level)
                lv = sub.xs("likelihood", level=coords_level)
                x = float(xv.iloc[0] if hasattr(xv, "iloc") else xv)
                y = float(yv.iloc[0] if hasattr(yv, "iloc") else yv)
                lik = float(lv.iloc[0] if hasattr(lv, "iloc") else lv)
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


# === Keypoint smoothing (1€ filter) =========================================

class _OneEuroFilter:
    """1€ filter (Casiez et al. 2012). Adaptive low-pass: smooths slow motion
    heavily, lets fast motion through with little lag."""

    def __init__(self, fps: float, mincutoff: float = 1.0, beta: float = 0.5,
                 dcutoff: float = 1.0):
        self.fps = fps
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float) -> float:
        if self._x_prev is None:
            self._x_prev = x
            return x
        dx = (x - self._x_prev) * self.fps
        a_d = self._alpha(self.dcutoff, self.fps)
        dx_smooth = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self.mincutoff + self.beta * abs(dx_smooth)
        a = self._alpha(cutoff, self.fps)
        x_smooth = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_smooth
        self._dx_prev = dx_smooth
        return x_smooth

    @property
    def last(self) -> Optional[float]:
        return self._x_prev


class KeypointSmoother:
    """Per-keypoint, per-axis 1€ filter for smoothing keypoint trajectories.

    State persists across calls so it works seamlessly across chunks in live
    mode. Low-confidence frames don't update the filter (so a brief detection
    failure doesn't anchor the smoother to a bad value); they instead emit the
    last smoothed position with the original (low) confidence.
    """

    def __init__(self, fps: float = 30.0, mincutoff: float = 1.0, beta: float = 0.5):
        self.fps = fps
        self.mincutoff = mincutoff
        self.beta = beta
        self._filters: dict[str, tuple[_OneEuroFilter, _OneEuroFilter]] = {}

    def smooth(self, frame: Frame) -> Frame:
        new_kps: dict[str, Keypoint] = {}
        for name, kp in frame.keypoints.items():
            if name not in self._filters:
                self._filters[name] = (
                    _OneEuroFilter(self.fps, self.mincutoff, self.beta),
                    _OneEuroFilter(self.fps, self.mincutoff, self.beta),
                )
            fx, fy = self._filters[name]
            if kp.confidence >= frame.confidence_threshold:
                sx, sy = fx.filter(kp.x), fy.filter(kp.y)
            else:
                sx = fx.last if fx.last is not None else kp.x
                sy = fy.last if fy.last is not None else kp.y
            new_kps[name] = Keypoint(sx, sy, kp.confidence)
        return Frame(keypoints=new_kps, confidence_threshold=frame.confidence_threshold)

    def reset(self) -> None:
        self._filters.clear()


# === Posture features =======================================================

@dataclass
class PostureFeatures:
    body_aspect_h_over_w: Optional[float]      # height / width of visible-keypoint bbox
    back_knee_angle_deg: Optional[float]       # ~180 straight, ~90 sharp bend
    trunk_above_ground_ratio: Optional[float]  # (ground_y - trunk_median_y) / bbox_diag
    head_above_ground_ratio: Optional[float]   # (ground_y - head_mean_y) / bbox_diag
    hip_above_ground_ratio: Optional[float]    # (ground_y - hip_y) / bbox_diag
    spine_pitch_deg: Optional[float]           # angle of spine vs horizontal; +ve = front higher

    # Diagnostic: True if the ground estimate coincided with a real paw, False
    # if it fell back to the lowest non-paw keypoint (knee / hock / tail).
    ground_from_paws: bool = False

    body_aspect_conf: float = 0.0
    back_knee_conf: float = 0.0
    trunk_above_ground_conf: float = 0.0
    head_above_ground_conf: float = 0.0
    hip_above_ground_conf: float = 0.0
    spine_pitch_conf: float = 0.0


def _median_y(pts: list[Keypoint]) -> float:
    ys = sorted(p.y for p in pts)
    return ys[len(ys) // 2]


def compute_posture_features(frame: Frame) -> PostureFeatures:
    visible = frame.visible()

    # Body H/W aspect (weak signal; weighted lightly in the classifier)
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

    # bbox_diag: diagonal of the bounding box of all visible keypoints. Used as
    # the normalizer for the height-ratio features below. The diagonal (vs. the
    # longer side) varies smoothly with viewpoint and never flips between the
    # width and height axis as the dog rotates.
    bbox_diag: Optional[float] = None
    ground_y: Optional[float] = None
    if len(visible) >= 2:
        xs = [kp.x for kp in visible.values()]
        ys = [kp.y for kp in visible.values()]
        diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        if diag > 1.0:
            bbox_diag = diag
        # Ground reference = lowest visible keypoint of ANY part (image y grows
        # downward). Anchoring to the lowest *paw* fails constantly because paws
        # are the first keypoints to drop out when a dog sits or lies (tucked /
        # occluded). The lowest visible point — a hock, knee, or tail if the
        # paws are gone — is a far more reliable floor and keeps every height
        # feature alive instead of collapsing them to None.
        ground_y = max(ys)

    # Whether the ground estimate landed on an actual paw (diagnostic only).
    paw_pts = [frame.get(n) for n in ALL_PAW_KEYPOINTS]
    paw_pts = [p for p in paw_pts if p is not None]
    ground_from_paws = bool(
        paw_pts and ground_y is not None
        and abs(max(p.y for p in paw_pts) - ground_y) < 1e-6
    )

    # Back-knee (stifle) angle: pick the leg with highest min-confidence.
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

    # Trunk-above-ground: median trunk Y vs ground, normalized by bbox_diag.
    # This is the primary lying detector — when the dog is down, the spine sits
    # right at the floor, so the clearance collapses toward zero.
    trunk_pts = [frame.get(n) for n in TRUNK_KEYPOINTS]
    trunk_pts = [p for p in trunk_pts if p is not None]

    trunk_above_ground: Optional[float] = None
    trunk_above_ground_conf = 0.0
    if len(trunk_pts) >= 2 and bbox_diag is not None and ground_y is not None:
        trunk_above_ground = (ground_y - _median_y(trunk_pts)) / bbox_diag
        trunk_above_ground_conf = float(np.median([p.confidence for p in trunk_pts]))

    # Head-above-ground: mean head Y (nose + eyes) vs ground. The head is
    # reliably detected on Aussies (face fur is short); head-near-ground is a
    # strong lying corroborator (though not required — sphinx lying keeps the
    # head up).
    head_pts = [frame.get(n) for n in (KP_NOSE, KP_LEFT_EYE, KP_RIGHT_EYE)]
    head_pts = [p for p in head_pts if p is not None]

    head_above_ground: Optional[float] = None
    head_above_ground_conf = 0.0
    if head_pts and bbox_diag is not None and ground_y is not None:
        head_y = sum(p.y for p in head_pts) / len(head_pts)
        head_above_ground = (ground_y - head_y) / bbox_diag
        head_above_ground_conf = float(np.median([p.confidence for p in head_pts]))

    # Hip-above-ground: primary sit/stand discriminator — hindquarters on the
    # ground (sitting) vs. raised on extended hind legs (standing). The "hip"
    # landmark in SuperAnimal is `back_end` (rear of the back), with tail_base
    # as a near-by fallback.
    hip = frame.get(KP_BACK_END) or frame.get(KP_TAIL_BASE)
    hip_above_ground: Optional[float] = None
    hip_above_ground_conf = 0.0
    if hip and bbox_diag is not None and ground_y is not None:
        hip_above_ground = (ground_y - hip.y) / bbox_diag
        hip_above_ground_conf = hip.confidence

    # Spine pitch (shoulder higher than hip = positive = sitting indicator).
    # Shoulder = `back_base` (front of back), hip = `back_end` (rear of back).
    shoulder = frame.get(KP_BACK_BASE) or frame.get(KP_NECK_BASE)
    spine_pitch: Optional[float] = None
    spine_pitch_conf = 0.0
    if hip and shoulder:
        dy = hip.y - shoulder.y
        dx = abs(shoulder.x - hip.x)
        if dx > 1.0 or abs(dy) > 1.0:
            spine_pitch = math.degrees(math.atan2(dy, dx + 1e-6))
            spine_pitch_conf = min(hip.confidence, shoulder.confidence)

    return PostureFeatures(
        body_aspect_h_over_w=body_aspect,
        back_knee_angle_deg=back_knee,
        trunk_above_ground_ratio=trunk_above_ground,
        head_above_ground_ratio=head_above_ground,
        hip_above_ground_ratio=hip_above_ground,
        spine_pitch_deg=spine_pitch,
        ground_from_paws=ground_from_paws,
        body_aspect_conf=body_aspect_conf,
        back_knee_conf=back_knee_conf,
        trunk_above_ground_conf=trunk_above_ground_conf,
        head_above_ground_conf=head_above_ground_conf,
        hip_above_ground_conf=hip_above_ground_conf,
        spine_pitch_conf=spine_pitch_conf,
    )


# === Posture classifier =====================================================

POSTURE_LABELS = ("sitting", "standing", "lying", "unknown")


def classify_posture(features: PostureFeatures) -> tuple[str, float]:
    """Return (label, score-in-0-1).

    All height features are measured against the lowest visible keypoint (the
    ground proxy), normalized by the keypoint-bbox diagonal — so they stay
    defined even when the paws are not detected, which is the usual case for a
    sitting or lying dog.

    Lying is detected by the *trunk* sitting at the ground (trunk clearance ~0)
    while the spine is roughly horizontal. The spine-pitch guard prevents a
    sitting dog (rear low, but spine steeply tilted) from being read as lying.

    Sit-vs-stand is then decided by hip clearance (hindquarters on the ground
    vs. raised on extended hind legs), reinforced by spine pitch and the hind
    knee/stifle angle.
    """
    scores = {"sitting": 0.0, "standing": 0.0, "lying": 0.0}
    available_weight = 0.0

    head_ratio = features.head_above_ground_ratio
    trunk_ratio = features.trunk_above_ground_ratio
    hip_ratio = features.hip_above_ground_ratio
    pitch = features.spine_pitch_deg

    spine_steep = pitch is not None and pitch > 20.0
    trunk_low = trunk_ratio is not None and trunk_ratio < 0.15
    head_low = head_ratio is not None and head_ratio < 0.15
    hip_low = hip_ratio is not None and hip_ratio < 0.15

    # Lying: trunk resting at the ground with a near-horizontal spine. A steep
    # spine means the dog is propped up at the front (sitting), not lying flat.
    body_on_ground = trunk_low and not spine_steep

    if body_on_ground:
        w = features.trunk_above_ground_conf
        scores["lying"] += 3.0 * w
        available_weight += w
        # Head down at the ground corroborates (but isn't required — a sphinx
        # lie keeps the head up).
        if head_low:
            hw = features.head_above_ground_conf
            scores["lying"] += 1.0 * hw
            available_weight += hw

    else:
        # Hip clearance: primary sit/stand discriminator.
        if hip_ratio is not None:
            w = features.hip_above_ground_conf
            available_weight += w
            if hip_ratio < 0.16:
                scores["sitting"] += 1.5 * w
            elif hip_ratio > 0.22:
                scores["standing"] += 1.5 * w
            # 0.16-0.22 ambiguous, no vote

        # Spine pitch: front clearly higher than rear = sitting.
        if pitch is not None:
            w = features.spine_pitch_conf
            if pitch > 20.0:
                scores["sitting"] += 1.2 * w
                available_weight += w
            elif pitch < 10.0:
                scores["standing"] += 0.6 * w
                available_weight += w

        # Hind knee/stifle angle: clear bend = sitting, clearly extended = standing.
        if features.back_knee_angle_deg is not None:
            w = features.back_knee_conf
            ang = features.back_knee_angle_deg
            if ang < 110:
                scores["sitting"] += 1.5 * w
                available_weight += w
            elif ang > 145:
                scores["standing"] += 1.5 * w
                available_weight += w
            # 110-145 ambiguous, no vote

        # Trunk clearance: reinforces sit/stand for an upright dog.
        if trunk_ratio is not None:
            w = features.trunk_above_ground_conf
            if trunk_ratio > 0.30:
                scores["standing"] += 0.6 * w
                available_weight += w

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


# === Learned posture classifier =============================================

class LearnedPostureClassifier:
    """Drop-in replacement for `classify_posture` backed by a trained model.

    Loads a bundle saved by `train_posture.py` and classifies a Frame directly
    from its viewpoint-robust feature vector (see `pose_features.py`), instead
    of the hand-tuned geometric rules. Unlike the rule-based path it makes no
    assumption about camera placement — it generalizes to whatever viewpoints
    appeared in the training data.

    Usage:
        clf = LearnedPostureClassifier("posture_model.joblib")
        label, prob = clf.classify(frame)
    """

    def __init__(self, model_path, min_proba: float = 0.5):
        import joblib  # local import: only needed when a learned model is used

        # Imported here (not at module top) to avoid a circular import:
        # pose_features imports names from this module.
        import pose_features as pf

        bundle = joblib.load(model_path)
        self._model = bundle["model"]
        self._classes = list(bundle["classes"])
        self._feature_names = list(bundle["feature_names"])
        self.confidence_threshold = float(bundle.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD))
        self.min_proba = min_proba
        self._pf = pf

        if len(self._feature_names) != pf.N_FEATURES:
            raise ValueError(
                f"Model expects {len(self._feature_names)} features but the current "
                f"pose_features produces {pf.N_FEATURES}. The feature definition "
                f"changed since training — retrain with build_dataset.py/train_posture.py."
            )

    def classify(self, frame: "Frame") -> tuple[str, float]:
        """Return (label, probability). 'unknown' if too few keypoints or the
        top class probability is below `min_proba`."""
        vec = self._pf.feature_vector(frame)
        if vec is None:
            return ("unknown", 0.0)
        proba = self.classify_batch(np.array([vec]))[0]
        best = int(np.argmax(proba))
        label, p = self._classes[best], float(proba[best])
        if p < self.min_proba:
            return ("unknown", p)
        return (label, p)

    def classify_batch(self, vecs: np.ndarray) -> np.ndarray:
        """Batch predict_proba — avoids per-call sklearn parallel overhead."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._model.predict_proba(vecs)

    def classify_frames(self, frames: "list[Frame]") -> "list[tuple[str, float]]":
        """Classify a list of frames in one batch call (much faster for RF)."""
        vecs, indices = [], []
        results: list[tuple[str, float]] = [("unknown", 0.0)] * len(frames)
        for i, frame in enumerate(frames):
            vec = self._pf.feature_vector(frame)
            if vec is not None:
                vecs.append(vec)
                indices.append(i)
        if vecs:
            probas = self.classify_batch(np.stack(vecs))
            for idx, proba in zip(indices, probas):
                best = int(np.argmax(proba))
                label, p = self._classes[best], float(proba[best])
                results[idx] = (label, p) if p >= self.min_proba else ("unknown", p)
        return results


# === Head tilt ==============================================================

HEAD_TILT_LABELS = ("upright", "tilt_left", "tilt_right", "unknown")


def classify_head_tilt(
    frame: Frame,
    tilt_threshold_deg: float = 15.0,
    orientation: Optional[OrientationResult] = None,
) -> tuple[str, float]:
    """Tilt of the eye-line relative to (perpendicular-to-head-axis).

    Returns (label, signed-tilt-deg). Sign convention: positive = right side
    lower in image space; negative = left side lower.

    When `orientation` is supplied and the dog is mostly in profile
    (bilateral_conf < 0.25), the eye-line measurement is unreliable and
    "unknown" is returned instead of a likely-wrong label.
    """
    if orientation is not None and orientation.bilateral_conf < 0.25:
        return ("unknown", 0.0)

    nose = frame.get(KP_NOSE)
    neck = frame.get(KP_NECK_BASE)
    le = frame.get(KP_LEFT_EYE)
    re = frame.get(KP_RIGHT_EYE)
    if not le or not re:
        le = frame.get(KP_LEFT_EAR_BASE) or le
        re = frame.get(KP_RIGHT_EAR_BASE) or re

    if not (nose and neck and le and re):
        return ("unknown", 0.0)

    head_axis = _vec(nose, neck)
    head_len = float(np.linalg.norm(head_axis))
    if head_len < 5.0:
        return ("unknown", 0.0)

    eye_axis = _vec(le, re)
    if float(np.linalg.norm(eye_axis)) < 1e-6:
        return ("unknown", 0.0)

    head_perp = np.array([-head_axis[1], head_axis[0]])
    head_perp /= float(np.linalg.norm(head_perp))

    eye_norm = eye_axis / float(np.linalg.norm(eye_axis))
    cos = float(np.clip(np.dot(eye_norm, head_perp), -1.0, 1.0))
    sin = float(eye_norm[0] * head_perp[1] - eye_norm[1] * head_perp[0])
    angle_deg = math.degrees(math.atan2(sin, cos))
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180

    if abs(angle_deg) < tilt_threshold_deg:
        return ("upright", angle_deg)
    return (("tilt_left" if angle_deg > 0 else "tilt_right"), angle_deg)


# === Sliding-window smoothing ==============================================

class LabelSmoother:
    """Majority-vote label over a sliding window of recent predictions.

    When `suppress_unknown` is True (the default), "unknown" labels are not
    pushed into the vote buffer — they're replaced by the last non-unknown
    result so transient detection failures don't pollute the majority.
    """

    def __init__(self, window: int = 10, suppress_unknown: bool = True):
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self.suppress_unknown = suppress_unknown
        self._buffer: deque[str] = deque(maxlen=window)
        self._last_stable: str = "unknown"

    def push(self, label: str) -> str:
        if self.suppress_unknown and label == "unknown":
            effective = self._last_stable
        else:
            effective = label
        self._buffer.append(effective)
        result = Counter(self._buffer).most_common(1)[0][0]
        if result != "unknown":
            self._last_stable = result
        return result

    def reset(self) -> None:
        self._buffer.clear()
        self._last_stable = "unknown"


def merge_short_segments(
    labels: list[str],
    min_length: int = 25,
    unknown_label: str = "unknown",
) -> list[str]:
    """Post-process a label sequence: fill unknown gaps then merge short segments.

    1. Replace any remaining "unknown" labels with the nearest preceding
       non-unknown label (or the first following one if at the start).
    2. Iteratively merge the shortest segment into its larger neighbor until
       all segments are >= min_length frames.
    """
    if not labels:
        return labels

    result = list(labels)

    # Pass 1: forward-fill unknown with the last stable label.
    stable: Optional[str] = None
    for i in range(len(result)):
        if result[i] != unknown_label:
            stable = result[i]
        elif stable is not None:
            result[i] = stable

    # Back-fill any leading unknowns from the first stable label.
    first_stable = next((l for l in result if l != unknown_label), unknown_label)
    for i in range(len(result)):
        if result[i] == unknown_label:
            result[i] = first_stable
        else:
            break

    # Pass 2: iteratively merge shortest segment into its longer neighbour.
    def _segments(seq: list[str]) -> list[tuple[int, int, int, str]]:
        if not seq:
            return []
        segs: list[tuple[int, int, int, str]] = []
        s, cur = 0, seq[0]
        for i, v in enumerate(seq):
            if v != cur:
                segs.append((s, i - 1, i - s, cur))
                s, cur = i, v
        segs.append((s, len(seq) - 1, len(seq) - s, cur))
        return segs

    changed = True
    while changed:
        changed = False
        segs = _segments(result)
        # Find the shortest segment below the threshold.
        short_segs = [(idx, s) for idx, s in enumerate(segs) if s[2] < min_length]
        if not short_segs:
            break
        # Pick the shortest one (ties broken by first occurrence).
        seg_idx, (start, end, _length, _label) = min(short_segs, key=lambda x: x[1][2])
        if seg_idx == 0:
            nb_label = segs[1][3] if len(segs) > 1 else _label
        elif seg_idx == len(segs) - 1:
            nb_label = segs[seg_idx - 1][3]
        else:
            left_len = segs[seg_idx - 1][2]
            right_len = segs[seg_idx + 1][2]
            nb_label = segs[seg_idx - 1][3] if left_len >= right_len else segs[seg_idx + 1][3]
        for i in range(start, end + 1):
            result[i] = nb_label
        changed = True

    return result

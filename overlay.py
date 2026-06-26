"""Drawing utilities: keypoints, skeleton, and posture/tilt labels."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from posture import (
    Frame,
    PostureFeatures,
    KP_NOSE, KP_LEFT_EYE, KP_RIGHT_EYE, KP_LEFT_EAR_BASE, KP_RIGHT_EAR_BASE,
    KP_NECK_BASE, KP_BACK_BASE, KP_BACK_MIDDLE, KP_BACK_END,
    KP_TAIL_BASE, KP_TAIL_END,
    KP_FRONT_LEFT_THIGH, KP_FRONT_RIGHT_THIGH,
    KP_FRONT_LEFT_KNEE, KP_FRONT_RIGHT_KNEE,
    KP_FRONT_LEFT_PAW, KP_FRONT_RIGHT_PAW,
    KP_BACK_LEFT_THIGH, KP_BACK_RIGHT_THIGH,
    KP_BACK_LEFT_KNEE, KP_BACK_RIGHT_KNEE,
    KP_BACK_LEFT_PAW, KP_BACK_RIGHT_PAW,
)

# Colors are BGR (OpenCV convention)
HEAD_COLOR = (0, 220, 220)        # yellow
SPINE_COLOR = (60, 220, 60)       # green
LEG_COLOR = (220, 200, 0)         # cyan
TAIL_COLOR = (200, 0, 200)        # magenta
DEFAULT_KP_COLOR = (220, 220, 220)

POSTURE_COLORS = {
    "sitting": (0, 200, 255),
    "standing": (0, 255, 0),
    "lying": (255, 100, 100),
    "unknown": (160, 160, 160),
}

HEAD_TILT_COLORS = {
    "upright": (200, 200, 200),
    "tilt_left": (0, 200, 255),
    "tilt_right": (255, 100, 0),
    "unknown": (160, 160, 160),
}

SKELETON_EDGES: list[tuple[str, str, tuple[int, int, int]]] = [
    # Head
    (KP_NOSE, KP_LEFT_EYE, HEAD_COLOR),
    (KP_NOSE, KP_RIGHT_EYE, HEAD_COLOR),
    (KP_LEFT_EYE, KP_LEFT_EAR_BASE, HEAD_COLOR),
    (KP_RIGHT_EYE, KP_RIGHT_EAR_BASE, HEAD_COLOR),
    (KP_NOSE, KP_NECK_BASE, HEAD_COLOR),
    # Spine: neck → front-of-back → middle → rear-of-back → tail
    (KP_NECK_BASE, KP_BACK_BASE, SPINE_COLOR),
    (KP_BACK_BASE, KP_BACK_MIDDLE, SPINE_COLOR),
    (KP_BACK_MIDDLE, KP_BACK_END, SPINE_COLOR),
    (KP_BACK_END, KP_TAIL_BASE, SPINE_COLOR),
    # Tail
    (KP_TAIL_BASE, KP_TAIL_END, TAIL_COLOR),
    # Front legs attach at the FRONT of the back (back_base = withers area)
    (KP_BACK_BASE, KP_FRONT_LEFT_THIGH, LEG_COLOR),
    (KP_FRONT_LEFT_THIGH, KP_FRONT_LEFT_KNEE, LEG_COLOR),
    (KP_FRONT_LEFT_KNEE, KP_FRONT_LEFT_PAW, LEG_COLOR),
    (KP_BACK_BASE, KP_FRONT_RIGHT_THIGH, LEG_COLOR),
    (KP_FRONT_RIGHT_THIGH, KP_FRONT_RIGHT_KNEE, LEG_COLOR),
    (KP_FRONT_RIGHT_KNEE, KP_FRONT_RIGHT_PAW, LEG_COLOR),
    # Back legs attach at the REAR of the back (back_end = hip area)
    (KP_BACK_END, KP_BACK_LEFT_THIGH, LEG_COLOR),
    (KP_BACK_LEFT_THIGH, KP_BACK_LEFT_KNEE, LEG_COLOR),
    (KP_BACK_LEFT_KNEE, KP_BACK_LEFT_PAW, LEG_COLOR),
    (KP_BACK_END, KP_BACK_RIGHT_THIGH, LEG_COLOR),
    (KP_BACK_RIGHT_THIGH, KP_BACK_RIGHT_KNEE, LEG_COLOR),
    (KP_BACK_RIGHT_KNEE, KP_BACK_RIGHT_PAW, LEG_COLOR),
]

_HEAD_PARTS = {KP_NOSE, KP_LEFT_EYE, KP_RIGHT_EYE, KP_LEFT_EAR_BASE, KP_RIGHT_EAR_BASE}
_SPINE_PARTS = {KP_NECK_BASE, KP_BACK_END, KP_BACK_MIDDLE, KP_BACK_BASE}
_TAIL_PARTS = {KP_TAIL_BASE, KP_TAIL_END}


def _kp_color(name: str) -> tuple[int, int, int]:
    if name in _HEAD_PARTS:
        return HEAD_COLOR
    if name in _SPINE_PARTS:
        return SPINE_COLOR
    if name in _TAIL_PARTS:
        return TAIL_COLOR
    return LEG_COLOR


def _draw_text(image: np.ndarray, text: str, origin: tuple[int, int],
               color: tuple[int, int, int], scale: float = 0.7) -> None:
    thickness = max(2, round(_scale(image, 2)))
    shadow = max(4, round(_scale(image, 6)))
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), shadow, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def _scale(image: np.ndarray, base: float, base_height: int = 720) -> float:
    """Scale a size proportionally to image height relative to a 720p baseline."""
    return base * image.shape[0] / base_height


def draw_skeleton(image: np.ndarray, frame: Frame,
                  edge_thickness: int = 2, point_radius: int = 4) -> None:
    """Draw skeleton edges and keypoints from a Frame onto the image (in place)."""
    thickness = max(1, round(_scale(image, edge_thickness)))
    radius = max(2, round(_scale(image, point_radius)))
    for k1, k2, color in SKELETON_EDGES:
        p1 = frame.get(k1)
        p2 = frame.get(k2)
        if p1 is None or p2 is None:
            continue
        cv2.line(image,
                 (int(p1.x), int(p1.y)),
                 (int(p2.x), int(p2.y)),
                 color, thickness, cv2.LINE_AA)

    for name, kp in frame.visible().items():
        color = _kp_color(name)
        center = (int(kp.x), int(kp.y))
        cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)
        cv2.circle(image, center, radius, (0, 0, 0), 1, cv2.LINE_AA)


def draw_overlay(
    image: np.ndarray,
    frame: Optional[Frame],
    posture: tuple[str, float] = ("unknown", 0.0),
    head_tilt: tuple[str, float] = ("unknown", 0.0),
    debug_features: Optional[PostureFeatures] = None,
) -> None:
    """Full overlay: skeleton + keypoints + posture/tilt labels (in place)."""
    if frame is not None:
        draw_skeleton(image, frame)

    font_scale = _scale(image, 0.8)
    line_gap = max(30, round(_scale(image, 30)))
    margin = max(12, round(_scale(image, 12)))

    p_label, p_score = posture
    t_label, t_angle = head_tilt
    _draw_text(image, f"posture: {p_label} ({p_score:.2f})",
               (margin, line_gap),
               POSTURE_COLORS.get(p_label, (255, 255, 255)), scale=font_scale)
    _draw_text(image, f"head:    {t_label} ({t_angle:+.0f} deg)",
               (margin, line_gap * 2),
               HEAD_TILT_COLORS.get(t_label, (255, 255, 255)), scale=font_scale)

    if debug_features is not None:
        f = debug_features
        debug_scale = _scale(image, 0.55)
        debug_gap = max(22, round(_scale(image, 22)))
        lines = [
            f"H/W:        {f.body_aspect_h_over_w:.2f}" if f.body_aspect_h_over_w is not None else "H/W:        -",
            f"knee:       {f.back_knee_angle_deg:.0f} deg" if f.back_knee_angle_deg is not None else "knee:       -",
            f"head/grnd:  {f.head_above_ground_ratio:.2f}" if f.head_above_ground_ratio is not None else "head/grnd:  -",
            f"trunk/grnd: {f.trunk_above_ground_ratio:.2f}" if f.trunk_above_ground_ratio is not None else "trunk/grnd: -",
            f"hip/grnd:   {f.hip_above_ground_ratio:.2f}" if f.hip_above_ground_ratio is not None else "hip/grnd:   -",
            f"spine:      {f.spine_pitch_deg:+.0f} deg" if f.spine_pitch_deg is not None else "spine:      -",
            f"ground:     {'paw' if f.ground_from_paws else 'kp'}",
        ]
        for k, line in enumerate(lines):
            _draw_text(image, line, (margin, line_gap * 3 + k * debug_gap),
                       (220, 220, 220), scale=debug_scale)

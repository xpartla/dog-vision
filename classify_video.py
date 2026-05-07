"""Phase-2 entry point: read DLC predictions for a video, classify posture
and head tilt per frame, and write an annotated output video.

Run after `process_video.py` has produced an .h5 prediction file.

Examples:
    # Auto-discover the .h5 (assumes it's in output/ next to the labeled video)
    python classify_video.py samples/Video1.mp4

    # Explicit predictions path
    python classify_video.py samples/Video1.mp4 --predictions output/Video1<scorer>.h5

    # Print which keypoint names DLC actually wrote — useful if the
    # constants in posture.py don't match your DLC version
    python classify_video.py samples/Video1.mp4 --list-keypoints

    # Show debug overlay (per-feature values printed on each frame)
    python classify_video.py samples/Video1.mp4 --debug
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from posture import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    LabelSmoother,
    classify_head_tilt,
    classify_posture,
    compute_posture_features,
    list_keypoint_names,
    load_keypoint_frames,
)


def find_predictions_h5(video: Path, predictions_dir: Path) -> Optional[Path]:
    """Look for a DLC .h5 that corresponds to `video` in `predictions_dir`."""
    stem = video.stem
    candidates = sorted(predictions_dir.glob(f"{stem}*.h5"))
    return candidates[0] if candidates else None


def _draw_label(frame: np.ndarray, text: str, origin: tuple[int, int],
                color: tuple[int, int, int], scale: float = 0.7) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


POSTURE_COLORS = {
    "sitting": (0, 200, 255),    # amber
    "standing": (0, 255, 0),     # green
    "lying": (255, 100, 100),    # blue-ish
    "unknown": (160, 160, 160),  # grey
}

HEAD_TILT_COLORS = {
    "upright": (200, 200, 200),
    "tilt_left": (0, 200, 255),
    "tilt_right": (255, 100, 0),
    "unknown": (160, 160, 160),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to the input video")
    parser.add_argument("--predictions", type=Path, default=None,
                        help="DLC .h5 with keypoints (auto-discovered in --predictions-dir if omitted)")
    parser.add_argument("--predictions-dir", type=Path, default=Path("output"),
                        help="Directory to search for the .h5 if --predictions is omitted")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output video path (default: output/<stem>_posture.mp4)")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                        help="Minimum keypoint likelihood to consider a keypoint visible")
    parser.add_argument("--smooth-window", type=int, default=10,
                        help="Sliding-window size (frames) for label smoothing")
    parser.add_argument("--list-keypoints", action="store_true",
                        help="Print the bodypart names found in the .h5 and exit")
    parser.add_argument("--debug", action="store_true",
                        help="Overlay per-feature numeric values on each frame")
    args = parser.parse_args()

    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")

    h5_path = args.predictions or find_predictions_h5(args.video, args.predictions_dir)
    if h5_path is None or not h5_path.exists():
        raise SystemExit(
            f"No predictions .h5 found. Run `process_video.py {args.video}` first, "
            f"or pass --predictions explicitly."
        )

    if args.list_keypoints:
        names = list_keypoint_names(h5_path)
        print(f"{len(names)} keypoints in {h5_path.name}:")
        for n in names:
            print(f"  - {n}")
        return

    print(f"Loading predictions from {h5_path}")
    frames = load_keypoint_frames(h5_path, confidence_threshold=args.confidence)
    print(f"  {len(frames)} frames")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video {args.video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if n_video_frames != len(frames):
        print(f"  warning: video has {n_video_frames} frames but predictions has {len(frames)}; "
              f"will process min({n_video_frames}, {len(frames)})")

    output = args.output or (args.predictions_dir / f"{args.video.stem}_posture.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))

    posture_smoother = LabelSmoother(window=args.smooth_window)
    head_tilt_smoother = LabelSmoother(window=args.smooth_window)

    n = min(n_video_frames, len(frames))
    for i in range(n):
        ret, img = cap.read()
        if not ret:
            break

        frame = frames[i]
        features = compute_posture_features(frame)
        raw_posture, posture_score = classify_posture(features)
        raw_tilt, tilt_angle = classify_head_tilt(frame)

        posture = posture_smoother.push(raw_posture)
        tilt = head_tilt_smoother.push(raw_tilt)

        _draw_label(img,
                    f"posture: {posture} ({posture_score:.2f})",
                    (12, 30),
                    POSTURE_COLORS.get(posture, (255, 255, 255)),
                    scale=0.8)
        _draw_label(img,
                    f"head:    {tilt} ({tilt_angle:+.0f} deg)",
                    (12, 60),
                    HEAD_TILT_COLORS.get(tilt, (255, 255, 255)),
                    scale=0.8)

        if args.debug:
            lines = [
                f"H/W:      {features.body_aspect_h_over_w:.2f}" if features.body_aspect_h_over_w is not None else "H/W:      -",
                f"knee:     {features.back_knee_angle_deg:.0f} deg" if features.back_knee_angle_deg is not None else "knee:     -",
                f"hip/spine:{features.hip_height_ratio:.2f}" if features.hip_height_ratio is not None else "hip/spine:-",
                f"spine:    {features.spine_pitch_deg:+.0f} deg" if features.spine_pitch_deg is not None else "spine:    -",
                f"raw post: {raw_posture}",
                f"raw tilt: {raw_tilt}",
            ]
            for k, line in enumerate(lines):
                _draw_label(img, line, (12, 100 + k * 22), (220, 220, 220), scale=0.55)

        writer.write(img)

    cap.release()
    writer.release()

    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()

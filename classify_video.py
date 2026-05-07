"""Phase-2 entry point: read DLC predictions for a video, classify posture
and head tilt per frame, and write an annotated output video with the full
keypoint overlay drawn on top.

Run after `process_video.py` has produced an .h5 prediction file.

Examples:
    # Auto-discover the .h5 (assumes it's in output/ next to the labeled video)
    python classify_video.py samples/Video1.mp4

    # Explicit predictions path
    python classify_video.py samples/Video1.mp4 --predictions output/Video1<scorer>.h5

    # Print which keypoint names DLC actually wrote
    python classify_video.py samples/Video1.mp4 --list-keypoints

    # Show debug overlay (per-feature numeric values)
    python classify_video.py samples/Video1.mp4 --debug

    # Disable keypoint smoothing (e.g. to compare jitter before/after)
    python classify_video.py samples/Video1.mp4 --no-smooth-keypoints
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2

from overlay import draw_overlay
from posture import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    KeypointSmoother,
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
    parser.add_argument("--no-smooth-keypoints", action="store_true",
                        help="Disable 1-Euro smoothing of keypoint trajectories")
    parser.add_argument("--smooth-mincutoff", type=float, default=1.0,
                        help="1-Euro filter min cutoff Hz; lower = more smoothing")
    parser.add_argument("--smooth-beta", type=float, default=0.5,
                        help="1-Euro filter beta; higher = more responsive to fast motion")
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
    raw_frames = load_keypoint_frames(h5_path, confidence_threshold=args.confidence)
    print(f"  {len(raw_frames)} frames")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video {args.video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if n_video_frames != len(raw_frames):
        print(f"  warning: video has {n_video_frames} frames but predictions has "
              f"{len(raw_frames)}; will process min({n_video_frames}, {len(raw_frames)})")

    output = args.output or (args.predictions_dir / f"{args.video.stem}_posture.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))

    kp_smoother: Optional[KeypointSmoother] = None
    if not args.no_smooth_keypoints:
        kp_smoother = KeypointSmoother(fps=fps,
                                       mincutoff=args.smooth_mincutoff,
                                       beta=args.smooth_beta)
    posture_smoother = LabelSmoother(window=args.smooth_window)
    head_tilt_smoother = LabelSmoother(window=args.smooth_window)

    n = min(n_video_frames, len(raw_frames))
    for i in range(n):
        ret, img = cap.read()
        if not ret:
            break

        frame = raw_frames[i]
        if kp_smoother is not None:
            frame = kp_smoother.smooth(frame)

        features = compute_posture_features(frame)
        raw_posture, posture_score = classify_posture(features)
        raw_tilt, tilt_angle = classify_head_tilt(frame)

        posture_label = posture_smoother.push(raw_posture)
        tilt_label = head_tilt_smoother.push(raw_tilt)

        draw_overlay(
            img,
            frame,
            posture=(posture_label, posture_score),
            head_tilt=(tilt_label, tilt_angle),
            debug_features=features if args.debug else None,
        )
        writer.write(img)

    cap.release()
    writer.release()

    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()

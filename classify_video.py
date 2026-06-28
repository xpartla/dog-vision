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
import csv
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from orientation import estimate_orientation
from overlay import draw_overlay
from posture import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    KeypointSmoother,
    LabelSmoother,
    LearnedPostureClassifier,
    classify_posture,
    compute_posture_features,
    list_keypoint_names,
    load_keypoint_frames,
    merge_short_segments,
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
    parser.add_argument("--smooth-window", type=int, default=30,
                        help="Sliding-window size (frames) for label smoothing")
    parser.add_argument("--min-segment", type=int, default=25,
                        help="Minimum segment length (frames) for post-processing; "
                             "segments shorter than this are merged into their larger "
                             "neighbour.  Set to 0 to disable.")
    parser.add_argument("--no-smooth-keypoints", action="store_true",
                        help="Disable 1-Euro smoothing of keypoint trajectories")
    parser.add_argument("--smooth-mincutoff", type=float, default=1.0,
                        help="1-Euro filter min cutoff Hz; lower = more smoothing")
    parser.add_argument("--smooth-beta", type=float, default=0.5,
                        help="1-Euro filter beta; higher = more responsive to fast motion")
    parser.add_argument("--list-keypoints", action="store_true",
                        help="Print the bodypart names found in the .h5 and exit")
    parser.add_argument("--posture-model", type=Path, default=None,
                        help="Trained posture model (.joblib from train_posture.py). "
                             "Uses the learned, viewpoint-robust classifier instead "
                             "of the hand-tuned geometric rules.")
    parser.add_argument("--debug", action="store_true",
                        help="Overlay per-feature numeric values on each frame")
    parser.add_argument("--dump-features", type=Path, default=None,
                        help="Write per-frame feature values + labels to a CSV "
                             "for tuning the classifier thresholds on real footage")
    parser.add_argument("--rerender", action="store_true",
                        help="Skip classification; reload labels from the sidecar JSON "
                             "saved by a previous run and re-render the overlay only. "
                             "Much faster than a full re-run when only overlay visuals changed.")
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

    posture_clf: Optional[LearnedPostureClassifier] = None
    if args.posture_model is not None:
        if not args.posture_model.exists():
            raise SystemExit(f"Posture model not found: {args.posture_model}")
        posture_clf = LearnedPostureClassifier(args.posture_model)
        # Match the keypoint visibility threshold the model was trained with.
        if abs(posture_clf.confidence_threshold - args.confidence) > 1e-6:
            print(f"  using model's training confidence threshold "
                  f"{posture_clf.confidence_threshold:.2f} (overrides --confidence)")
            args.confidence = posture_clf.confidence_threshold
        print(f"Using learned posture model {args.posture_model}")

    output = args.output or (args.predictions_dir / f"{args.video.stem}_posture.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    labels_cache = output.with_suffix(".labels.json")

    print(f"Loading predictions from {h5_path}")
    raw_frames = load_keypoint_frames(h5_path, confidence_threshold=args.confidence)
    print(f"  {len(raw_frames)} frames")

    # --- probe video dimensions without decoding all frames ---
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if n_video_frames != len(raw_frames):
        print(f"  warning: video has {n_video_frames} frames but predictions has "
              f"{len(raw_frames)}; will process min({n_video_frames}, {len(raw_frames)})")

    n = min(n_video_frames, len(raw_frames))

    # --- build per-frame smoothed keypoints (always fast) ---
    kp_smoother: Optional[KeypointSmoother] = None
    if not args.no_smooth_keypoints:
        kp_smoother = KeypointSmoother(fps=fps,
                                       mincutoff=args.smooth_mincutoff,
                                       beta=args.smooth_beta)
    smoothed_frames = []
    for i in range(n):
        frame = raw_frames[i]
        if kp_smoother is not None:
            frame = kp_smoother.smooth(frame)
        smoothed_frames.append(frame)

    # --- classification: load from cache or run ---
    if args.rerender:
        if not labels_cache.exists():
            raise SystemExit(
                f"No labels cache found at {labels_cache}. "
                "Run without --rerender first to generate it."
            )
        print(f"Rerender mode: loading labels from {labels_cache}")
        rows = json.loads(labels_cache.read_text())
        per_frame_labels = [
            (r["posture_label"], r["posture_score"])
            for r in rows
        ]
    else:
        posture_smoother = LabelSmoother(window=args.smooth_window)
        dump_writer = None
        dump_file = None
        if args.dump_features is not None:
            args.dump_features.parent.mkdir(parents=True, exist_ok=True)
            dump_file = args.dump_features.open("w", newline="")
            dump_writer = csv.writer(dump_file)
            dump_writer.writerow([
                "frame", "posture_raw", "posture_smoothed", "posture_score",
                "head_above_ground", "trunk_above_ground", "hip_above_ground",
                "spine_pitch_deg", "back_knee_deg", "body_aspect_hw",
                "ground_from_paws",
                "orient_spine_deg", "orient_bilateral_conf", "orient_confidence",
            ])

        # Batch-classify all frames at once (avoids per-call RF overhead).
        if posture_clf is not None:
            raw_clf_results = posture_clf.classify_frames(smoothed_frames)
        else:
            raw_clf_results = None

        per_frame_labels = []
        for i, frame in enumerate(smoothed_frames):
            features = compute_posture_features(frame)
            if raw_clf_results is not None:
                raw_posture, posture_score = raw_clf_results[i]
            else:
                raw_posture, posture_score = classify_posture(features)
            posture_label = posture_smoother.push(raw_posture)
            per_frame_labels.append((posture_label, posture_score))

            if dump_writer is not None:
                orientation = estimate_orientation(frame)
                fmt = lambda v: "" if v is None else round(v, 4)
                dump_writer.writerow([
                    i, raw_posture, posture_label, round(posture_score, 4),
                    fmt(features.head_above_ground_ratio),
                    fmt(features.trunk_above_ground_ratio),
                    fmt(features.hip_above_ground_ratio),
                    fmt(features.spine_pitch_deg),
                    fmt(features.back_knee_angle_deg),
                    fmt(features.body_aspect_h_over_w),
                    int(features.ground_from_paws),
                    round(orientation.spine_angle_deg, 1),
                    round(orientation.bilateral_conf, 3),
                    round(orientation.confidence, 3),
                ])

        if dump_file is not None:
            dump_file.close()
            print(f"Wrote feature dump {args.dump_features.resolve()}")

        # Post-process: merge segments that are too short (jitter suppression).
        if args.min_segment > 0:
            raw_labels = [pl for pl, _ps in per_frame_labels]
            merged = merge_short_segments(raw_labels, min_length=args.min_segment)
            per_frame_labels = [
                (merged[i], ps) for i, (_pl, ps) in enumerate(per_frame_labels)
            ]

        labels_cache.write_text(json.dumps([
            {"posture_label": pl, "posture_score": ps}
            for pl, ps in per_frame_labels
        ]))
        print(f"Saved labels cache → {labels_cache}")

    # --- render: decode with ffmpeg pipe, encode directly to H.264 ---
    frame_bytes = width * height * 3
    decode = subprocess.Popen(
        ["ffmpeg", "-i", str(args.video), "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-vframes", str(n), "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    encode = subprocess.Popen(
        ["ffmpeg", "-y",
         "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
         "-c:v", "libx264", "-crf", "23", "-preset", "fast",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )

    for i, (frame, (posture_label, posture_score)) in \
            enumerate(zip(smoothed_frames, per_frame_labels)):
        raw = decode.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        img = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()
        orientation = estimate_orientation(frame)
        draw_overlay(
            img, frame,
            posture=(posture_label, posture_score),
            orientation=orientation,
        )
        encode.stdin.write(img.tobytes())

    encode.stdin.close()
    encode.wait()
    decode.stdout.close()
    decode.wait()
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()

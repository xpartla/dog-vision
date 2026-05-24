"""Chunked live webcam loop with SuperAnimal-Quadruped + phase-2 overlays.

Captures ~1.5-second chunks from the webcam, runs SuperAnimal on each chunk,
loads the resulting keypoints, classifies posture and head tilt per frame
(with 1-Euro keypoint smoothing and majority-vote label smoothing that persist
across chunks), then plays back the captured chunk with the full overlay.

Net latency is roughly chunk-length plus inference time (a couple of seconds
on an RTX 2060/3060). True frame-by-frame real-time is a follow-up.

Example:
    python live_webcam.py
    python live_webcam.py --chunk-seconds 1.0 --no-posture
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import deeplabcut

from overlay import draw_overlay
from posture import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    KeypointSmoother,
    LabelSmoother,
    LearnedPostureClassifier,
    classify_head_tilt,
    classify_posture,
    compute_posture_features,
    load_keypoint_frames,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--chunk-seconds", type=float, default=1.5)
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Frame rate. If omitted, auto-detected from the camera (fallback 30).",
    )
    parser.add_argument("--model", default="hrnet_w32", choices=["hrnet_w32", "resnet_50"])
    parser.add_argument("--detector", default="fasterrcnn_resnet50_fpn_v2")
    parser.add_argument("--pcutoff", type=float, default=0.6)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                        help="Min keypoint likelihood for posture features")
    parser.add_argument("--smooth-window", type=int, default=10,
                        help="Majority-vote window size (frames) for posture/tilt labels")
    parser.add_argument("--no-smooth-keypoints", action="store_true",
                        help="Disable 1-Euro smoothing of keypoint trajectories")
    parser.add_argument("--smooth-mincutoff", type=float, default=1.0)
    parser.add_argument("--smooth-beta", type=float, default=0.5)
    parser.add_argument("--posture-model", type=Path, default=None,
                        help="Trained posture model (.joblib). Uses the learned, "
                             "viewpoint-robust classifier instead of the geometric rules.")
    parser.add_argument("--no-posture", action="store_true",
                        help="Skip phase-2 classification and just draw keypoints")
    parser.add_argument("--debug", action="store_true",
                        help="Overlay per-feature numeric values on each frame")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.fps is not None:
        fps, fps_source = args.fps, "user override"
    else:
        reported = cap.get(cv2.CAP_PROP_FPS)
        if 5.0 <= reported <= 120.0:
            fps, fps_source = reported, "camera-reported"
        else:
            fps, fps_source = 30.0, f"fallback (camera reported {reported:.1f})"

    chunk_frames = max(1, int(args.chunk_seconds * fps))
    print(
        f"Live mode: {chunk_frames} frames per chunk "
        f"({args.chunk_seconds}s @ {fps:.1f}fps, {fps_source}), {width}x{height}"
    )
    print("Press q in the display window to quit.")

    posture_clf = None
    if args.posture_model is not None:
        if not args.posture_model.exists():
            raise SystemExit(f"Posture model not found: {args.posture_model}")
        posture_clf = LearnedPostureClassifier(args.posture_model)
        if abs(posture_clf.confidence_threshold - args.confidence) > 1e-6:
            args.confidence = posture_clf.confidence_threshold
        print(f"Using learned posture model {args.posture_model} "
              f"(confidence {args.confidence:.2f})")

    # Persistent smoothers — state carries across chunks for visual continuity.
    kp_smoother = None
    if not args.no_smooth_keypoints:
        kp_smoother = KeypointSmoother(fps=fps,
                                       mincutoff=args.smooth_mincutoff,
                                       beta=args.smooth_beta)
    posture_smoother = LabelSmoother(window=args.smooth_window)
    tilt_smoother = LabelSmoother(window=args.smooth_window)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    with tempfile.TemporaryDirectory(prefix="dogvision-") as tmp_root:
        tmp_root = Path(tmp_root)
        chunk_idx = 0
        stop = False

        while not stop:
            chunk_dir = tmp_root / f"chunk_{chunk_idx}"
            chunk_dir.mkdir()
            chunk_path = chunk_dir / "chunk.mp4"

            writer = cv2.VideoWriter(str(chunk_path), fourcc, fps, (width, height))
            t0 = time.time()
            captured = 0
            for _ in range(chunk_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
                captured += 1
            writer.release()
            t_capture = time.time() - t0

            if captured == 0:
                print("Camera read failed; exiting.")
                break

            t0 = time.time()
            try:
                deeplabcut.video_inference_superanimal(
                    videos=[str(chunk_path)],
                    superanimal_name="superanimal_quadruped",
                    model_name=args.model,
                    detector_name=args.detector,
                    dest_folder=str(chunk_dir),
                    pcutoff=args.pcutoff,
                )
            except Exception as exc:
                print(f"chunk {chunk_idx}: inference failed: {exc}")
                shutil.rmtree(chunk_dir, ignore_errors=True)
                chunk_idx += 1
                continue
            t_infer = time.time() - t0

            h5_files = list(chunk_dir.glob("*.h5"))
            kp_frames = []
            if h5_files:
                try:
                    kp_frames = load_keypoint_frames(h5_files[0],
                                                     confidence_threshold=args.confidence)
                except Exception as exc:
                    print(f"chunk {chunk_idx}: failed to read predictions: {exc}")

            print(f"chunk {chunk_idx}: capture {t_capture:.2f}s, infer {t_infer:.2f}s, "
                  f"frames={len(kp_frames)}")

            source = cv2.VideoCapture(str(chunk_path))
            try:
                for i in range(captured):
                    ret, img = source.read()
                    if not ret:
                        break

                    kpf = kp_frames[i] if i < len(kp_frames) else None
                    if kpf is not None and kp_smoother is not None:
                        kpf = kp_smoother.smooth(kpf)

                    if args.no_posture or kpf is None:
                        posture = ("unknown", 0.0)
                        tilt = ("unknown", 0.0)
                        features = None
                    else:
                        features = compute_posture_features(kpf)
                        if posture_clf is not None:
                            raw_p, score_p = posture_clf.classify(kpf)
                        else:
                            raw_p, score_p = classify_posture(features)
                        raw_t, ang_t = classify_head_tilt(kpf)
                        posture = (posture_smoother.push(raw_p), score_p)
                        tilt = (tilt_smoother.push(raw_t), ang_t)

                    draw_overlay(
                        img,
                        kpf,
                        posture=posture,
                        head_tilt=tilt,
                        debug_features=features if args.debug else None,
                    )

                    cv2.imshow("Dog Vision (q to quit)", img)
                    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
                        stop = True
                        break
            finally:
                source.release()

            shutil.rmtree(chunk_dir, ignore_errors=True)
            chunk_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

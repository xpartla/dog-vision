"""Live webcam loop: capture → inference → display, each on its own thread.

The capture thread reads frames at the camera's native FPS. The inference thread
blocks waiting for frames, runs SuperAnimalInferencer (weights loaded once at
startup) and classifies posture. The display thread reads the latest live frame and
overlays the last-known keypoints, giving smooth 30-fps video even while inference
is running. Keypoints fade slightly when they are stale.

Net latency is ~inference time per single frame (~100-300ms on GPU) instead of the
previous chunk-length + inference (~2s).

Example:
    python live_webcam.py
    python live_webcam.py --posture-model posture_model_mlp.joblib
    python live_webcam.py --no-posture
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
from pathlib import Path

import cv2

from inferencer import SuperAnimalInferencer
from orientation import estimate_orientation
from overlay import draw_overlay
from posture import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    KeypointSmoother,
    LabelSmoother,
    LearnedPostureClassifier,
    classify_posture,
    compute_posture_features,
)

# Seconds of stale keypoints before the skeleton starts fading.
_STALE_FADE_START = 0.3
# Seconds at which the skeleton reaches minimum opacity.
_STALE_FADE_END = 2.0
_SKELETON_MIN_ALPHA = 0.2


def _stale_alpha(stale_secs: float) -> float:
    if stale_secs <= _STALE_FADE_START:
        return 1.0
    if stale_secs >= _STALE_FADE_END:
        return _SKELETON_MIN_ALPHA
    t = (stale_secs - _STALE_FADE_START) / (_STALE_FADE_END - _STALE_FADE_START)
    return 1.0 - t * (1.0 - _SKELETON_MIN_ALPHA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", type=Path, default=None,
                        help="Use a video file instead of the webcam (for testing).")
    parser.add_argument("--loop", action="store_true",
                        help="Loop the video file instead of stopping at the end.")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Display frame rate. If omitted, auto-detected from source (fallback 30).",
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
                             "viewpoint-robust classifier instead of geometric rules.")
    parser.add_argument("--no-posture", action="store_true",
                        help="Skip phase-2 classification and just draw keypoints")
    parser.add_argument("--debug", action="store_true",
                        help="Overlay per-feature numeric values on each frame")
    parser.add_argument("--max-individuals", type=int, default=10)
    args = parser.parse_args()

    # --- Capture source setup ---
    if args.video is not None:
        if not args.video.exists():
            raise SystemExit(f"Video file not found: {args.video}")
        cap = cv2.VideoCapture(str(args.video))
        source_desc = f"video file {args.video.name}"
    else:
        cap = cv2.VideoCapture(args.camera)
        source_desc = f"camera index {args.camera}"
    if not cap.isOpened():
        raise SystemExit(f"Could not open {source_desc}")

    if args.fps is not None:
        fps, fps_source = args.fps, "user override"
    else:
        reported = cap.get(cv2.CAP_PROP_FPS)
        if 5.0 <= reported <= 120.0:
            fps, fps_source = reported, "source-reported"
        else:
            fps, fps_source = 30.0, f"fallback (source reported {reported:.1f})"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Source: {source_desc}  {width}x{height} @ {fps:.1f}fps ({fps_source})")

    # --- Posture classifier ---
    posture_clf = None
    if args.posture_model is not None:
        if not args.posture_model.exists():
            raise SystemExit(f"Posture model not found: {args.posture_model}")
        posture_clf = LearnedPostureClassifier(args.posture_model)
        if abs(posture_clf.confidence_threshold - args.confidence) > 1e-6:
            args.confidence = posture_clf.confidence_threshold
        print(f"Using learned posture model {args.posture_model} "
              f"(confidence {args.confidence:.2f})")

    # --- Smoothers (state persists across frames) ---
    kp_smoother = None
    if not args.no_smooth_keypoints:
        kp_smoother = KeypointSmoother(fps=fps,
                                       mincutoff=args.smooth_mincutoff,
                                       beta=args.smooth_beta)
    posture_smoother = LabelSmoother(window=args.smooth_window)

    # --- Load inference runners once ---
    print("Loading SuperAnimal model weights (downloading if needed)...")
    inferencer = SuperAnimalInferencer(
        model_name=args.model,
        detector_name=args.detector,
        max_individuals=args.max_individuals,
        confidence_threshold=args.confidence,
    )
    print("Model loaded. Starting live feed. Press q to quit.")

    # --- Shared state ---
    stop_event = threading.Event()

    # Latest raw frame from camera (for display thread).
    _frame_lock = threading.Lock()
    _latest_frame: list = [None]  # list as mutable container

    # Queue from capture → inference (depth=1; old frames dropped automatically).
    infer_queue: queue.Queue = queue.Queue(maxsize=1)

    # Latest inference result (for display thread).
    _result_lock = threading.Lock()
    _latest_result: list = [None]  # (kpf, posture, orientation, features, timestamp)

    # --- Capture thread ---
    def capture_loop() -> None:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                if args.loop and args.video is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                print("End of source; stopping.")
                stop_event.set()
                break
            with _frame_lock:
                _latest_frame[0] = frame

            # Drop old frame if inference hasn't consumed yet; put new one.
            try:
                infer_queue.put_nowait(frame.copy())
            except queue.Full:
                try:
                    infer_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    infer_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass

    # --- Inference thread ---
    def inference_loop() -> None:
        while not stop_event.is_set():
            try:
                frame = infer_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            t0 = time.time()
            try:
                kp_frames = inferencer.infer(frame)
            except Exception as exc:
                print(f"Inference error: {exc}")
                continue
            infer_ms = (time.time() - t0) * 1000

            kpf = kp_frames[0] if kp_frames else None

            if kpf is not None and kp_smoother is not None:
                kpf = kp_smoother.smooth(kpf)

            if args.no_posture or kpf is None:
                posture = ("unknown", 0.0)
                features = None
                orientation = None
            else:
                features = compute_posture_features(kpf)
                if posture_clf is not None:
                    raw_p, score_p = posture_clf.classify(kpf)
                else:
                    raw_p, score_p = classify_posture(features)
                orientation = estimate_orientation(kpf)
                posture = (posture_smoother.push(raw_p), score_p)

            print(f"infer {infer_ms:.0f}ms  posture={posture[0]}")

            with _result_lock:
                _latest_result[0] = (kpf, posture, orientation, features, time.time())

    # --- Display thread ---
    def display_loop() -> None:
        frame_delay_ms = max(1, int(1000 / fps))

        while not stop_event.is_set():
            with _frame_lock:
                frame = _latest_frame[0]

            if frame is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                continue

            img = frame.copy()

            with _result_lock:
                result = _latest_result[0]

            if result is not None:
                kpf, posture, orientation, features, result_ts = result
                stale_secs = time.time() - result_ts
            else:
                kpf, posture, orientation, features = None, ("unknown", 0.0), None, None
                stale_secs = 9999.0

            draw_overlay(
                img,
                kpf,
                posture=posture,
                orientation=orientation,
                debug_features=features if args.debug else None,
                skeleton_alpha=_stale_alpha(stale_secs),
            )

            cv2.imshow("Dog Vision (q to quit)", img)
            if cv2.waitKey(frame_delay_ms) & 0xFF == ord("q"):
                stop_event.set()
                break

    # --- Launch threads ---
    threads = [
        threading.Thread(target=capture_loop, name="capture", daemon=True),
        threading.Thread(target=inference_loop, name="inference", daemon=True),
        threading.Thread(target=display_loop, name="display", daemon=True),
    ]
    for t in threads:
        t.start()

    # Main thread just waits for stop signal.
    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()

    for t in threads:
        t.join(timeout=3.0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""Chunked live webcam loop with SuperAnimal-Quadruped inference.

Captures ~1.5-second chunks from the webcam, runs SuperAnimal on each chunk,
then plays back the annotated chunk. Net latency is roughly chunk-length plus
inference time (a couple of seconds on an RTX 2060/3060).

This is "live enough" to verify the pose pipeline against a moving dog. True
frame-by-frame real-time inference (where the model stays resident in memory)
is a follow-up; see README.

Example:
    python live_webcam.py
    python live_webcam.py --chunk-seconds 1.0 --model hrnet_w32
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import deeplabcut


def find_labeled_video(folder: Path, exclude: Path) -> Path | None:
    """Return the labeled MP4 produced by DLC, ignoring the input chunk."""
    for candidate in folder.glob("*.mp4"):
        if candidate.resolve() != exclude.resolve():
            return candidate
    return None


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

            labeled = find_labeled_video(chunk_dir, exclude=chunk_path)
            print(
                f"chunk {chunk_idx}: capture {t_capture:.2f}s, infer {t_infer:.2f}s, "
                f"labeled={'yes' if labeled else 'no'}"
            )

            playback_path = labeled if labeled is not None else chunk_path
            playback = cv2.VideoCapture(str(playback_path))
            try:
                while True:
                    ret, frame = playback.read()
                    if not ret:
                        break
                    cv2.imshow("Dog Vision (q to quit)", frame)
                    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
                        stop = True
                        break
            finally:
                playback.release()

            shutil.rmtree(chunk_dir, ignore_errors=True)
            chunk_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

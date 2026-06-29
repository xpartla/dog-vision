"""Record raw footage from the webcam to a video file.

Useful for capturing dog footage on the run-laptop, then processing it
offline (on either machine) with process_video.py.

Example:
    python -m dogvision.tools.webcam_record --output samples/dog.mp4 --duration 30
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("samples/recording.mp4"))
    parser.add_argument("--duration", type=float, default=30.0, help="Seconds to record")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Frame rate. If omitted, auto-detected from the camera (fallback 30).",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

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

    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    print(f"Recording {args.duration}s to {args.output} ({width}x{height} @ {fps:.1f}fps, {fps_source})")
    print("Press q in the preview window to stop early.")

    start = time.time()
    try:
        while time.time() - start < args.duration:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed; stopping.")
                break
            writer.write(frame)
            cv2.imshow("Recording (q to stop)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()

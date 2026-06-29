"""Run SuperAnimal-Quadruped on a video file and write an annotated copy.

This is the safe, documented inference path. Use it on either machine.

Example:
    python -m dogvision.tools.process_video samples/dog.mp4
    python -m dogvision.tools.process_video samples/dog.mp4 --output-dir output --model hrnet_w32
"""

from __future__ import annotations

import argparse
from pathlib import Path

import deeplabcut


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to input video")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Where to write the annotated video and prediction files (default: output/)",
    )
    parser.add_argument(
        "--model",
        default="hrnet_w32",
        choices=["hrnet_w32", "resnet_50"],
        help="SuperAnimal pose model variant (default: hrnet_w32)",
    )
    parser.add_argument(
        "--detector",
        default="fasterrcnn_resnet50_fpn_v2",
        help="Object detector for the dog bounding box",
    )
    parser.add_argument(
        "--pcutoff",
        type=float,
        default=0.6,
        help="Confidence threshold for drawing keypoints (default: 0.6)",
    )
    parser.add_argument(
        "--video-adapt",
        action="store_true",
        help="Fine-tune the model on the video using its own pseudo-labels. "
             "Significantly reduces keypoint jitter at the cost of much longer "
             "runtime. Recommended for final renders, not for iterative tuning.",
    )
    parser.add_argument(
        "--video-adapt-batch-size",
        type=int,
        default=None,
        help="Adaptation training batch size. DLC's default is 8, which OOMs "
             "the detector training on a 6-12 GB GPU. Try 1 or 2 if you hit "
             "CUDA out-of-memory during --video-adapt.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for inference: 'auto' (default, uses GPU if available), "
             "'cpu', or 'cuda'. Use 'cpu' if CUDA causes a segfault.",
    )
    args = parser.parse_args()

    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    extra_kwargs = {}
    if args.video_adapt_batch_size is not None:
        extra_kwargs["video_adapt_batch_size"] = args.video_adapt_batch_size

    deeplabcut.video_inference_superanimal(
        videos=[str(args.video)],
        superanimal_name="superanimal_quadruped",
        model_name=args.model,
        detector_name=args.detector,
        dest_folder=str(args.output_dir),
        pcutoff=args.pcutoff,
        video_adapt=args.video_adapt,
        device=args.device,
        **extra_kwargs,
    )

    print(f"\nDone. Annotated outputs in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

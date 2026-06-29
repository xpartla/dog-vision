"""Turn labeled keypoint clips into a training table for the posture model.

Expected layout — one subfolder per posture label, each containing the DLC
`.h5` prediction files for clips of the dog in that posture:

    data/
      standing/  clipA.h5  clipB.h5 ...
      sitting/   clipC.h5 ...
      lying/     clipD.h5 ...

Produce the `.h5` files first with `process_video.py` on each clip, then move
them into the label folders. Filming tip: record each clip with the dog holding
a *single* posture, so the whole clip shares one label — cheap, accurate
labeling. Vary camera angle, height, distance, and the dog's orientation across
clips; that variety is exactly what makes the learned model viewpoint-robust.

Each clip becomes a "group": train/test splitting is done by group (in
`train_posture.py`) so highly-correlated frames from one clip never straddle the
split and inflate the score.

Example:
    python -m dogvision.tools.build_dataset data/ --out dataset.npz --stride 2 --augment-flip
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dogvision import pose_features as pf
from dogvision.posture import DEFAULT_CONFIDENCE_THRESHOLD, load_keypoint_frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", type=Path,
                        help="Directory with one subfolder per label, each holding .h5 files")
    parser.add_argument("--out", type=Path, default=Path("dataset.npz"),
                        help="Output .npz path (default: dataset.npz)")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                        help="Min keypoint likelihood to treat a keypoint as visible")
    parser.add_argument("--stride", type=int, default=1,
                        help="Keep every Nth frame (consecutive frames are highly "
                             "correlated; >1 reduces redundancy). Default 1.")
    parser.add_argument("--augment-flip", action="store_true",
                        help="Add a horizontally-mirrored copy of every sample "
                             "(kept in the same clip group so it doesn't leak)")
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.data_dir}")

    label_dirs = sorted(p for p in args.data_dir.iterdir() if p.is_dir())
    if not label_dirs:
        raise SystemExit(f"No label subfolders found in {args.data_dir}")

    X: list[np.ndarray] = []
    y: list[str] = []
    groups: list[str] = []
    per_label: dict[str, int] = {}

    for label_dir in label_dirs:
        label = label_dir.name
        h5_files = sorted(label_dir.rglob("*.h5"))
        if not h5_files:
            print(f"  [{label}] no .h5 files — run process_video.py first; skipping")
            continue
        for h5 in h5_files:
            try:
                frames = load_keypoint_frames(h5, confidence_threshold=args.confidence)
            except Exception as exc:
                print(f"  [{label}] {h5.name}: failed to read ({exc}); skipping")
                continue
            clip_id = f"{label}/{h5.stem}"
            kept = 0
            for i in range(0, len(frames), max(1, args.stride)):
                vec = pf.feature_vector(frames[i])
                if vec is None:
                    continue
                X.append(vec); y.append(label); groups.append(clip_id)
                kept += 1
                if args.augment_flip:
                    X.append(pf.flip_feature_vector(vec))
                    y.append(label); groups.append(clip_id)
            per_label[label] = per_label.get(label, 0) + kept
            print(f"  [{label}] {h5.name}: {kept} frames"
                  + (" (x2 with flip)" if args.augment_flip else ""))

    if not X:
        raise SystemExit("No usable frames extracted. Check confidence / inputs.")

    X_arr = np.stack(X).astype(np.float32)
    y_arr = np.asarray(y)
    groups_arr = np.asarray(groups)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        X=X_arr, y=y_arr, groups=groups_arr,
        feature_names=np.asarray(pf.FEATURE_NAMES),
        confidence_threshold=np.float32(args.confidence),
    )

    print(f"\nSaved {X_arr.shape[0]} samples x {X_arr.shape[1]} features to {args.out.resolve()}")
    print(f"  clips (groups): {len(set(groups))}")
    print("  per-label sample counts (pre-flip):")
    for lab, n in sorted(per_label.items()):
        print(f"    {lab:12} {n}")


if __name__ == "__main__":
    main()

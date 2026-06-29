"""SuperAnimalInferencer: load DLC model weights once, infer on individual BGR frames.

Replaces the per-chunk `video_inference_superanimal` call with a persistent object
that holds the detector + pose runners in memory and processes single frames on demand.
"""
from __future__ import annotations

import math

import numpy as np

from .posture import DEFAULT_CONFIDENCE_THRESHOLD, Frame, Keypoint


class SuperAnimalInferencer:
    """Persistent detector + pose runners for frame-by-frame inference.

    Args:
        superanimal_name: DLC SuperAnimal project name.
        model_name: Pose model architecture (e.g. "hrnet_w32").
        detector_name: Object detector name (e.g. "fasterrcnn_resnet50_fpn_v2").
        max_individuals: Max detections per frame.
        device: Torch device string or "auto".
        confidence_threshold: Min keypoint likelihood for posture features.

    Usage:
        inf = SuperAnimalInferencer()  # downloads weights once on first run
        frames = inf.infer(bgr_frame)  # list[Frame], one element per call
    """

    def __init__(
        self,
        superanimal_name: str = "superanimal_quadruped",
        model_name: str = "hrnet_w32",
        detector_name: str = "fasterrcnn_resnet50_fpn_v2",
        max_individuals: int = 10,
        device: str | None = "auto",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        from deeplabcut.pose_estimation_pytorch.modelzoo.utils import (
            get_super_animal_snapshot_path,
            load_super_animal_config,
        )
        from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners

        self.confidence_threshold = confidence_threshold

        config = load_super_animal_config(
            super_animal=superanimal_name,
            model_name=model_name,
            detector_name=detector_name,
            max_individuals=max_individuals,
            device=device,
        )

        pose_path = get_super_animal_snapshot_path(superanimal_name, model_name)
        det_path = get_super_animal_snapshot_path(superanimal_name, detector_name)

        self._bodyparts: list[str] = config["metadata"]["bodyparts"]

        self._pose_runner, self._detector_runner = get_inference_runners(
            model_config=config,
            snapshot_path=pose_path,
            max_individuals=max_individuals,
            num_bodyparts=len(self._bodyparts),
            num_unique_bodyparts=0,
            batch_size=1,
            detector_batch_size=1,
            detector_path=det_path,
        )

    def infer(self, bgr_frame: np.ndarray) -> list[Frame]:
        """Run detector + pose on a single BGR frame.

        Args:
            bgr_frame: Frame from OpenCV (BGR uint8 HxWx3).

        Returns:
            A one-element list containing a Frame with the best-detected individual.
        """
        rgb = bgr_frame[..., ::-1].copy()  # BGR → RGB, ensure contiguous

        if self._detector_runner is not None:
            det_preds = self._detector_runner.inference([rgb])
            if not det_preds:
                # No detector output this frame (e.g. no animal found).
                return [self._empty_frame()]
            bbox_ctx = {k: v for k, v in det_preds[0].items()
                        if k in ("bboxes", "bbox_scores")}
            bboxes = bbox_ctx.get("bboxes")
            if bboxes is None or len(bboxes) == 0:
                # No animal detected; skip pose inference (DLC raises on empty bboxes).
                return [self._empty_frame()]
            pose_input = [(rgb, bbox_ctx)]
        else:
            pose_input = [rgb]

        pose_preds = self._pose_runner.inference(pose_input)
        if not pose_preds:
            return [self._empty_frame()]
        return [self._to_frame(pose_preds[0])]

    def _empty_frame(self) -> Frame:
        return Frame(keypoints={}, confidence_threshold=self.confidence_threshold)

    def _to_frame(self, pred: dict) -> Frame:
        poses = pred.get("bodyparts")
        if poses is None or len(poses) == 0:
            return self._empty_frame()

        if poses.ndim == 2:
            poses = poses[np.newaxis]

        # Pick the individual with the highest mean keypoint confidence.
        mean_conf = np.nanmean(poses[:, :, 2], axis=1)
        best = int(np.argmax(mean_conf))
        kp_arr = poses[best]  # (n_bodyparts, 3): x, y, likelihood

        keypoints: dict[str, Keypoint] = {}
        for i, name in enumerate(self._bodyparts):
            x, y, lik = float(kp_arr[i, 0]), float(kp_arr[i, 1]), float(kp_arr[i, 2])
            if math.isnan(x) or math.isnan(y) or math.isnan(lik):
                continue
            keypoints[name] = Keypoint(x=x, y=y, confidence=lik)

        return Frame(keypoints=keypoints, confidence_threshold=self.confidence_threshold)

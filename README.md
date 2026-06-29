# dog-vision

**Real-time pose estimation and posture classification for dogs.** A markerless
computer-vision pipeline that tracks ~39 body keypoints on a dog from ordinary
video, then classifies its posture — sitting, standing, or lying — frame by
frame, robust to camera angle, fur, and the dog facing any direction.

<p align="center">
  <img src="assets/demo.gif" alt="Skeleton overlay with live posture label tracking a dog from standing to sitting" width="640">
</p>

<p align="center"><sub>Live skeleton overlay + posture label. The classifier follows in real time.</sub></p>

---

## What it does

- **Markerless pose tracking** — locates ~39 keypoints (ears, eyes, snout,
  spine, hips, four limbs, three tail landmarks) on a dog in each frame, using
  the [SuperAnimal-Quadruped](https://www.nature.com/articles/s41467-024-48792-2)
  model via [DeepLabCut](https://www.deeplabcut.org/).
- **Posture classification** — turns those raw keypoints into a `sitting` /
  `standing` / `lying` label, plus a head-tilt readout, on every frame.
- **Viewpoint-robust by design** — the classifier reads *relative geometry*
  (joint angles, body aspect ratio, spine pitch), not pixel positions, so it
  survives the dog rotating, walking toward or away from the camera, and a
  tripod set at an arbitrary height.
- **Graceful degradation** — when fur occludes a keypoint, the affected
  features are dropped and the model votes on what survives, falling back to
  `unknown` rather than guessing.
- **Runs offline or live** — batch-annotate recorded clips, or stream a live
  webcam overlay to the browser.

## How it works

```
 video frame
     │
     ▼
┌─────────────────────┐   ~39 keypoints + confidences
│ SuperAnimal-Quadruped│ ─────────────────────────────┐
│   (DeepLabCut)       │                               │
└─────────────────────┘                               ▼
                                            ┌───────────────────────┐
                                            │ feature engineering   │
                                            │ 128 viewpoint-robust   │
                                            │ features (angles,      │
                                            │ ratios, distances)     │
                                            └───────────┬───────────┘
                                                        ▼
                                            ┌───────────────────────┐
                                            │ posture classifier     │
                                            │ Random Forest / MLP    │
                                            └───────────┬───────────┘
                                                        ▼
                                            ┌───────────────────────┐
                                            │ temporal smoothing     │
                                            │ (sliding-window vote)  │
                                            └───────────┬───────────┘
                                                        ▼
                                                annotated frame
```

**Learned classifier.** A Random Forest (default) or MLP trained on labeled
clips — see the [comparison](#results) below.

**Feature engineering over raw coordinates.** Rather than feed pixel locations
to the model, the pipeline derives 128 features that are invariant to where the
dog is in the frame and which way it faces — back-knee angle, hip-above-paws,
spine pitch, body height/width ratio, eye-line vs. head axis, and more. This is
what lets a model trained on a handful of clips generalize to new camera setups.

**Honest evaluation.** Models are scored by **held-out clip**, never by
held-out frame — frames from the same clip are near-duplicates, so a frame
split would massively overstate accuracy. Reported numbers reflect
generalization to *unseen recordings*.

## Results

Trained on ~10k labeled frames from 48 clips across 3 postures, evaluated on
clips the model never saw during training (grouped split — see below). Both
classifiers train from the same 128-feature vectors:

| Model | Grouped 5-fold CV | Held-out test acc. | Macro F1 | Recall — lying / sitting / standing |
|---|---|---|---|---|
| **Random Forest** (shipped default) | **76%** (±8%) | **80%** | **0.81** | 0.72 / 0.97 / 0.82 |
| MLP (128→64) | 76% (±6%) | 75% | 0.75 | 0.68 / 0.98 / 0.72 |

The Random Forest generalizes better to unseen clips and is the default; the MLP
trains with slightly tighter cross-validation variance but trails on held-out
accuracy. Both nail `sitting` and share the same weak spot — `lying` ↔
`standing` confusion from elevated camera angles underrepresented in the
training set, which is the clearest lever for the next round of data collection.

## Tech stack

`Python` · `DeepLabCut` (SuperAnimal-Quadruped, PyTorch backend) ·
`scikit-learn` (Random Forest, MLP) · `OpenCV` · `NumPy` / `pandas`

## Engineering notes

A few problems that were more interesting than they first looked:

- **Real-time on a chunked API.** DeepLabCut's high-level inference call reloads
  the model from disk per invocation. `live_webcam.py` works around this by
  processing short webcam chunks pipelined with playback (~2 s latency on an RTX
  3060).
- **Temporal smoothing.** Per-frame predictions flicker; a sliding-window
  majority vote suppresses jitter without adding noticeable lag.

## Repo layout

```
dog-vision/
├── dogvision/                # importable package
│   ├── posture.py            # Frame/Keypoint model, classifiers, smoothing
│   ├── pose_features.py      # 128-feature viewpoint-robust vector
│   ├── orientation.py        # body-orientation estimate
│   ├── overlay.py            # skeleton + label rendering
│   ├── inferencer.py         # keeps SuperAnimal resident for live use
│   ├── mjpeg_server.py       # browser preview for headless / WSL hosts
│   └── tools/                # command-line entry points (run via `python -m`)
│       ├── process_video.py  #   video → keypoints (.h5)
│       ├── classify_video.py #   annotate a video with posture labels
│       ├── live_webcam.py     #  live chunked webcam overlay
│       ├── build_dataset.py  #   labeled clips → dataset.npz
│       ├── train_posture.py  #   train RF / MLP, grouped CV, confusion matrix
│       └── webcam_record.py  #   capture raw footage
├── models/                   # trained classifiers (committed, ready to run)
├── assets/                   # demo media
├── dataset.npz               # extracted training features
└── retrain.sh                # end-to-end: process clips → rebuild → retrain
```

## Running it

```bash
git clone https://github.com/xpartla/dog-vision.git
cd dog-vision
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or: pip install -e .

# 1. Pose estimation: video → keypoints (.h5 in output/)
python -m dogvision.tools.process_video samples/dog.mp4

# 2. Classify: keypoints → posture-labeled video (uses the trained RF model)
python -m dogvision.tools.classify_video samples/dog.mp4 \
    --posture-model models/posture_model.joblib
```

Every tool runs as `python -m dogvision.tools.<name>` (or, after
`pip install -e .`, as `dogvision-<name>`). Without `--posture-model`, the
classifier falls back to interpretable geometric rules. DeepLabCut downloads the
SuperAnimal weights (a few hundred MB) on first run; a CUDA-capable GPU is
recommended for live use, while CPU is fine for offline processing.

## Training your own model

The shipped models come from 48 single-posture clips. To retrain on your own
footage — film clips each holding one posture, where the folder name becomes the
label — chain the three stages (or just run `./retrain.sh`):

```bash
# 1. keypoints per clip, arranged as data/<label>/*.h5
python -m dogvision.tools.process_video data/sitting/clip.mp4 --output-dir data/sitting
# 2. clips → feature dataset (--stride skips near-duplicate frames; --augment-flip mirrors orientation)
python -m dogvision.tools.build_dataset data/ --out dataset.npz --stride 2 --augment-flip
# 3. train + evaluate, grouped by clip so scores reflect unseen-clip generalization
python -m dogvision.tools.train_posture dataset.npz --model rf --out models/posture_model.joblib
```

The classifier is only as viewpoint-robust as the data, so vary camera angle,
height, distance, and the dog's orientation across clips.

## Roadmap

1. ✅ Pose landmark overlay
2. ✅ Posture classification (sitting / standing / lying) + head tilt
3. ⬜ Gaze / attention proxy via head + snout direction
4. ⬜ Excitement meter: body velocity, head oscillation, tail movement

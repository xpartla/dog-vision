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

**Learned classifier** A Random Forest / MLP trained on labeled clips.

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

Trained on ~10k labeled frames from ~50 clips across 3 postures, evaluated on
clips the model never saw during training:

| Metric | Score |
|---|---|
| Grouped 5-fold CV accuracy | **~75%** (±9%) |
| Held-out test accuracy (unseen clips) | **~77%** |
| `sitting` recall | ~0.97 |
| `standing` recall | ~0.80 |
| `lying` recall | ~0.65 |

The main confusion is `lying` ↔ `standing` from elevated camera angles underrepresented in the training set.

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
├── process_video.py     # run SuperAnimal on a video → keypoints (.h5)
├── pose_features.py     # 128-feature vector from raw keypoints
├── posture.py           # Frame/Keypoint model, classifiers, smoothing
├── classify_video.py    # annotate a video with posture labels
├── live_webcam.py       # live chunked webcam overlay (→ browser via MJPEG)
├── build_dataset.py     # labeled clips → dataset.npz
├── train_posture.py     # train RF / MLP, grouped CV, confusion matrix
├── retrain.sh           # end-to-end: process clips → rebuild → retrain
├── posture_model.joblib # trained classifier (committed, ready to run)
└── dataset.npz          # extracted training features
```

## Running it

```bash
git clone https://github.com/xpartla/dog-vision.git
cd dog-vision
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Annotate a recorded clip
python process_video.py samples/dog.mp4     # → keypoints
python classify_video.py samples/dog.mp4    # → posture-labeled video
```

DeepLabCut downloads the SuperAnimal weights (a few hundred MB) on first run.
A CUDA-capable GPU is recommended for live use; CPU is fine for offline
processing. Full setup, GPU notes, and per-tool flags are in
**[HOWTO.md](HOWTO.md)**; training details are in **[TRAINING.md](TRAINING.md)**.

## Roadmap

1. ✅ Pose landmark overlay
2. ✅ Posture classification (sitting / standing / lying) + head tilt
3. ⬜ Gaze / attention proxy via head + snout direction
4. ⬜ Excitement meter: body velocity, head oscillation, tail movement

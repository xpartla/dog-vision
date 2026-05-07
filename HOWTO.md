# How to run

Quick reference for running this project on a fresh machine. For background on
*what* the project is and *why* it's structured this way, see [README.md](README.md).

---

## 1. First-time setup (do once per machine)

```bash
git clone <your-repo-url> dog-vision
cd dog-vision

python -m venv .venv
source .venv/bin/activate          # Windows cmd:  .venv\Scripts\activate
                                   # Windows PS:   .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

**On the GPU laptop:** verify CUDA is being used.

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

If that prints `False` despite an RTX card being present, install a CUDA wheel
explicitly:

```bash
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121
```

The first time you run `process_video.py` or `live_webcam.py`, DeepLabCut will
download SuperAnimal-Quadruped weights (~hundreds of MB) into a cache outside
the repo. Subsequent runs are instant-startup.

---

## 2. Pick a workflow

| You want… | Run this |
|---|---|
| Phase 1 only — keypoints on a recorded clip | [Workflow A](#workflow-a-process-a-recorded-video-phase-1-only) |
| Phase 1 + 2 — keypoints **and** posture/head-tilt labels on a recorded clip | [Workflow B](#workflow-b-process-a-recorded-video-with-posture-classification) |
| Phase 1 only — live webcam overlay | [Workflow C](#workflow-c-live-webcam-phase-1) |
| Capture footage from the webcam, then process offline | [Workflow D](#workflow-d-record-then-process) |

Phase-2 in the live webcam loop is not yet wired in — see [§5](#5-not-yet-implemented).

---

### Workflow A: process a recorded video, phase 1 only

Drop your video into `samples/`, then:

```bash
python process_video.py samples/myclip.mp4
```

Outputs land in `output/`:
- `myclip<scorer>.mp4` — annotated video with keypoints drawn
- `myclip<scorer>.h5` — raw per-frame keypoint predictions

Useful flags:

```bash
python process_video.py samples/myclip.mp4 \
    --model resnet_50 \           # faster but slightly less accurate than hrnet_w32 (default)
    --pcutoff 0.5                 # lower → draws more (noisier) keypoints; default 0.6
```

---

### Workflow B: process a recorded video with posture classification

Two-step: phase 1 first to produce keypoints, then phase 2 for posture/head-tilt.
Step 2 reads the `.h5` from step 1, so you can iterate on the rules in step 2
without paying for inference again.

```bash
# Step 1 — phase 1 (run once per video, or whenever you re-record)
python process_video.py samples/myclip.mp4

# Step 2 — phase 2 (cheap, re-run as you tune thresholds)
python classify_video.py samples/myclip.mp4
```

Output: `output/myclip_posture.mp4` — same frames as the source, with two text
labels overlaid: posture (`sitting` / `standing` / `lying` / `unknown`) and
head tilt (`upright` / `tilt_left` / `tilt_right` / `unknown`).

**First run on a new model version, do this once:**

```bash
python classify_video.py samples/myclip.mp4 --list-keypoints
```

That prints the bodypart names DLC actually wrote. The constants at the top of
`posture.py` need to match — if any name is different, edit the constants and
re-run step 2 (no need to re-run step 1).

**While tuning the rule thresholds:**

```bash
python classify_video.py samples/myclip.mp4 --debug
```

Adds a per-frame readout of every feature value (body H/W, back-knee angle,
hip-height ratio, spine pitch) so you can see why the classifier picked what
it picked. Edit the threshold constants in `posture.py` until the labels
agree with what you see.

Other useful flags:

```bash
python classify_video.py samples/myclip.mp4 \
    --confidence 0.6 \            # min keypoint likelihood to consider a keypoint "visible"
    --smooth-window 15            # sliding-window size for label smoothing; bigger = stickier
```

---

### Workflow C: live webcam, phase 1

```bash
python live_webcam.py
```

Captures ~1.5-second chunks from the camera, runs SuperAnimal on each, plays
back the annotated chunk. Press **q** in the display window to quit.

Useful flags:

```bash
python live_webcam.py \
    --camera 0 \                  # OpenCV camera index; try 1, 2 if 0 is wrong
    --chunk-seconds 1.0 \         # smaller = lower latency, more inference overhead
    --model resnet_50 \           # faster than the default hrnet_w32
    --pcutoff 0.5
```

`--fps` is auto-detected from the camera with a fallback to 30; pass it
explicitly only if the auto-detection picks something off.

> **Windows + WSL2 note:** USB webcams aren't exposed inside WSL2 by default.
> On the run-laptop, either run this script natively on Windows, or set up
> `usbipd-win` to forward the camera into WSL2.

---

### Workflow D: record then process

Useful when you want to capture a clean session, label it later, and iterate
on phase-2 rules off-line.

```bash
# Record 30 seconds from camera 0 to samples/session1.mp4
python webcam_record.py --output samples/session1.mp4 --duration 30

# Then run Workflow A or B against the recording
python process_video.py samples/session1.mp4
python classify_video.py samples/session1.mp4
```

`samples/*.mp4` is gitignored, so recordings stay local.

---

## 3. Where outputs go

```
output/
├── <stem><scorer>.mp4       # phase 1: keypoint overlay (from process_video.py)
├── <stem><scorer>.h5        # phase 1: raw predictions
└── <stem>_posture.mp4       # phase 2: posture + head-tilt overlay
```

`<scorer>` is a model identifier DLC adds to the filename, e.g.
`Video1_superanimal_quadruped_hrnet_w32.mp4`.

---

## 4. Troubleshooting

**`torch.cuda.is_available()` is False on the GPU laptop.**
Install a CUDA-built wheel: `pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121`.

**`classify_video.py` says "No predictions .h5 found".**
You haven't run `process_video.py` against that input yet (or you're pointing
at a different `--predictions-dir`). Either run phase 1 first, or pass
`--predictions path/to/file.h5` explicitly.

**Posture labels are mostly `unknown`.**
Run `python classify_video.py <video> --list-keypoints` and compare against
the constants at the top of `posture.py`. Mismatched names mean every feature
gets `None` and the classifier never has anything to vote on.

**DeepLabCut install fails on `wxpython` / GUI dependencies.**
You don't need the GUI for this project. Try:
`pip install --no-deps deeplabcut && pip install dlclibrary opencv-python numpy pandas tables`
— then re-run.

**Webcam opens but `live_webcam.py` shows a black window.**
Try a different `--camera` index (0, 1, 2). On Windows with multiple cameras
(integrated + capture card), the index isn't always what you'd expect.

**The first DLC run hangs on "downloading model".**
Weights come from HuggingFace; if your network is restricted, set
`HF_HUB_OFFLINE=0` and ensure outbound HTTPS to `huggingface.co` works.

---

## 5. Not yet implemented

- **Phase 2 in the live loop.** `live_webcam.py` only draws keypoints; it
  doesn't run the posture classifier on each chunk. Easy follow-up — the
  classifier reads `.h5` data and a chunk produces one — but not wired in
  yet.
- **True frame-by-frame real-time.** `live_webcam.py` is chunked because
  the high-level DLC inference API loads the model from disk per call. A
  thin wrapper that keeps the model resident is the planned next step.
- **Phases 3 and 4.** Gaze/attention and the excitement meter are not
  scaffolded yet.

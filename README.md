# dog-vision

Real-time pose landmark detection on a dog (Australian Shepherd), built on
SuperAnimal-Quadruped via DeepLabCut.

> **Just want to run it?** See [HOWTO.md](HOWTO.md) for setup and the
> per-workflow commands.

## Roadmap

1. **MVP — pose landmark overlay** (current scope)
2. Pose classification: sitting, standing, lying, head tilt
3. Gaze / attention (likely approximated via head/snout direction)
4. Excitement meter: body velocity, head oscillation, tail movement

## Two machines, two roles

| Machine | Hardware | Role |
|---|---|---|
| Dev | WSL2, CPU only | Write code, process pre-recorded videos |
| Run | Laptop with NVIDIA RTX 2060/3060 | Live webcam inference |

Native USB webcams are not exposed to WSL2 by default. On the run machine,
either run on native Windows (Python + a CUDA build of PyTorch) or set up
`usbipd-win` to forward the camera into WSL2.

## Setup (both machines)

```bash
git clone <your-github-url> dog-vision
cd dog-vision
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

DeepLabCut downloads the SuperAnimal-Quadruped weights (a few hundred MB) on
first run, into a cache outside this repo.

### GPU acceleration on the run machine

PyTorch's pip wheel auto-selects CUDA on most systems. If
`python -c "import torch; print(torch.cuda.is_available())"` prints `False`
despite the GPU being present, install a CUDA wheel explicitly:

```bash
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121
```

## Scripts

### `process_video.py` — offline processing

Process a video file end-to-end and write an annotated copy.

```bash
python process_video.py samples/dog.mp4
# → output/<dog>_<model>.mp4 + .h5 with raw keypoint predictions
```

This is the safe path that works on either machine and uses DLC's documented
public API.

### `webcam_record.py` — capture from webcam

Record raw footage to a file. Useful on the run machine.

```bash
python webcam_record.py --output samples/dog.mp4 --duration 30
```

### `live_webcam.py` — chunked live overlay

Captures short chunks from the webcam, runs SuperAnimal, plays back the
annotated chunk, repeats. Total latency is approximately
`chunk-seconds + inference-time` (≈2s on a 3060). Functional for an MVP demo;
not yet true frame-by-frame.

```bash
python live_webcam.py
# Tweaks:
python live_webcam.py --chunk-seconds 1.0 --model hrnet_w32
```

### `classify_video.py` — phase 2: posture and head-tilt labels

Reads the `.h5` predictions written by `process_video.py` and produces a new
annotated video tagged with posture (sitting / standing / lying / unknown) and
head tilt (upright / tilt_left / tilt_right / unknown). Rule-based geometry
over keypoint angles, with sliding-window majority voting to suppress flicker.

```bash
# Auto-discovers output/<stem>*.h5
python classify_video.py samples/Video1.mp4

# Inspect the bodypart names DLC actually wrote (verify they match
# the constants at the top of posture.py)
python classify_video.py samples/Video1.mp4 --list-keypoints

# Show per-feature numeric values on each frame for tuning
python classify_video.py samples/Video1.mp4 --debug
```

Designed for the live-demo setup: tripod at ~ribcage height, slight downward
tilt, dog free to face any direction. Features used (body H/W, back-knee
angle, hip-above-paws, spine pitch, eye-line vs. head axis) are all relative
geometry, not absolute pixel positions, so they survive dog rotation. When
keypoints are occluded by Aussie fur, the missing features are simply skipped
and the classifier votes with what it has — falling back to `unknown` rather
than guessing if too few features survive.

## Path to true real-time

`live_webcam.py` is chunked because the high-level
`deeplabcut.video_inference_superanimal` API loads the model from disk on each
call. To reach 30 fps frame-by-frame, the next iteration needs to keep the
PyTorch model resident in memory and skip file I/O — i.e., a thin wrapper
around DLC's lower-level pose-estimation classes. That refactor is deferred
until the chunked pipeline is verified end-to-end.

## Phases 2–4 keypoint coverage

SuperAnimal-Quadruped predicts ~39 keypoints including separate points for
both ears, snout, eyes, neck, spine, hips, four limbs, and three tail
landmarks (base / mid / tip). That set covers everything the later phases
need:

- Phase 2 (head tilt / sitting): ear / spine / hip angles
- Phase 3 (attention proxy): eyes + snout direction
- Phase 4 (excitement): tail-tip oscillation, head oscillation, body
  centroid velocity

## Layout

```
dog-vision/
├── .gitignore
├── .gitattributes
├── README.md
├── requirements.txt
├── process_video.py
├── webcam_record.py
├── live_webcam.py
└── samples/        # local-only test footage (gitignored)
```

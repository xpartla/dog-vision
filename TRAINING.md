# Training the learned posture classifier

The rule-based classifier (`classify_posture`) assumes a side-on camera at the
dog's height, because it reads vertical *image* position as height. That breaks
on other viewpoints (e.g. an elevated front-on camera, where the dog's rear is
farther away and projects to the *top* of the frame — read as "standing" even
when lying).

The learned classifier removes that assumption. It trains a model on
**viewpoint-robust features** (joint angles + scale/translation-normalized
keypoint geometry + per-keypoint visibility, see `pose_features.py`) and learns
the posture cues from your own footage, across whatever angles you film.

## The 4 steps

### 1. Collect labeled clips

Film the dog and **record each clip with a single posture held throughout** —
then the whole clip carries one label, which is the cheapest accurate labeling.

The model is only as view-robust as your data, so **vary deliberately**:
- camera **angle** (side, front, rear, 3/4), **height** (floor, ribcage, above),
  and **distance**
- the dog's **orientation** (facing left/right/toward/away) and which side faces
  the camera
- lighting / rooms / the dog moving within the posture (shifting, head turns)

Aim for **many clips per posture** (≥10, the more varied the better) rather than
a few long ones. A few hundred frames per clip is plenty.

### 2. Run pose estimation on each clip

On the GPU laptop, produce a `.h5` of keypoints per clip:

```bash
python process_video.py path/to/clip.mp4 --output-dir output
```

Then arrange the `.h5` files into one folder per label:

```
data/
  standing/   *.h5
  sitting/    *.h5
  lying/      *.h5
```

(Folder names become the class labels — use whatever set of postures you want.)

### 3. Build the dataset

```bash
python build_dataset.py data/ --out dataset.npz --stride 2 --augment-flip
```

- `--stride 2` keeps every 2nd frame (consecutive frames are nearly identical).
- `--augment-flip` adds a horizontally-mirrored copy of each sample so the model
  doesn't care which way the dog faces. Recommended.

### 4. Train and evaluate

```bash
python train_posture.py dataset.npz --out posture_model.joblib
```

This splits **by clip** (never by frame), so the reported accuracy reflects
generalization to unseen clips/views. Read the output:

- **Grouped CV accuracy** — the headline generalization number.
- **Confusion matrix** — *which* postures get confused (e.g. sit↔stand).
- **Top features** — sanity check on what the model relies on.

If a posture is weak or confused, the fix is almost always **more varied clips
of that posture** (and of the one it's confused with). Try `--model mlp` to
compare against the random forest.

## Use the trained model

Both the offline and live tools take `--posture-model`; it replaces the
geometric rules with the learned classifier (and matches the keypoint
confidence threshold used during training):

```bash
python classify_video.py samples/clip.mp4 --posture-model posture_model.joblib
python live_webcam.py --posture-model posture_model.joblib
```

Without `--posture-model`, both tools fall back to the rule-based classifier.

## Notes

- `pose_features.py` defines the feature vector and is shared by training and
  inference, so they can't drift. If you change it, **retrain** — the loader
  raises a clear error if a saved model's feature count no longer matches.
- The model is per-frame. Frame-to-frame jitter is smoothed downstream by the
  existing 1-Euro keypoint filter and the majority-vote `LabelSmoother`.

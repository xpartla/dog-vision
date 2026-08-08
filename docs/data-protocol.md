# Data protocol v2 — continuous movement sessions

Recording protocol for the movement-dynamics corpus (roadmap phase 3). It
replaces protocol v1, which produced single-posture clips for the posture
classifier. Those 48 clips remain in use as the posture probe benchmark; they
are deliberately static and contain almost none of what a dynamics model must
learn — **transitions and locomotion**. This protocol collects exactly that.

## The unit: one session

A **session** is one continuous, uncut take:

- **2–5 minutes** long (~3 min is the sweet spot: long enough for the full
  checklist, short enough that the dog stays engaged).
- **One file, camera untouched from start to end.** Do not stop-start the
  recording — temporal continuity is the point. If something interrupts,
  either let it roll or discard and start a new session.
- One session becomes one `sequences/<session>.npz` and one **group** in the
  train/val/test split. Frames from a session never straddle a split, so more
  short-ish sessions beat fewer long ones.

**Corpus target:** ~20 sessions (60–90 min total). The WP2 acceptance bar is
10. Expect 2–3 sessions per filming day before motivation drops; the corpus
compounds — every future session adds value.

## Per-session behavior checklist

Each session should contain, in any order, mixed naturally:

- [ ] **Walking** — crossing the frame in both directions
- [ ] **Trotting** — at least a few passes
- [ ] **Toward / away** — at least one pass each, straight at and away from
      the camera
- [ ] **Sit ↔ stand** transitions — a few reps
- [ ] **Stand ↔ lie** transitions — a few reps
- [ ] **Sit ↔ lie** transitions — a few reps
- [ ] **Play bow** — treats or a toy make this easy
- [ ] **Turning / spins** — direction changes, cued spins
- [ ] Some idle standing/lying is fine and natural — it just must not
      dominate the session

Cued reps are perfect. The movement *between* postures is the training
signal, not the postures themselves.

A loose 3-minute script that covers everything: ~30 s free movement and
walking passes → cued transition block (sit–stand–down, 3–4 reps) → play
block (bow, spins, short tosses) → more locomotion passes both directions →
settle into a lie.

## Camera rules

- **Fixed tripod for the whole session.** A moving camera injects camera
  motion into every keypoint trajectory and the model would learn your hand,
  not the dog. All viewpoint diversity comes from varying the setup
  **between** sessions.
- **Vary across sessions** — pick one height bucket and one dominant angle
  per session, and spread the corpus over the grid:

  | Height bucket | Lens height                        |
  |---------------|------------------------------------|
  | `floor`       | ≤ ~30 cm                           |
  | `mid`         | ~50–80 cm (≈ dog ribcage, tripod)  |
  | `high`        | ≥ ~1.2 m, tilted down              |

  | Angle bucket | Camera vs. the dog's main plane of movement |
  |--------------|---------------------------------------------|
  | `side`       | roughly perpendicular (profile passes)      |
  | `quarter`    | ~45°                                        |
  | `front`      | roughly head-on                             |

  The dog moves freely, so the angle is the *dominant* one, not a guarantee.
  Vary the room/location across sessions too.
- **Frame rate:** ≥ 30 fps required; 60 fps welcome if the camera does it
  easily (per-session fps is stored in the manifest; mixed rates are fine and
  useful for the irregular-sampling experiments later).
- **Resolution:** 1080p landscape is plenty. Lock exposure/focus if the
  camera allows — autofocus hunting blurs keypoints.
- **Framing:** whole dog in frame with several body lengths of room to move;
  dog roughly ¼–½ of frame height. Brief exits from frame are tolerable
  (visibility masks absorb them) but are wasted footage.
- **Light:** bright is better. Dim rooms mean motion blur on moving legs,
  which the pose estimator turns into keypoint jitter.
- Favor setups where the **tail** is visible (side/quarter views) — the
  evaluation breaks out error by keypoint group, tail included.
- Keep the handler out of frame where possible (lure from behind the camera
  or the frame edge). Avoid mirrors and glass doors — a reflected dog is a
  second detection.

## Metadata — record it immediately after each take

One line per session, written right after filming (it cannot be
reconstructed later). Filename carries the essentials:

```
data/sessions/<date>_s<nn>_<height>_<angle>_<location>.mp4
data/sessions/2026-08-09_s01_mid_side_livingroom.mp4
```

plus a running `data/sessions/sessions.csv`:

| column     | example                | notes                          |
|------------|------------------------|--------------------------------|
| session_id | `2026-08-09_s01`       | matches filename               |
| date       | `2026-08-09`           |                                |
| location   | `livingroom`           |                                |
| height     | `mid`                  | `floor` / `mid` / `high`       |
| angle      | `side`                 | `side` / `quarter` / `front`   |
| fps        | `30`                   | as recorded                    |
| notes      | `no play bow, distracted by doorbell` | anything unusual |

`build_sequences.py` (WP2 item 2) folds this into `sequences/manifest.json`.

## Common mistakes

- Chasing the dog with the camera. Tripod, always.
- Every session in the same room at the same height — across-session variety
  is what made the posture classifier viewpoint-robust; the same applies here.
- Staging only held postures. That corpus already exists; this one is about
  movement.
- Cutting the recording between behaviors. One session = one uncut take.
- Filming at dusk indoors and donating the footage to motion blur.

## After filming

Copy files to `data/sessions/` (the `data/` tree is gitignored; raw footage
never enters git). Pose estimation runs on the GPU machine via
`build_sequences.py`, producing one `sequences/<session>.npz` per session
plus `sequences/manifest.json` — see PLAN.md WP2 for the format contract.
Budget several GB of disk for ~20 sessions of 1080p footage.

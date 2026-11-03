# Point answers: coordinates from a one-pass decision model

smooth-decider adds a fourth question type to decider, `Point`, next to `Choice`,
`Score` and `Noul`. A Point question asks for a location in a declared coordinate
frame ("move Alpha halfway to Bravo", "where should you tap?") and is answered in
the same single forward pass, from the same answer-slot hidden state decider
already uses, by a small head that outputs a **mixture of Gaussians** over (x, y).

Why a mixture: when there are several good answers (either flank, two open
teammates) a single predicted point would land between them on a bad spot; the
mixture keeps both, and its spread is a calibrated statement of confidence.

## Design

| piece | where | what it does |
|---|---|---|
| frame | `decider/point.py` `Frame` | x/y extent, units and y direction; written into the question; the head works in the frame normalized to 0-1, so pixels, metres and tiles are one problem |
| prompt | `build_point` | decider's plain state-first layout; the answer slot is the last token of `Answer as a point: (` |
| head | `PointHead` | LayerNorm -> Linear(hidden, 1024) -> SiLU -> Linear(1024, 8 x 6): weight, mean (x, y), sigma (x, y), correlation per component |
| loss | `point_nll` | negative log-likelihood of the true point, summed with decider's own Choice loss in mixed batches |
| confidence | `hpd_level` | smallest highest-density region containing the truth; uniform on [0, 1] when calibrated (tested) |
| model | `decider/point_model.py` `PointModel` | decider's `DecisionModel` + head; one backbone pass answers Choice (decider's letter readout, unchanged) and Point; LoRA, adapter or merged save/load, weight averaging |
| API | `decider/point_infer.py` `PointDecider` | `point(state, questions)` -> best point, mixture components, 90% radius, in the question's units |

Choice, Score and Noul behave exactly as in decider: a test checks that Choice
logits through `PointModel` equal decider's own readout, and a fresh LoRA
changes nothing until trained.

## Usage

```python
from decider.point_infer import PointDecider
d = PointDecider("path/to/point/model")
d.point("Map: x from 0 to 512, y from 0 to 512 (pixels; y increases downward).\n"
        "- Alpha (ally) at (120, 340)\n- Bravo (enemy) at (400, 90)",
        [{"question": "Move Alpha halfway to Bravo.",
          "frame": {"x0": 0, "x1": 512, "y0": 0, "y1": 512, "units": "pixels", "y_down": True}}])
```

Install the extra: `pip install -e ".[train,point]"`.

## Training data (all non-game, rendered as text scenes)

| source | questions | frame | terms |
|---|---|---|---|
| synthetic geometry (`decider/data/synth_geometry.py`) | 8 families with exact answers: fraction along a line, step toward, centroid, nearest walkable point, nearest point out of enemy range, formation slot, first waypoint around an obstacle, either-side flank (two right answers) | normalized, metres, pixels, tiles | ours (Apache 2.0) |
| StatsBomb 360 open data | pass end: instructed ("pass to teammate T4") and open ("plays a ground pass") | 120 x 80 yards | StatsBomb public data user agreement |
| SkillCorner open data | where an off-ball run ends, by run type | metres from the centre spot | open with attribution requested |
| AndroidControl | step instruction + accessibility element list -> tap point | screen pixels | Apache 2.0 |
| Metrica sample data | held out entirely (transfer test) | normalized pitch | Metrica sample-data terms |
| decider's own mixture (replay) | Choice / Score / Noul | - | as decider |

Converters live in `decider/data/point_sources/` and stream downloads; build with
`python -m decider.data.point_build --out data/point`. Two data notes worth knowing:

- **SkillCorner's event coordinates are normalized to the attacking direction,
  but its tracking is not.** Events of a team attacking right to left must be
  rotated 180 degrees; without it half the run targets land on the wrong half of
  the pitch (39% of runners were at their run's start; 99% after rotating).
- **AndroidControl episodes can have no actions field at all** (protobuf omits
  empty lists); the converter skips them. Its accessibility trees are decoded by
  a small built-in protobuf reader, so neither TensorFlow nor android_env is needed.

## Training recipe

decider-4b v2.1, LoRA rank 64 / alpha 128 on every attention, GatedDeltaNet and
MLP projection (decider's own stage recipe for this model; a full fine-tune of
4B does not fit a 40 GB GPU), LR 1e-4 for the LoRA and 1e-3 for the new head.

1. **Point run** (`decider/point_train.py`): 35M tokens of Point rows + 25M of
   decider replay, 2,088 steps, about 6.6 h on one A100 40 GB (2,850 tokens/s).
2. **Continuation** from that adapter: the hard geometry families were
   imprecise rather than wrong (`decider/point_diag.py` found only 7% of errors
   came from choosing the wrong corner, side or range), so 33k more rows of them.
3. **Rebalanced continuation**: the original mix at natural shares plus some
   hard geometry, which recovers the real-data tasks the continuation had
   under-replayed. This is the released recipe's final stage on `main`.
4. **Packed continuation** (branch `bw-smooth-decider`): crowded scenes with up to 40 questions
   packed in one row (`decider/data/crowded.py`); packed answers then equal independent ones.
   Results in [BW.md](BW.md).

## Results (held-out rows, final checkpoint vs base decider-4b)

| task | median error (share of frame) | NLL, base untrained head -> trained | 90% region holds the truth |
|---|---|---|---|
| synthetic centroid / formation / fraction | 1.2-1.5% | +0.3..+0.4 -> -5.9..-6.1 | 93-97% |
| synthetic waypoint around obstacle | 0.8-0.9% | +0.33 -> -6.5 | 95% |
| synthetic out of range / toward / walkable | 1.6-2.6% | +0.39..+0.42 -> -4.5..-5.3 | 85-91% |
| Android tap (1080 x 2400 px) | 2.1% (about 35 px) | +0.54 -> -4.13 | 88% |
| StatsBomb pass to a named teammate | 2.2% (about 2 yards) | +0.50 -> -5.26 | 88% |
| StatsBomb pass, receiver not given | 13.6% | +0.51 -> -1.57 | 90% |
| SkillCorner run end | 7.1% | +0.49 -> -2.49 | 87% |
| Metrica pass / move (never trained on) | 3.4% / 4.7% | +0.47 -> -2.9 | **53-55%** |

decider's own regression eval (Choice/Score/Noul): in-task accuracy 0.827 ->
0.824, held-out 0.790 -> 0.800; Choice calibration error (ECE) 0.108 -> 0.076.

## Known limitations

- **Calibration does not transfer to a new provider.** On Metrica, never seen in
  training, answers are accurate but overconfident: the 90% region holds the
  truth only about 55% of the time. Treat the spread as a lower bound in a new domain.
- **Latency.** Eager PyTorch on an A100: 70 ms for one question, 700 ms for 32
  short rows. decider's CUDA-graph engine is 5-6x faster for Choice but does not
  yet return slot hidden states for the Point head; long states dominate cost.
- **Weights are not released.** They would carry the training data's terms
  (above); the recipe and code are here to reproduce them.

## Credits

Built on [Mapika/decider](https://github.com/Mapika/decider) (Apache 2.0) by Mark
Marosi. smooth-decider adds the files listed above; decider's own files are
unchanged except this documentation, the README's fork section, packaging, and
`.gitignore`.

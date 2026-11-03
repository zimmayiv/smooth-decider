# Brood War serving: latency, sweep layouts, packed Point answers

This branch (`bw-smooth-decider`) adds the tools and the training stage used to serve smooth-decider as the
fast "system one" of a Brood War agent: every few game frames, one model call (a *sweep*) answers the
questions for the units being controlled. It answers two questions: **how should a sweep be laid out to
fit the time budget**, and **does packing many questions into one row hurt the answers**.

All numbers: one A100 40 GB, eager PyTorch, bf16, decider-4b + the Point LoRA. Brood War sweeps are
synthetic (`decider/data/bw_bench.py`): sizes are assumptions from general Brood War knowledge, not a
designed observation format, and their answers are meaningless; they measure cost only.

## Budget

A decision every N frames at "fastest" speed (42 ms per frame) leaves N x 42 ms, minus the network
round trip (72 ms measured from a laptop to the GPU): **180 / 264 / 348 / 432 ms at 6 / 8 / 10 / 12
frames**.

## Sweep layouts

| layout | what is in the row | mid game tokens | late game tokens |
|---|---|---|---|
| `make_sweep` (every unit) | every unit and structure listed and asked about (~70 tokens per question) | 22k | 39-41k |
| `make_grouped_sweep` | army in groups of `group_size`; workers summarized; only the highest-priority groups asked (`q_budget`); structures listed and asked about only on macro sweeps; command list and map frame stated once (`compact`) | 2.3-7k | 3.3-11k |
| ... with `fine_squads` | the 1-3 most important squads (here: closest to an enemy; in play the planner decides) split finely, `fine_size` 1 or 2 | 3.2-6k | 4.4-7.5k |

Ways of running a sweep (`decider/point_latency.py`):
**a** one row per question (state repeated), **b** packed (one row: state, then every question),
**c** shared prefix (state once, cache copied per question), **e** as c with the static part cached
across sweeps. b was fastest in every grouped configuration; a is minutes per sweep; e saves nothing
(the static part is small). Throughput is ~16k tokens/s plus a ~70-90 ms fixed overhead.

Packed, ms, micro-only / macro sweep, 20 micro questions:

| army grouping | early | mid | late |
|---|---|---|---|
| every unit on its own | 97 / 113 | 365 / 431 | 599 / 691 |
| groups of 4 | 89 / 102 | 220 / 276 | 303 / 390 |
| groups of 8 | 89 / 101 | 175 / 228 | 264 / 343 |
| groups of 8, 2 squads per-unit | - | 243 / 300 | 304 / 390 |
| groups of 8, 3 squads per-unit, 40 questions | - | 330 / 381 | 391 / 471 |

**At 12 frames everything fits except fully per-unit late-game play** and the late macro sweep with 2-3
per-unit squads and 40 questions. The ~90 ms overhead floor is what CUDA graphs and compile (decider's
engine) remove; they are not wired to the Point head yet.

```bash
python -m decider.point_latency --model path/to/point/model --layout grouped --phases mid,late \
  --group_sizes 8 --fine_squads 0,1,2,3 --fine_sizes 1,2 --q_budgets 20,40 --macro 0,1 --out lat.json
python -m decider.point_latency --tokens_only --tokenizer path/to/qwen3.5 --layout grouped ...   # no GPU
python -m decider.point_latency --model ... --check   # shared-prefix answers == independent rows
```

## Packing and the packed-Point stage

`decider/point_packing.py` compares packed rows with independent rows on labeled multi-question scenes,
decider's held-out multi-question examples, crowded rows (below) and unlabeled sweeps. With the
released recipe's checkpoint, packing cost nothing for Choice but made Point answers 12-20% worse,
growing with depth in the pack (questions 16-40: 5.8% vs 4.6% error), and crowded scenes were hard
either way (5% error vs 1-2% on the single-question geometry tasks). Every Point training row had
had one question and at most ~10 entities.

`decider/data/crowded.py` generates crowded scenes: 2-40 own units, 4-16 enemies and 3-8 markers, one
question per unit in one row, six exact-answer kinds (fraction between markers, step toward an enemy,
centre of named enemies, N/S/E/W of a marker, nearest enemy, farthest marker) in shuffled order, plain
or Brood War-style names, half with the frame stated once (`Frame M`). Stage 4 of the recipe continues
from stage 3 on 5M tokens of the original Point mix + 4M of crowded rows + 2.5M of decider replay
(LR 5e-5, head 3e-4, 408 steps, 75 min):

```bash
python -m decider.data.crowded --out data/point_packed --train 6000 --val 300 --test 300
python -m decider.point_train --model path/to/decider-4b --init_adapter stage3/model \
  --point_data data/point --point_tokens 5e6 --point_data2 data/point_packed --point_tokens2 4e6 \
  --replay mixture.pkl --replay_tokens 2.5e6 --mode point --lr 5e-5 --head_lr 3e-4 --warmup 20 --seed 2 --out stage4
```

| | stage 3 | **stage 4** |
|---|---|---|
| crowded held-out, Point error, packed / independent | 5.6% / 5.0% | **1.5% / 1.5%** |
| crowded held-out, Choice, packed / independent | 82.8% / 85.1% | **97.9% / 97.9%** |
| crowded, questions 16-40, Point packed | 5.8% | **1.5%** |
| decider regression eval, in-task / held-out | 0.824 / 0.800 | 0.825 / 0.802 |
| Android tap | 2.05% | **1.85%** |
| fraction / toward / formation | 1.45 / 2.0 / 1.16% | **1.04 / 1.65 / 1.11%** |
| waypoint / out of range / flank | **0.87 / 2.6 / 10.3%** | 0.98 / 2.68 / 11.1% |
| Metrica move / pass (never trained on) | 4.68 / 3.42% | 4.36 / 3.34% |

The packing penalty is gone. Small regressions on waypoint, flank, SkillCorner (6.1 -> 6.5 m) and open
passes; 90% coverage fell a few points on some tasks (Android 0.88 -> 0.85, walkable 0.85 -> 0.79).

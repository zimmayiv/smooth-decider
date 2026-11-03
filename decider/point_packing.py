"""Does packing many questions into one row change the answers? (latency test 3)

Packed rows (all of a sweep's questions after one state, the fastest mode) let later questions
attend to earlier ones; independent rows (state + one question) do not. Three comparisons:

  scenes   labeled: one synthetic scene with many units and n questions about it (move U3 halfway
           between two markers, step toward an enemy, which enemy is nearest), exact answers;
           accuracy packed vs independent, by n and by the question's position in the pack
  decider  labeled: decider's own held-out examples with several questions, packed (as decider
           trains) vs one question per row
  sweeps   unlabeled: Brood War-shaped grouped sweeps; how often packed and independent agree, and
           how often packed agrees with itself when the question order is reversed

  python -m decider.point_packing --model path/to/point/model --replay data/mixture_delta.pkl --out results/packing.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch

from decider.data.bw_bench import make_grouped_sweep
from decider.point_model import PointModel
from decider.point import ChoiceQ, PointExample, PointQ, build_point, collate, mixture_logpdf
from decider.data.synth_geometry import random_frame, snap, rand_point, fmt

from decider.prompt import NARROW


# ---- labeled multi-question scenes

def multi_scene(rng: random.Random, n_q: int, n_enemies: int = 8, n_markers: int = 6) -> PointExample:
    """One scene, n_q allies, one question per ally (cycling through three kinds), exact answers."""
    f = random_frame(rng)
    P = lambda: snap(rand_point(rng, f), f)
    allies = [(f"U{i + 1}", P()) for i in range(n_q)]
    enemies = [(f"E{i + 1}", P()) for i in range(n_enemies)]
    markers = [(f"M{i + 1}", P()) for i in range(n_markers)]
    ents = [(n, "ally", p) for n, p in allies] + [(n, "enemy", p) for n, p in enemies] + [(n, "marker", p) for n, p in markers]
    rng.shuffle(ents)
    ctx = f"Map: {f.describe()}.\n" + "\n".join(f"- {n} ({k}) at ({fmt(p[0], f)}, {fmt(p[1], f)})" for n, k, p in ents)
    qs = []
    for i, (u, pu) in enumerate(allies):
        kind = ("fraction", "toward", "nearest")[i % 3]
        if kind == "fraction":
            (a, pa), (b, pb) = rng.sample(markers, 2)
            frac, words = rng.choice([(0.5, "halfway"), (1 / 3, "one third of the way"), (2 / 3, "two thirds of the way")])
            qs.append(PointQ(f"Move {u} to the point {words} from {a} to {b}.", f,
                             (pa[0] + frac * (pb[0] - pa[0]), pa[1] + frac * (pb[1] - pa[1]))))
        elif kind == "toward":
            e, pe = rng.choice(enemies)
            d = math.dist(pu, pe) or 1.0
            step = float(fmt(rng.uniform(0.2, 0.9) * d, f)) or d / 2
            s = min(step, d) / d
            qs.append(PointQ(f"Move {u} {fmt(step, f)} {f.units} straight toward {e}.", f,
                             (pu[0] + s * (pe[0] - pu[0]), pu[1] + s * (pe[1] - pu[1]))))
        else:
            # the nearest enemy plus up to 4 that are clearly farther, so the answer is unambiguous
            # (rejection sampling here could loop forever when all enemies sit at similar distances)
            by = sorted(enemies, key=lambda e: math.dist(pu, e[1]))
            d0, margin = math.dist(pu, by[0][1]), 0.05 * (f.x1 - f.x0)
            far = [e for e in by[1:] if math.dist(pu, e[1]) - d0 >= margin] or by[1:]
            opts = [by[0]] + rng.sample(far, min(4, len(far)))
            rng.shuffle(opts)
            gold = min(range(len(opts)), key=lambda j: math.dist(pu, opts[j][1]))
            qs.append(ChoiceQ(f"Which enemy is nearest to {u}?", [n for n, _ in opts], gold))
    return PointExample(ctx, qs, "multi_scene")


# ---- running a model on packed vs independent rows

def mode_point(p: dict) -> torch.Tensor:
    """The component mean with the highest mixture density: a deterministic best point (no sampling, so
    packed and independent answers can be compared without sampling noise)."""
    N, K, _ = p["mu"].shape
    flat = {k: v.repeat_interleave(K, 0) for k, v in p.items()}
    dens = mixture_logpdf(flat, p["mu"].reshape(-1, 2)).view(N, K)
    return p["mu"][torch.arange(N), dens.argmax(1)]


@torch.no_grad()
def answers(model, ex: PointExample, mode: str, bs: int = 8, max_ctx: int = 8192) -> list:
    """Per question, in ex.qs order: ("choice", probs) or ("point", (u, v) best point in normalized frame)."""
    dev = next(model.parameters()).device
    if mode == "packed":
        rows = [build_point(ex, model.tok, max_ctx)]
    else:
        rows = [build_point(PointExample(ex.context, [q], ex.task), model.tok, max_ctx) for q in ex.qs]
    outs = []
    for c in range(0, len(rows), bs):
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in collate(rows[c:c + bs], model.tok.pad_token_id).items()}
        o = model(b)
        ch = torch.softmax(o["choice_logits"].float(), -1).cpu() if "choice_logits" in o else None
        pt = mode_point(o["point"]).float().cpu() if "point" in o else None
        outs.append((ch, pt))
    # reassemble in question order: within each batch, choice rows then point rows, both in row/slot order
    ci = pi = 0
    flat_c = torch.cat([c for c, _ in outs if c is not None]) if any(c is not None for c, _ in outs) else None
    flat_p = torch.cat([p for _, p in outs if p is not None]) if any(p is not None for _, p in outs) else None
    res = []
    for q in ex.qs:
        if isinstance(q, ChoiceQ):
            res.append(("choice", flat_c[ci][: len(q.options)])); ci += 1
        else:
            res.append(("point", tuple(flat_p[pi].tolist()))); pi += 1
    return res


def score(ex: PointExample, ans: list) -> list:
    """Per question: (kind, correct or normalized error)."""
    out = []
    for q, (kind, a) in zip(ex.qs, ans):
        if kind == "choice":
            out.append(("choice", float(int(a.argmax()) == q.gold)))
        else:
            u, v = q.frame.normalize(*q.target)
            out.append(("point", math.dist(a, (u, v))))
    return out


def pos_bucket(i: int) -> str:
    return "1-5" if i < 5 else "6-15" if i < 15 else "16-40" if i < 40 else "41+"


def run_scenes(model, ns=(5, 20, 40), per_n=40, seed=0) -> dict:
    rng = random.Random(seed)
    res = {}
    for n in ns:
        acc = defaultdict(lambda: defaultdict(list))  # (mode, kind, bucket) -> values
        for _ in range(per_n):
            ex = multi_scene(rng, n)
            for mode in ("packed", "independent"):
                for i, (kind, v) in enumerate(score(ex, answers(model, ex, mode))):
                    acc[mode][(kind, "all")].append(v)
                    acc[mode][(kind, pos_bucket(i))].append(v)
        res[n] = {mode: {f"{k}/{b}": summarize(k, vs) for (k, b), vs in d.items()} for mode, d in acc.items()}
        print("[scenes]", n, json.dumps(res[n]), flush=True)
    return res


def summarize(kind, vs):
    vs = sorted(vs)
    if kind == "choice":
        return {"acc": sum(vs) / len(vs), "n": len(vs)}
    return {"err_median": vs[len(vs) // 2], "err_mean": sum(vs) / len(vs), "within_2pct": sum(v < 0.02 for v in vs) / len(vs), "n": len(vs)}


# ---- held-out crowded rows (decider.data.crowded test split)

def run_crowded(model, path: str, limit=100) -> dict:
    from decider.data.point_build import read_jsonl
    exs = read_jsonl(path)[:limit]
    acc = defaultdict(lambda: defaultdict(list))
    for ex in exs:
        for mode in ("packed", "independent"):
            for i, (kind, v) in enumerate(score(ex, answers(model, ex, mode))):
                acc[mode][(kind, "all")].append(v)
                acc[mode][(kind, pos_bucket(i))].append(v)
    res = {mode: {f"{k}/{b}": summarize(k, vs) for (k, b), vs in d.items()} for mode, d in acc.items()}
    print("[crowded]", json.dumps(res), flush=True)
    return res


# ---- decider's own held-out multi-question examples

def from_decider(e, rng) -> PointExample | None:
    qs = []
    for q in e.qs:
        if q.gold is None or q.gold < 0 or not 2 <= len(q.options) <= NARROW:
            return None
        perm = list(range(len(q.options)))
        rng.shuffle(perm)
        qs.append(ChoiceQ(q.text, [q.options[j] for j in perm], perm.index(q.gold)))
    return PointExample(e.context, qs, e.task) if len(qs) >= 2 else None


def run_decider(model, replay: str, limit=400, seed=0) -> dict:
    from decider import data as D
    _, evals = D.load_cache(replay)
    rng = random.Random(seed)
    exs = []
    for task, xs in evals.items():
        for e in xs:
            x = from_decider(e, rng)
            if x is not None:
                exs.append(x)
    rng.shuffle(exs)
    exs = exs[:limit]
    acc = {m: defaultdict(list) for m in ("packed", "independent")}
    agree = []
    for ex in exs:
        a = {m: answers(model, ex, m) for m in acc}
        for m in acc:
            for i, (_, v) in enumerate(score(ex, a[m])):
                acc[m]["all"].append(v); acc[m][pos_bucket(i)].append(v)
        agree += [int(p.argmax()) == int(q.argmax()) for (_, p), (_, q) in zip(a["packed"], a["independent"])]
    nq = [len(x.qs) for x in exs]
    res = {"examples": len(exs), "questions_per_example": {"mean": sum(nq) / max(1, len(nq)), "max": max(nq, default=0)},
           "acc": {m: {b: {"acc": sum(v) / len(v), "n": len(v)} for b, v in d.items()} for m, d in acc.items()},
           "argmax_agreement": sum(agree) / max(1, len(agree))}
    print("[decider]", json.dumps(res), flush=True)
    return res


# ---- unlabeled Brood War-shaped sweeps

def run_sweeps(model, configs, seeds=(0, 1, 2)) -> dict:
    out = []
    for cfg in configs:
        agg = defaultdict(list)
        for s in seeds:
            sw = make_grouped_sweep(seed=s, **cfg)
            ex = PointExample(sw.context, sw.questions, "bw")
            rev = PointExample(sw.context, sw.questions[::-1], "bw")
            pk, ind, rv = answers(model, ex, "packed"), answers(model, ex, "independent"), answers(model, rev, "packed")[::-1]
            for other, name in ((ind, "vs_independent"), (rv, "vs_reversed")):
                for (k, a), (_, b) in zip(pk, other):
                    if k == "choice":
                        agg[f"{name}/choice_agree"].append(float(int(a.argmax()) == int(b.argmax())))
                        agg[f"{name}/choice_tv"].append(0.5 * float((a - b).abs().sum()))
                    else:
                        agg[f"{name}/point_dist"].append(math.dist(a, b))
        row = {**cfg, **{k: sum(v) / len(v) for k, v in agg.items()},
               **{k.replace("point_dist", "point_dist_median"): sorted(v)[len(v) // 2] for k, v in agg.items() if "point_dist" in k}}
        out.append(row)
        print("[sweeps]", json.dumps(row), flush=True)
    return out


SWEEP_CONFIGS = [
    dict(phase="mid", group_size=8, q_budget=20),
    dict(phase="mid", group_size=8, q_budget=40, fine_squads=2, fine_size=1),
    dict(phase="late", group_size=4, q_budget=40, macro=True),
    dict(phase="late", group_size=8, q_budget=40, fine_squads=3, fine_size=1),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--replay", default=None)
    ap.add_argument("--scene_ns", default="5,20,40")
    ap.add_argument("--per_n", type=int, default=40)
    ap.add_argument("--decider_limit", type=int, default=400)
    ap.add_argument("--parts", default="sweeps,scenes,decider,crowded")
    ap.add_argument("--sweep_seeds", type=int, default=2)
    ap.add_argument("--crowded", default=None, help="crowded test split (jsonl.gz) for the 'crowded' part")
    ap.add_argument("--crowded_limit", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model = PointModel(a.model, dtype=getattr(torch, a.dtype), grad_ckpt=False).to(a.device).eval()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(out.read_text()) if out.exists() else {"model": a.model}  # resume: finished parts are kept
    jobs = {"scenes": lambda: run_scenes(model, [int(x) for x in a.scene_ns.split(",")], a.per_n),
            "decider": lambda: run_decider(model, a.replay, a.decider_limit),
            "sweeps": lambda: run_sweeps(model, SWEEP_CONFIGS, seeds=tuple(range(a.sweep_seeds))),
            "crowded": lambda: run_crowded(model, a.crowded, a.crowded_limit)}
    for part in a.parts.split(","):
        if part in res or (part == "decider" and not a.replay) or (part == "crowded" and not a.crowded):
            continue
        res[part] = jobs[part]()
        out.write_text(json.dumps(res, indent=1))  # after every part, so a cut-off run keeps what finished
    print("[done]", a.out, flush=True)


if __name__ == "__main__":
    main()

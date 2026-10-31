"""Evaluate a coordinate model on held-out Point data, decider's own Choice eval, and latency.

  python -m decider.point_evaluate --model results/point_run/model --point_data data/point \
      --splits val,test --replay data/mixture_delta.pkl --out results/point_run/eval.json

Point metrics per task (task = source family):
  err_median / err_p90  distance from the best guess to the true point, in the frame's own units
  err_norm_median       the same in normalized frame units (fraction of the frame's width/height)
  nll                   -log density of the true point in normalized space (comparable across models)
  cov50 / cov90         share of true points inside the model's 50% / 90% highest-density regions
  calib_err             mean |coverage(q) - q| over q = 0.1 ... 0.9 (0 = perfectly calibrated)
  off_frame             share of best guesses outside the declared frame
Reference rows: "centre" guesses the frame centre with a uniform density (NLL 0, the floor any model must beat).
Grid models (point.json mode = grid) are decoded with two passes; their coverage is not computed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from decider import grid as G
from decider.data.point_build import read_jsonl
from decider.point_model import PointModel
from decider.point import PointQ, build_point, collate, hpd_level, mixture_logpdf, point_estimate

from decider import data as D
from decider.evaluate import aggregate, run_eval
from decider.prompt import build

QS = np.arange(0.1, 1.0, 0.1)


def to_dev(b, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}


def batches(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


@torch.no_grad()
def point_rows(model, examples, bs=16, max_ctx=4096):
    """One record per Point question: normalized estimate, target, log-density, HPD level, frame."""
    dev = next(model.parameters()).device
    gen = torch.Generator(device=dev).manual_seed(0)
    rows = []
    for chunk in batches(examples, bs):
        items = [build_point(e, model.tok, max_ctx_tokens=max_ctx) for e in chunk]
        b = to_dev(collate(items, model.tok.pad_token_id), dev)
        p = model(b)["point"]
        uv = b["point_targets"]
        est = point_estimate(p).cpu().numpy()
        logd = mixture_logpdf(p, uv).cpu().numpy()
        lev = hpd_level(p, uv, generator=gen).cpu().numpy()
        qs = [(e, q) for e in chunk for q in e.qs if isinstance(q, PointQ)]
        for k, (e, q) in enumerate(qs):
            rows.append({"task": e.task, "est": est[k].tolist(), "uv": uv[k].tolist(), "logd": float(logd[k]),
                         "hpd": float(lev[k]), "frame": q.frame})
    return rows


@torch.no_grad()
def grid_rows(model, examples, bs=8, max_ctx=4096):
    """Two-pass decode for the grid baseline; log-density from the teacher-forced fine pass."""
    dev = next(model.parameters()).device
    rng = random.Random(0)

    def probs(exs):
        items = [build(x, model.tok, rng, max_options=255, max_ctx_tokens=max_ctx) for x in exs]
        b = to_dev(collate(items, model.tok.pad_token_id), dev)
        pr = torch.softmax(model(b)["choice_logits"], -1).cpu().numpy()
        perms = [p for it in items for p in it["perms"]]
        out = []
        for row, perm in zip(pr, perms):  # back to the original option order
            v = np.zeros(len(perm))
            v[perm] = row[: len(perm)]
            out.append(v)
        return out

    rows = []
    for chunk in batches(examples, bs):
        qs = [(e, q) for e in chunk for q in e.qs]
        pc = probs([G.coarse_example(e) for e in chunk])
        pred_coarse = [divmod(int(np.argmax(p)), G.N) for p in pc]
        true = [G.fine_target(*q.frame.normalize(*q.target)) for _, q in qs]
        k = 0
        pred_c_by_ex, true_c_by_ex = [], []
        for e in chunk:
            pred_c_by_ex.append(pred_coarse[k:k + len(e.qs)])
            true_c_by_ex.append([t[0] for t in true[k:k + len(e.qs)]])
            k += len(e.qs)
        pf_pred = probs([G.fine_example(e, cc) for e, cc in zip(chunk, pred_c_by_ex)])
        pf_true = probs([G.fine_example(e, cc) for e, cc in zip(chunk, true_c_by_ex)])
        for i, (e, q) in enumerate(qs):
            (tr, tc), (fr, fc) = true[i]
            est = G.centre(pred_coarse[i], divmod(int(np.argmax(pf_pred[i])), G.N))
            logd = G.log_density(pc[i][tr * G.N + tc], pf_true[i][fr * G.N + fc])
            rows.append({"task": e.task, "est": list(est), "uv": list(q.frame.normalize(*q.target)), "logd": logd,
                         "hpd": None, "frame": q.frame})
    return rows


def summarize(rows):
    out = {}
    by = defaultdict(list)
    for r in rows:
        by[r["task"]].append(r)
    for task, rs in sorted(by.items()):
        est = np.array([r["est"] for r in rs])
        uv = np.array([r["uv"] for r in rs])
        w = np.array([[r["frame"].x1 - r["frame"].x0, r["frame"].y1 - r["frame"].y0] for r in rs])
        err = np.linalg.norm((est - uv) * w, axis=1)
        err_n = np.linalg.norm(est - uv, axis=1)
        centre_err = np.linalg.norm((0.5 - uv) * w, axis=1)
        s = {"n": len(rs), "units": rs[0]["frame"].units, "err_median": float(np.median(err)),
             "err_p90": float(np.percentile(err, 90)), "err_norm_median": float(np.median(err_n)),
             "nll": float(-np.mean([r["logd"] for r in rs])),
             "off_frame": float(np.mean((est < 0).any(1) | (est > 1).any(1))),
             "centre_err_median": float(np.median(centre_err))}
        if rs[0]["hpd"] is not None:
            lev = np.array([r["hpd"] for r in rs])
            cov = np.array([(lev <= q).mean() for q in QS])
            s.update({"cov50": float((lev <= 0.5).mean()), "cov90": float((lev <= 0.9).mean()),
                      "calib_err": float(np.mean(np.abs(cov - QS)))})
        out[task] = s
    return out


@torch.no_grad()
def latency(model, examples, sizes=(1, 8, 32), reps=5):
    dev = next(model.parameters()).device
    out = {}
    for n in sizes:
        items = [build_point(e, model.tok) for e in examples[:n]]
        b = to_dev(collate(items, model.tok.pad_token_id), dev)
        model(b)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t = []
        for _ in range(reps):
            t0 = time.perf_counter()
            model(b)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t.append(time.perf_counter() - t0)
        out[f"batch_{n}"] = {"ms": 1000 * float(np.median(t)), "tokens": int(b["attention_mask"].sum())}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--point_data", required=True)
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--limit_per_task", type=int, default=600)
    ap.add_argument("--replay", default=None, help="decider mixture pickle; runs decider's Choice regression eval")
    ap.add_argument("--choice_limit", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--soup", default=None, help="second model folder: evaluate the weight average of --model and --soup")
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of --soup in the average")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg_file = Path(a.model) / "point.json"
    mode = json.loads(cfg_file.read_text())["mode"] if cfg_file.exists() else "point"
    model = PointModel(a.model, dtype=getattr(torch, a.dtype), grad_ckpt=False).to(a.device).eval()
    if a.soup:
        other = PointModel(a.soup, dtype=getattr(torch, a.dtype), grad_ckpt=False).to(a.device).eval()
        model.interpolate_(other, a.alpha)
        del other
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"[soup] {1 - a.alpha:g} x {a.model} + {a.alpha:g} x {a.soup}", flush=True)
    result = {"model": a.model, "soup": a.soup, "alpha": a.alpha if a.soup else None, "mode": mode, "point": {}}
    for split in a.splits.split(","):
        exs = read_jsonl(Path(a.point_data) / f"{split}.jsonl.gz")
        by, keep = defaultdict(int), []
        for e in exs:
            if by[e.task] < a.limit_per_task:
                by[e.task] += 1
                keep.append(e)
        t0 = time.time()
        rows = point_rows(model, keep) if mode == "point" else grid_rows(model, keep)
        result["point"][split] = summarize(rows)
        print(f"[{split}] {len(keep)} rows in {time.time() - t0:.0f}s", flush=True)
        for task, s in result["point"][split].items():
            cal = f" cov50 {s['cov50']:.2f} cov90 {s['cov90']:.2f} calib_err {s['calib_err']:.3f}" if "cov50" in s else ""
            print(f"  {task:22s} n={s['n']:4d} err_med {s['err_median']:.3g} {s['units']} (centre {s['centre_err_median']:.3g}) "
                  f"p90 {s['err_p90']:.3g} nll {s['nll']:+.3f}{cal} off_frame {s['off_frame']:.3f}", flush=True)
        if mode == "point" and split == a.splits.split(",")[0]:
            result["latency"] = latency(model, keep)
            print("[latency]", json.dumps(result["latency"]), flush=True)
    if a.replay:
        _, evals = D.load_cache(a.replay)
        evals = {k: v[: a.choice_limit] for k, v in evals.items() if v}
        res, _ = run_eval(model.dm, evals, log=lambda *s: None)
        result["choice"] = aggregate(res)
        print("[choice]", json.dumps(result["choice"]), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1))
    print("[done]", a.out, flush=True)


if __name__ == "__main__":
    main()

"""Why are the hard geometry families off? Split each error into a wrong discrete choice vs imprecision.

For each scene the geometric candidates are re-derived from the question text:
  out_of_range  exit points: each enemy's radial exit and pairwise range-boundary crossings (outside_candidates)
  waypoint      the obstacle's 4 corners, plus the goal itself
  walkable      the nearest point on each of the 4 sides of the obstacle holding the ordered point (or the point)
A prediction "chooses" the candidate nearest to it. Wrong choice = that candidate is not the true one.
For toward (no discrete choice) the error is split into along-the-line and sideways parts.

Errors are in normalized frame units (as evaluate.py). Run on the GPU box:
  python -m decider.point_diag --model results/point/model --point_data data/point --out results/diag_geometry.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from decider.data.point_build import read_jsonl
from decider.point_evaluate import point_rows
from decider.point_model import PointModel
from decider.data.synth_geometry import inside, outside_candidates

NUM = r"(-?[\d.]+)"
FAMILIES = ("geo_out_of_range", "geo_waypoint", "geo_walkable", "geo_toward")


def pts(s):
    return [(float(a), float(b)) for a, b in re.findall(rf"\({NUM}, {NUM}\)", s)]


def candidates(e) -> list | None:
    """[(point, label)] in frame units, or None when the family has no discrete choice."""
    q = e.qs[0]
    if e.task == "geo_out_of_range":
        ens = [((float(x), float(y)), float(r)) for x, y, r in
               re.findall(rf"\(enemy\) at \({NUM}, {NUM}\), attack range {NUM}", e.context)]
        unit = pts(e.context.split("Units:")[1])[0]
        return [(p, "+".join(f"E{i}" for i in k)) for p, k in outside_candidates(unit, ens)]
    if e.task == "geo_waypoint":
        (x0, y0), (x1, y1) = pts(e.context.split("Obstacle:")[1].split("\n")[0])
        goal = pts(e.context.split("Goal:")[1])[0]
        return [((x0, y0), "corner x0y0"), ((x1, y0), "corner x1y0"), ((x1, y1), "corner x1y1"),
                ((x0, y1), "corner x0y1"), (goal, "goal")]
    if e.task == "geo_walkable":
        rects = [tuple(map(float, r)) for r in re.findall(rf"from \({NUM}, {NUM}\) to \({NUM}, {NUM}\)", e.context)]
        p = pts(q.text)[0]
        holder = next((r for r in rects if inside(p, r)), None)
        if holder is None:
            return [(p, "already walkable")]
        l, t, r_, b = holder
        return [((l, p[1]), "left side"), ((r_, p[1]), "right side"), ((p[0], t), "low-y side"), ((p[0], b), "high-y side")]
    return None


def classify(e, est_uv) -> dict:
    q = e.qs[0]
    f = q.frame
    est = f.denormalize(*est_uv)
    tu, tv = f.normalize(*q.target)
    err = math.hypot(est_uv[0] - tu, est_uv[1] - tv)
    out = {"task": e.task, "err": err}
    cands = candidates(e)
    if cands:
        true_i = min(range(len(cands)), key=lambda i: math.dist(cands[i][0], q.target))
        pick_i = min(range(len(cands)), key=lambda i: math.dist(cands[i][0], est))
        cu, cv = f.normalize(*cands[pick_i][0])
        out.update({"n_cands": len(cands), "wrong_choice": pick_i != true_i,
                    "true": cands[true_i][1], "picked": cands[pick_i][1],
                    "err_to_picked": math.hypot(est_uv[0] - cu, est_uv[1] - cv)})
    elif e.task == "geo_toward":
        name = q.text.split()[1]
        m = re.search(rf"- {name} \(ally\) at \({NUM}, {NUM}\)", e.context)
        unit = (float(m.group(1)), float(m.group(2)))
        uu, uv = f.normalize(*unit)
        d = np.array([tu - uu, tv - uv])
        n = np.linalg.norm(d)
        if n > 1e-9:
            d = d / n
            r = np.array([est_uv[0] - tu, est_uv[1] - tv])
            out.update({"along": float(r @ d), "sideways": float(abs(r[0] * d[1] - r[1] * d[0]))})
    return out


def summarize(rows) -> dict:
    out = {}
    by = defaultdict(list)
    for r in rows:
        by[r["task"]].append(r)
    for task, rs in sorted(by.items()):
        err = np.array([r["err"] for r in rs])
        s = {"n": len(rs), "err_median": float(np.median(err)), "err_mean": float(err.mean())}
        if "wrong_choice" in rs[0]:
            w = np.array([r["wrong_choice"] for r in rs])
            s.update({"wrong_choice_share": float(w.mean()),
                      "err_median_right_choice": float(np.median(err[~w])) if (~w).any() else None,
                      "err_median_wrong_choice": float(np.median(err[w])) if w.any() else None,
                      "error_mass_from_wrong_choice": float(err[w].sum() / err.sum()) if err.sum() else 0.0,
                      "err_to_picked_median_right_choice": float(np.median([r["err_to_picked"] for r in rs if not r["wrong_choice"]]))
                      if (~w).any() else None})
            conf = defaultdict(int)
            for r in rs:
                if r["wrong_choice"]:
                    conf[f"{r['true']} -> {r['picked']}"] += 1
            s["top_confusions"] = sorted(conf.items(), key=lambda kv: -kv[1])[:5]
        if any("along" in r for r in rs):
            a = np.array([abs(r["along"]) for r in rs if "along" in r])
            sd = np.array([r["sideways"] for r in rs if "sideways" in r])
            s.update({"along_median": float(np.median(a)), "sideways_median": float(np.median(sd))})
        out[task] = s
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--point_data", required=True)
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model = PointModel(a.model, dtype=getattr(torch, a.dtype), grad_ckpt=False).to(a.device).eval()
    result, all_rows = {}, []
    for split in a.splits.split(","):
        exs = [e for e in read_jsonl(Path(a.point_data) / f"{split}.jsonl.gz") if e.task in FAMILIES]
        rows = point_rows(model, exs)
        cls = [classify(e, r["est"]) for e, r in zip(exs, rows)]
        all_rows += [{**c, "split": split} for c in cls]
        result[split] = summarize(cls)
    result["both"] = summarize(all_rows)
    Path(a.out).write_text(json.dumps(result, indent=1))
    for task, s in result["both"].items():
        extra = (f" wrong_choice {s['wrong_choice_share']:.2f} (error mass {s['error_mass_from_wrong_choice']:.2f}); "
                 f"median err right/wrong {s['err_median_right_choice']:.4f}/{s['err_median_wrong_choice']}"
                 if "wrong_choice_share" in s else "")
        extra += f" along {s['along_median']:.4f} sideways {s['sideways_median']:.4f}" if "along_median" in s else ""
        print(f"{task:18s} n={s['n']} err_median {s['err_median']:.4f}{extra}", flush=True)
        if s.get("top_confusions"):
            print("   ", s["top_confusions"], flush=True)


if __name__ == "__main__":
    main()

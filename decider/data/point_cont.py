"""Data for the targeted continuation: fresh rows for the hard geometry families, optionally with an
auxiliary Choice question that names the discrete decision (asked in the same pass, before the Point).

  python -m decider.data.point_cont --out data/point_cont --per_family 20000 [--aux]

Auxiliary questions (only where the decision is unambiguous):
  waypoint      "Which obstacle corner does the shortest path turn at first?"   options: the 4 corners
  out_of_range  "The nearest safe point is on the edge of whose attack range?"   options: the enemies
                (skipped when the answer is where two ranges meet)
  walkable      "Which side of the blocked rectangle is the nearest walkable point on?"
Labels come from diag_geometry.candidates, the same derivation the diagnostic scores with.
"""

from __future__ import annotations

import argparse
import math
import random
import re
from pathlib import Path

from decider.data.point_build import Writer
from decider.point_diag import NUM, candidates
from decider.point import ChoiceQ
from decider.data.synth_geometry import make_example, pt

HARD = ("out_of_range", "waypoint", "toward", "walkable")


def aux_question(e, rng) -> ChoiceQ | None:
    cands = candidates(e)
    if not cands:
        return None
    q = e.qs[0]
    true = min(cands, key=lambda c: math.dist(c[0], q.target))
    f = q.frame
    if e.task == "geo_waypoint":
        corners = [c for c in cands if c[1].startswith("corner")]
        if true[1] == "goal":
            return None
        opts = [pt(c[0], f) for c in corners]
        order = list(range(4)); rng.shuffle(order)
        return ChoiceQ("Which obstacle corner does the shortest path turn at first?", [opts[i] for i in order],
                       order.index(corners.index(true)))
    if e.task == "geo_out_of_range":
        if "+" in true[1]:
            return None
        names = re.findall(rf"- (\w+) \(enemy\) at \({NUM}, {NUM}\)", e.context)
        names = [n[0] if isinstance(n, tuple) else n for n in names]
        i = int(true[1][1:])
        order = list(range(len(names))); rng.shuffle(order)
        return ChoiceQ("The nearest safe point is on the edge of whose attack range?", [names[j] for j in order], order.index(i))
    if e.task == "geo_walkable":
        if true[1] == "already walkable":
            return None
        sides = [c[1] for c in cands]
        order = list(range(len(sides))); rng.shuffle(order)
        return ChoiceQ("Which side of the blocked rectangle is the nearest walkable point on?",
                       [sides[j] for j in order], order.index(sides.index(true[1])))
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_family", type=int, default=20000)
    ap.add_argument("--aux", action="store_true")
    ap.add_argument("--seed", type=int, default=100)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    w = Writer(Path(a.out))
    n_aux = 0
    for fam in HARD:
        for _ in range(a.per_family):
            e = make_example(rng, fam)
            e.meta = {"source": "synthetic", "group": e.task}
            if a.aux:
                aq = aux_question(e, rng)
                if aq is not None:
                    e.qs = [aq] + e.qs
                    n_aux += 1
            w.add("train", e)
    w.close()
    print(f"[cont] {a.per_family * len(HARD)} rows, {n_aux} with an auxiliary choice question -> {a.out}", flush=True)


if __name__ == "__main__":
    main()

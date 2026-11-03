"""Crowded scenes with many questions packed in one row, exact answers: training data for the
packed-Point continuation (packing made Point answers 12-20% worse; every earlier Point row had
one question and at most ~10 entities).

A scene has 2-40 own units, 4-16 enemies and 3-8 markers in a random frame, written in one of
two styles (plain names, or Brood War-like "#41 Dragoon" / "E#87 Hydralisk" lines), and half the
scenes state the frame once ("Frame M") with Point questions referring to it. Each own unit
gets one question, kinds shuffled so Point and Choice questions appear at every depth:
  fraction  Point   the point a given fraction of the way from one marker to another
  toward    Point   a fixed distance straight toward an enemy (stopping at it)
  centroid  Point   the centre of 2-4 named enemies
  offset    Point   a distance north / south / east / west of a marker (y direction from the frame)
  nearest   Choice  which of up to 5 enemies is nearest (the nearest is clearly nearest)
  farthest  Choice  which of up to 5 markers is farthest

  python -m decider.data.crowded --out data/point_packed --train 6000 --val 300 --test 300
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from decider.point import ChoiceQ, PointExample, PointQ
from decider.data.synth_geometry import fmt, rand_point, random_frame, snap

OWN_TYPES = ["Zealot", "Dragoon", "Reaver", "High Templar", "Archon", "Marine", "Tank", "Wraith", "Vulture", "Goliath"]
FOE_TYPES = ["Zergling", "Hydralisk", "Mutalisk", "Lurker", "Ultralisk", "Defiler", "Scourge", "Queen"]
KINDS = ("fraction", "toward", "centroid", "offset", "nearest", "farthest")


def names(rng, style, n_own, n_foe, n_mark):
    if style == "plain":
        return [f"U{i + 1}" for i in range(n_own)], [f"E{i + 1}" for i in range(n_foe)], [f"M{i + 1}" for i in range(n_mark)]
    ids = rng.sample(range(10, 400), n_own + n_foe)
    own = [f"#{ids[i]} {rng.choice(OWN_TYPES)}" for i in range(n_own)]
    foe = [f"E#{ids[n_own + i]} {rng.choice(FOE_TYPES)}" for i in range(n_foe)]
    return own, foe, [f"waypoint W{i + 1}" for i in range(n_mark)]


def ambiguous_top(dists, margin):
    s = sorted(dists)
    return len(s) > 1 and s[1] - s[0] < margin


def pick_clear(rng, cands, key, margin, k=5):
    """The extreme candidate (smallest key) plus up to k-1 others clearly beyond it, shuffled; gold index."""
    by = sorted(cands, key=key)
    best = by[0]
    rest = [c for c in by[1:] if key(c) - key(best) >= margin]
    if not rest:
        return None
    opts = [best] + rng.sample(rest, min(k - 1, len(rest)))
    rng.shuffle(opts)
    return opts, opts.index(best)


def crowded_scene(rng: random.Random, n_q: int | None = None) -> PointExample:
    f = random_frame(rng)
    n_q = n_q or rng.choice([2, 3, 5, 8, 12, 16, 20, 25, 30, 35, 40])
    n_foe, n_mark = rng.randint(4, 16), rng.randint(3, 8)
    style = rng.choice(["plain", "bw"])
    own_n, foe_n, mark_n = names(rng, style, n_q, n_foe, n_mark)
    P = lambda: snap(rand_point(rng, f), f)
    own = [(n, P()) for n in own_n]
    foes = [(n, P()) for n in foe_n]
    marks = [(n, P()) for n in mark_n]
    ents = [(n, "ally", p) for n, p in own] + [(n, "enemy", p) for n, p in foes] + [(n, "marker", p) for n, p in marks]
    rng.shuffle(ents)
    compact = rng.random() < 0.5  # as Brood War sweeps will: frame stated once, questions name it
    head = f"Map: {f.describe()}.\nFrame M is the map.\n" if compact else f"Map: {f.describe()}.\n"
    ctx = head + "\n".join(f"- {n} ({k}) at ({fmt(p[0], f)}, {fmt(p[1], f)})" for n, k, p in ents)
    W = f.x1 - f.x0
    margin = 0.05 * min(W, f.y1 - f.y0)
    qs = []
    for u, pu in own:
        for kind in rng.sample(KINDS, len(KINDS)):  # first kind that can be asked unambiguously here
            q = make_q(rng, kind, u, pu, foes, marks, f, margin)
            if q is not None:
                if compact and isinstance(q, PointQ):
                    q.frame_ref = "M"
                qs.append(q)
                break
    return PointExample(ctx, qs, "crowded", {"n_q": len(qs), "entities": len(ents), "style": style, "compact": compact})


def make_q(rng, kind, u, pu, foes, marks, f, margin):
    if kind == "fraction":
        (a, pa), (b, pb) = rng.sample(marks, 2)
        frac, words = rng.choice([(0.5, "halfway"), (1 / 3, "one third of the way"), (2 / 3, "two thirds of the way"),
                                  (0.25, "a quarter of the way"), (0.75, "three quarters of the way")])
        return PointQ(f"Move {u} to the point {words} from {a} to {b}.", f,
                      (pa[0] + frac * (pb[0] - pa[0]), pa[1] + frac * (pb[1] - pa[1])))
    if kind == "toward":
        e, pe = rng.choice(foes)
        d = math.dist(pu, pe)
        if d < margin:
            return None
        step = float(fmt(rng.uniform(0.2, 1.2) * d, f)) or d / 2
        s = min(step, d) / d
        return PointQ(f"Move {u} {fmt(step, f)} {f.units} straight toward {e}, stopping at {e} if it is closer than that.", f,
                      (pu[0] + s * (pe[0] - pu[0]), pu[1] + s * (pe[1] - pu[1])))
    if kind == "centroid":
        grp = rng.sample(foes, rng.randint(2, min(4, len(foes))))
        names_ = ", ".join(n for n, _ in grp[:-1]) + f" and {grp[-1][0]}"
        return PointQ(f"Move {u} to the centre (average position) of {names_}.", f,
                      (sum(p[0] for _, p in grp) / len(grp), sum(p[1] for _, p in grp) / len(grp)))
    if kind == "offset":
        m, pm = rng.choice(marks)
        d = float(fmt(rng.uniform(0.05, 0.3) * min(f.x1 - f.x0, f.y1 - f.y0), f)) or 1.0
        direction = rng.choice(["north", "south", "east", "west"])
        north = -1 if f.y_down else 1  # north = up on the map
        dx, dy = {"east": (d, 0), "west": (-d, 0), "north": (0, north * d), "south": (0, -north * d)}[direction]
        t = (pm[0] + dx, pm[1] + dy)
        if not (f.x0 <= t[0] <= f.x1 and f.y0 <= t[1] <= f.y1):
            return None
        return PointQ(f"Move {u} to {fmt(d, f)} {f.units} {direction} of {m} (north is up on the map).", f, t)
    if kind == "nearest":
        r = pick_clear(rng, foes, lambda e: math.dist(pu, e[1]), margin)
        if r is None:
            return None
        opts, gold = r
        return ChoiceQ(f"Which enemy is nearest to {u}?", [n for n, _ in opts], gold)
    if kind == "farthest":
        r = pick_clear(rng, marks, lambda m: -math.dist(pu, m[1]), margin)
        if r is None:
            return None
        opts, gold = r
        return ChoiceQ(f"Which of these is farthest from {u}?", [n for n, _ in opts], gold)
    raise ValueError(kind)


def main():
    from decider.data.point_build import Writer
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=6000)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    w = Writer(Path(a.out))
    for split, n, s in (("train", a.train, a.seed), ("val", a.val, a.seed + 1), ("test", a.test, a.seed + 2)):
        rng = random.Random(s)
        for _ in range(n):
            w.add(split, crowded_scene(rng))
    w.close()
    print("wrote", a.out, dict(w.counts), flush=True)


if __name__ == "__main__":
    main()

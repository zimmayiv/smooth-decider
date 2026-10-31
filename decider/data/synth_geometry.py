"""Synthetic 2D geometry with exact answers, as Point questions over text scenes.

Every family is geometry a unit controller needs (where is halfway, where is out of
range, where is walkable, how do I get around this), with no game strategy in it.
Frames vary (normalized 0-1, metres, pixels with y downward, tiles) and are always
declared; scenes include distractor entities so the question must be read.

  families = FAMILIES; make_example(rng, family) -> PointExample
"""

from __future__ import annotations

import math
import random

from decider.point import Frame, PointExample, PointQ

NAMES = ["Alpha", "Bravo", "Cobalt", "Delta", "Ember", "Falcon", "Granite", "Harbor", "Iris", "Juniper",
         "Kestrel", "Lumen", "Mesa", "Nova", "Onyx", "Pike", "Quill", "Raven", "Sable", "Tarn"]


def random_frame(rng: random.Random) -> Frame:
    kind = rng.choice(["normalized", "metres", "pixels", "tiles"])
    if kind == "normalized":
        return Frame(0, 1, 0, 1, "normalized")
    if kind == "metres":
        w = rng.choice([50, 80, 105, 120, 200, 500])
        return Frame(0, w, 0, round(w * rng.uniform(0.5, 1.0)), "metres")
    if kind == "pixels":
        w = rng.choice([640, 1024, 1920, 2048, 4096, 8192])
        return Frame(0, w, 0, rng.choice([w, round(w * 0.75), round(w * 0.5625)]), "pixels", y_down=True)
    w = rng.choice([64, 96, 128, 192, 256])
    return Frame(0, w, 0, rng.choice([w, w // 2, round(w * 0.75)]), "tiles", y_down=True)


def fmt(v: float, frame: Frame) -> str:
    if frame.units == "normalized":
        return f"{v:.3f}"
    if frame.units == "metres":
        return f"{v:.1f}"
    return f"{round(v)}"


def pt(p, frame) -> str:
    return f"({fmt(p[0], frame)}, {fmt(p[1], frame)})"


def rand_point(rng, f: Frame, margin: float = 0.05):
    return (rng.uniform(f.x0 + margin * (f.x1 - f.x0), f.x1 - margin * (f.x1 - f.x0)),
            rng.uniform(f.y0 + margin * (f.y1 - f.y0), f.y1 - margin * (f.y1 - f.y0)))


def snap(p, f: Frame):
    """Round to the precision the scene is printed at, so the text and the label agree exactly."""
    return tuple(float(fmt(c, f)) for c in p)


def scale(f: Frame) -> float:
    return min(f.x1 - f.x0, f.y1 - f.y0)


def entities_text(ents, f: Frame) -> str:
    return "\n".join(f"- {name} ({kind}) at {pt(p, f)}" for name, kind, p in ents)


def header(f: Frame) -> str:
    return f"Map: {f.describe()}.\n"


def _scene(rng, f, n_extra, kinds=("ally", "enemy", "neutral")):
    names = rng.sample(NAMES, 4 + n_extra)
    return names, [(names[4 + i], rng.choice(kinds), snap(rand_point(rng, f), f)) for i in range(n_extra)]


# ---------------------------------------------------------------- families


def fam_fraction(rng):
    f = random_frame(rng)
    names, extra = _scene(rng, f, rng.randint(1, 4))
    u, a, b = names[0], names[1], names[2]
    pu, pa, pb = (snap(rand_point(rng, f), f) for _ in range(3))
    frac, words = rng.choice([(0.5, "halfway"), (1 / 3, "one third of the way"), (2 / 3, "two thirds of the way"),
                              (0.25, "a quarter of the way"), (0.75, "three quarters of the way")])
    ents = [(u, "ally", pu), (a, "marker", pa), (b, "marker", pb)] + extra
    rng.shuffle(ents)
    target = (pa[0] + frac * (pb[0] - pa[0]), pa[1] + frac * (pb[1] - pa[1]))
    return PointExample(header(f) + entities_text(ents, f), [PointQ(f"Move {u} to the point {words} from {a} to {b}.", f, target)],
                        "geo_fraction")


def fam_toward(rng):
    f = random_frame(rng)
    names, extra = _scene(rng, f, rng.randint(1, 4))
    u, b = names[0], names[1]
    pu, pb = snap(rand_point(rng, f), f), snap(rand_point(rng, f), f)
    dist = math.dist(pu, pb)
    step = float(fmt(rng.uniform(0.1, 1.3) * dist, f)) or 1.0
    s = min(step, dist) / dist if dist > 0 else 0.0
    target = (pu[0] + s * (pb[0] - pu[0]), pu[1] + s * (pb[1] - pu[1]))
    ents = [(u, "ally", pu), (b, "enemy", pb)] + extra
    rng.shuffle(ents)
    q = f"Move {u} {fmt(step, f)} {f.units} straight toward {b}, stopping at {b} if it is closer than that."
    return PointExample(header(f) + entities_text(ents, f), [PointQ(q, f, target)], "geo_toward")


def fam_centroid(rng):
    f = random_frame(rng)
    n = rng.randint(2, 6)
    names = rng.sample(NAMES, n + 4)
    group = [(names[i], "enemy", snap(rand_point(rng, f), f)) for i in range(n)]
    allies = [(names[n + i], "ally", snap(rand_point(rng, f), f)) for i in range(rng.randint(1, 4))]
    target = (sum(p[0] for _, _, p in group) / n, sum(p[1] for _, _, p in group) / n)
    ents = group + allies
    rng.shuffle(ents)
    q = f"Move {allies[0][0]} to the centre (average position) of all the enemy units."
    return PointExample(header(f) + entities_text(ents, f), [PointQ(q, f, target)], "geo_centroid")


def rects_disjoint(rng, f, n, min_gap):
    """Axis-aligned obstacles with a guaranteed gap between them (so nearest-free-point answers are exact)."""
    out, tries, s = [], 0, scale(f)
    while len(out) < n and tries < 500:
        tries += 1
        w, h = rng.uniform(0.08, 0.3) * s, rng.uniform(0.08, 0.3) * s
        x, y = rng.uniform(f.x0 + 0.05 * s, f.x1 - w - 0.05 * s), rng.uniform(f.y0 + 0.05 * s, f.y1 - h - 0.05 * s)
        r = tuple(float(fmt(v, f)) for v in (x, y, x + w, y + h))
        if all(r[0] > o[2] + min_gap or r[2] < o[0] - min_gap or r[1] > o[3] + min_gap or r[3] < o[1] - min_gap for o in out):
            out.append(r)
    return out


def inside(p, r) -> bool:
    return r[0] < p[0] < r[2] and r[1] < p[1] < r[3]


def nearest_outside_rect(p, r):
    """Nearest point on the boundary of r to a point inside it."""
    d = [p[0] - r[0], r[2] - p[0], p[1] - r[1], r[3] - p[1]]
    i = min(range(4), key=d.__getitem__)
    return [(r[0], p[1]), (r[2], p[1]), (p[0], r[1]), (p[0], r[3])][i]


def fam_walkable(rng):
    f = random_frame(rng)
    rects = rects_disjoint(rng, f, rng.randint(1, 4), min_gap=0.15 * scale(f))
    names, extra = _scene(rng, f, rng.randint(0, 3))
    u = names[0]
    if rng.random() < 0.8:  # usually ask about a point inside an obstacle
        r = rng.choice(rects)
        p = snap((rng.uniform(r[0], r[2]), rng.uniform(r[1], r[3])), f)
    else:
        p = snap(rand_point(rng, f), f)
    holder = next((r for r in rects if inside(p, r)), None)
    target = nearest_outside_rect(p, holder) if holder else p
    obs = "\n".join(f"- blocked rectangle from {pt((r[0], r[1]), f)} to {pt((r[2], r[3]), f)}" for r in rects)
    text = header(f) + "Unwalkable areas:\n" + obs + "\nUnits:\n" + entities_text([(u, "ally", snap(rand_point(rng, f), f))] + extra, f)
    q = f"{u} was ordered to {pt(p, f)}. Give the nearest walkable point to {pt(p, f)} (blocked rectangles are unwalkable inside, walkable on their edges)."
    return PointExample(text, [PointQ(q, f, target)], "geo_walkable")


def outside_candidates(p, circles):
    """Every candidate for the nearest point outside the union of open discs, with the discs it lies on:
    [(point, (i,) or (i, j))]. Candidates are radial projections onto each disc and pairwise boundary
    intersections, kept only if no disc contains them; the answer is the nearest of these."""
    cands = []
    for i, (c, r) in enumerate(circles):
        d = math.dist(p, c)
        u = ((p[0] - c[0]) / d, (p[1] - c[1]) / d) if d > 1e-12 else (1.0, 0.0)
        cands.append(((c[0] + r * u[0], c[1] + r * u[1]), (i,)))
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            (c1, r1), (c2, r2) = circles[i], circles[j]
            d = math.dist(c1, c2)
            if d == 0 or d > r1 + r2 or d < abs(r1 - r2):
                continue
            a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
            h = math.sqrt(max(0.0, r1 ** 2 - a ** 2))
            m = (c1[0] + a * (c2[0] - c1[0]) / d, c1[1] + a * (c2[1] - c1[1]) / d)
            off = (-(c2[1] - c1[1]) / d * h, (c2[0] - c1[0]) / d * h)
            cands += [((m[0] + off[0], m[1] + off[1]), (i, j)), ((m[0] - off[0], m[1] - off[1]), (i, j))]
    return [(q, k) for q, k in cands if not any(math.dist(q, c) < r - 1e-9 for c, r in circles)]


def nearest_outside_circles(p, circles):
    """Nearest point to p outside the union of open discs (exact: see outside_candidates)."""
    if not any(math.dist(p, c) < r for c, r in circles):
        return p
    return min((q for q, _ in outside_candidates(p, circles)), key=lambda q: math.dist(p, q))


def fam_out_of_range(rng):
    for _ in range(100):
        f = random_frame(rng)
        s = scale(f)
        n = rng.randint(1, 4)
        names = rng.sample(NAMES, n + 3)
        enemies = [(names[i], snap(rand_point(rng, f, 0.2), f), float(fmt(rng.uniform(0.08, 0.25) * s, f))) for i in range(n)]
        c, r = rng.choice([(e[1], e[2]) for e in enemies])
        ang, rad = rng.uniform(0, 2 * math.pi), rng.uniform(0, 0.95) * r
        pu = snap((c[0] + rad * math.cos(ang), c[1] + rad * math.sin(ang)), f)
        target = nearest_outside_circles(pu, [(e[1], e[2]) for e in enemies])
        if f.x0 <= target[0] <= f.x1 and f.y0 <= target[1] <= f.y1:
            break
    u = names[n]
    ens = "\n".join(f"- {nm} (enemy) at {pt(p, f)}, attack range {fmt(rr, f)} {f.units}" for nm, p, rr in enemies)
    text = header(f) + "Enemies:\n" + ens + f"\nUnits:\n- {u} (ally) at {pt(pu, f)}"
    q = f"Move {u} to the nearest point that is outside the attack range of every enemy."
    return PointExample(text, [PointQ(q, f, target)], "geo_out_of_range")


def fam_formation(rng):
    f = random_frame(rng)
    n = rng.randint(3, 8)
    k = rng.randint(1, n)
    names, extra = _scene(rng, f, rng.randint(0, 3))
    a, b = snap(rand_point(rng, f), f), snap(rand_point(rng, f), f)
    t = (k - 1) / (n - 1)
    target = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
    text = header(f) + f"Line formation endpoints: start {pt(a, f)}, end {pt(b, f)}.\nUnits:\n" + entities_text(extra, f) if extra else \
        header(f) + f"Line formation endpoints: start {pt(a, f)}, end {pt(b, f)}."
    q = (f"{n} units form an evenly spaced line from the start to the end, slot 1 at the start and slot {n} at the end. "
         f"Where does the unit in slot {k} stand?")
    return PointExample(text, [PointQ(q, f, target)], "geo_formation")


def segment_hits_rect(p, q, r, eps=1e-9) -> bool:
    """Does segment p-q pass through the open interior of rectangle r? (Liang-Barsky clip)"""
    t0, t1 = 0.0, 1.0
    dx, dy = q[0] - p[0], q[1] - p[1]
    for pv, qv in ((-dx, p[0] - r[0]), (dx, r[2] - p[0]), (-dy, p[1] - r[1]), (dy, r[3] - p[1])):
        if abs(pv) < eps:
            if qv <= eps:
                return False
            continue
        t = qv / pv
        if pv < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 >= t1 - eps:
            return False
    return t1 - t0 > eps


def shortest_first_waypoint(s, g, r):
    """First waypoint of the shortest path from s to g around rectangle r (corners as waypoints)."""
    if not segment_hits_rect(s, g, r):
        return g, math.dist(s, g)
    corners = [(r[0], r[1]), (r[2], r[1]), (r[2], r[3]), (r[0], r[3])]
    nodes = [s, g] + corners
    n = len(nodes)
    dist = [math.inf] * n
    first = [None] * n
    dist[0] = 0.0
    done = [False] * n
    for _ in range(n):
        i = min((j for j in range(n) if not done[j]), key=lambda j: dist[j])
        done[i] = True
        for j in range(n):
            if j != i and not done[j] and not segment_hits_rect(nodes[i], nodes[j], r):
                nd = dist[i] + math.dist(nodes[i], nodes[j])
                if nd < dist[j] - 1e-12:
                    dist[j] = nd
                    first[j] = nodes[j] if i == 0 else first[i]
    return first[1], dist[1]


def fam_waypoint(rng):
    for _ in range(200):
        f = random_frame(rng)
        s_ = scale(f)
        w, h = rng.uniform(0.15, 0.4) * s_, rng.uniform(0.15, 0.4) * s_
        x, y = rng.uniform(f.x0 + 0.25 * s_, f.x1 - w - 0.25 * s_), rng.uniform(f.y0 + 0.25 * s_, f.y1 - h - 0.25 * s_)
        r = tuple(float(fmt(v, f)) for v in (x, y, x + w, y + h))
        sp, gp = snap(rand_point(rng, f), f), snap(rand_point(rng, f), f)
        if inside(sp, r) or inside(gp, r) or not segment_hits_rect(sp, gp, r):
            continue
        wp, length = shortest_first_waypoint(sp, gp, r)
        # skip near-ties between going around either side: the label would be arbitrary
        corners = [(r[0], r[1]), (r[2], r[1]), (r[2], r[3]), (r[0], r[3])]
        alts = []
        for c in corners:
            if c != wp and not segment_hits_rect(sp, c, r):
                _, rest = shortest_first_waypoint(c, gp, r)
                alts.append(math.dist(sp, c) + rest)
        if alts and min(alts) - length < 0.05 * s_:
            continue
        break
    names = rng.sample(NAMES, 2)
    text = (header(f) + f"Obstacle: blocked rectangle from {pt((r[0], r[1]), f)} to {pt((r[2], r[3]), f)}.\n"
            f"Units:\n- {names[0]} (ally) at {pt(sp, f)}\nGoal: {pt(gp, f)}")
    q = (f"{names[0]} must reach the goal by the shortest path that does not cross the obstacle's interior. "
         f"Paths may touch the obstacle's corners. What is the first point the path turns at?")
    return PointExample(text, [PointQ(q, f, wp)], "geo_waypoint")


def fam_flank(rng):
    """Two equally correct answers: the label is one of them at random, so the head must learn two modes."""
    for _ in range(100):
        f = random_frame(rng)
        names, extra = _scene(rng, f, rng.randint(0, 3))
        u, e = names[0], names[1]
        pe = snap(rand_point(rng, f, 0.3), f)
        d = float(fmt(rng.uniform(0.08, 0.2) * scale(f), f))
        left, right = (pe[0] - d, pe[1]), (pe[0] + d, pe[1])
        if f.x0 <= left[0] and right[0] <= f.x1:
            break
    ents = [(u, "ally", snap(rand_point(rng, f), f)), (e, "enemy", pe)] + extra
    rng.shuffle(ents)
    q = f"Move {u} to a flanking spot exactly {fmt(d, f)} {f.units} from {e} along the x axis, on either side of it."
    return PointExample(header(f) + entities_text(ents, f), [PointQ(q, f, rng.choice([left, right]))], "geo_flank")


FAMILIES = {
    "fraction": fam_fraction, "toward": fam_toward, "centroid": fam_centroid, "walkable": fam_walkable,
    "out_of_range": fam_out_of_range, "formation": fam_formation, "waypoint": fam_waypoint, "flank": fam_flank,
}


def make_example(rng: random.Random, family: str) -> PointExample:
    return FAMILIES[family](rng)


def make_set(n: int, seed: int, families=None) -> list:
    rng = random.Random(seed)
    fams = list(families or FAMILIES)
    return [make_example(rng, fams[i % len(fams)]) for i in range(n)]

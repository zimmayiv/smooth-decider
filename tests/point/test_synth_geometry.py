import math
import random
import re

import pytest

from decider.data.synth_geometry import (FAMILIES, inside, make_example, make_set, nearest_outside_circles,
                                       segment_hits_rect, shortest_first_waypoint)

NUM = r"(-?[\d.]+)"


def points(text):
    return [(float(a), float(b)) for a, b in re.findall(rf"\({NUM}, {NUM}\)", text)]


@pytest.mark.parametrize("family", list(FAMILIES))
def test_every_family_builds_and_targets_lie_in_the_frame(family):
    rng = random.Random(0)
    for _ in range(200):
        ex = make_example(rng, family)
        q = ex.qs[0]
        f = q.frame
        tol = 1e-6 * (f.x1 - f.x0)
        assert f.x0 - tol <= q.target[0] <= f.x1 + tol and f.y0 - tol <= q.target[1] <= f.y1 + tol
        assert f.describe() in ex.context and ex.task == f"geo_{family}"


def test_out_of_range_matches_brute_force():
    rng = random.Random(1)
    for _ in range(300):
        circles = [((rng.uniform(0, 10), rng.uniform(0, 10)), rng.uniform(0.5, 3)) for _ in range(rng.randint(1, 4))]
        c, r = rng.choice(circles)
        p = (c[0] + rng.uniform(-r, r) * 0.7, c[1] + rng.uniform(-r, r) * 0.7)
        got = nearest_outside_circles(p, circles)
        assert not any(math.dist(got, cc) < rr - 1e-7 for cc, rr in circles)
        best = math.inf  # brute force over directions and radii
        for i in range(720):
            a = 2 * math.pi * i / 720
            for k in range(1, 600):
                q = (p[0] + 0.02 * k * math.cos(a), p[1] + 0.02 * k * math.sin(a))
                if not any(math.dist(q, cc) < rr for cc, rr in circles):
                    best = min(best, math.dist(p, q))
                    break
        assert math.dist(p, got) <= best + 0.03


def test_walkable_target_is_outside_obstacles_and_nearest():
    rng = random.Random(2)
    for _ in range(300):
        ex = make_example(rng, "walkable")
        q = ex.qs[0]
        rect_lines = re.findall(rf"from \({NUM}, {NUM}\) to \({NUM}, {NUM}\)", ex.context)
        rects = [tuple(map(float, r)) for r in rect_lines]
        p = points(q.text)[0]
        assert not any(inside(q.target, r) for r in rects)
        holder = [r for r in rects if inside(p, r)]
        if not holder:
            assert q.target == p
        else:
            r = holder[0]
            assert math.isclose(math.dist(p, q.target), min(p[0] - r[0], r[2] - p[0], p[1] - r[1], r[3] - p[1]))


def test_waypoint_is_on_a_valid_shortest_path():
    rng = random.Random(3)
    for _ in range(200):
        ex = make_example(rng, "waypoint")
        (rx0, ry0), (rx1, ry1), s, g = points(ex.context)[:4]
        r = (rx0, ry0, rx1, ry1)
        wp = ex.qs[0].target
        corners = [(r[0], r[1]), (r[2], r[1]), (r[2], r[3]), (r[0], r[3])]
        assert wp in corners and not segment_hits_rect(s, wp, r)
        # brute force over all corner sequences of length 1-2
        best = math.inf
        for c1 in corners:
            if segment_hits_rect(s, c1, r):
                continue
            if not segment_hits_rect(c1, g, r):
                best = min(best, math.dist(s, c1) + math.dist(c1, g))
            for c2 in corners:
                if c2 != c1 and not segment_hits_rect(c1, c2, r) and not segment_hits_rect(c2, g, r):
                    best = min(best, math.dist(s, c1) + math.dist(c1, c2) + math.dist(c2, g))
        _, length = shortest_first_waypoint(s, g, r)
        assert math.isclose(length, best, rel_tol=1e-9)


def test_simple_families_have_exact_closed_forms():
    rng = random.Random(4)
    for _ in range(200):
        ex = make_example(rng, "formation")
        a, b = points(ex.context)[:2]
        n, k = map(int, re.search(r"^(\d+) units.*slot (\d+) stand", ex.qs[0].text).groups())
        t = (k - 1) / (n - 1)
        assert ex.qs[0].target == pytest.approx((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))


def test_flank_is_genuinely_two_valued():
    rng = random.Random(5)
    sides = set()
    for _ in range(100):
        ex = make_example(rng, "flank")
        e = re.search(r"from (\w+) along", ex.qs[0].text).group(1)
        pe = points(re.search(rf"{e} \(enemy\) at (\([^)]*\))", ex.context).group(1))[0]
        d = float(re.search(rf"exactly {NUM}", ex.qs[0].text).group(1))
        tx, ty = ex.qs[0].target
        assert ty == pe[1] and math.isclose(abs(tx - pe[0]), d)
        sides.add(tx > pe[0])
    assert sides == {True, False}


def test_make_set_is_seeded_and_balanced():
    a, b = make_set(80, seed=7), make_set(80, seed=7)
    assert [e.context for e in a] == [e.context for e in b]
    assert {e.task for e in a} == {f"geo_{f}" for f in FAMILIES}

import math
import random
import re

from decider.data.crowded import crowded_scene
from decider.point import ChoiceQ, PointQ


def positions(ex):
    pos = {}
    for line in ex.context.splitlines()[1:]:
        m = re.match(r"- (.+) \((ally|enemy|marker)\) at \(([-\d.]+), ([-\d.]+)\)", line)
        if m is None:
            continue
        pos[m.group(1)] = (float(m.group(3)), float(m.group(4)))
    return pos


def test_scenes_terminate_and_answers_are_exact():
    rng = random.Random(0)
    kinds = set()
    for _ in range(400):
        ex = crowded_scene(rng)
        pos = positions(ex)
        assert 2 <= len(ex.qs) <= 40
        for q in ex.qs:
            if isinstance(q, ChoiceQ):
                u = re.search(r"(?:to|from) (.+)\?$", q.text).group(1)
                d = [math.dist(pos[u], pos[o]) for o in q.options]
                want = d.index(min(d)) if "nearest" in q.text else d.index(max(d))
                assert q.gold == want
                kinds.add("choice")
            else:
                f = q.frame
                assert f.x0 <= q.target[0] <= f.x1 + 1e-9 or "toward" in q.text or "centre" in q.text or "way" in q.text
                if "north of" in q.text:
                    m = q.text.split(" of ")[-1].split(" (")[0]
                    assert (q.target[1] < pos[m][1]) == f.y_down
                kinds.add(q.text.split()[2])
    assert "choice" in kinds and len(kinds) >= 4


def test_both_styles_and_point_questions_at_depth():
    rng = random.Random(1)
    exs = [crowded_scene(rng, 30) for _ in range(30)]
    assert {e.meta["style"] for e in exs} == {"plain", "bw"}
    deep = [q for e in exs for q in e.qs[20:]]
    assert any(isinstance(q, PointQ) for q in deep) and any(isinstance(q, ChoiceQ) for q in deep)


def test_compact_rows_round_trip_through_json():
    from decider.data.point_build import from_json, to_json
    from decider.point import question_piece
    rng = random.Random(3)
    exs = [crowded_scene(rng) for _ in range(40)]
    comp = [e for e in exs if e.meta["compact"]]
    assert comp and "Frame M is the map." in comp[0].context
    for e in exs:
        back = from_json(to_json(e))
        assert [question_piece(q) for q in back.qs] == [question_piece(q) for q in e.qs]
    assert any("Coordinate frame: M." in question_piece(q) for e in comp for q in e.qs if isinstance(q, PointQ))

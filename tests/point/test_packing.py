import math
import random
from pathlib import Path

import pytest
import torch

from decider.point_packing import answers, from_decider, multi_scene, score
from decider.point import ChoiceQ, PointQ, build_point

TINY = Path(__file__).resolve().parents[2] / ".cache" / "tiny_decider"


def test_multi_scene_answers_are_exact():
    ex = multi_scene(random.Random(0), 9)
    assert len(ex.qs) == 9 and sum(isinstance(q, ChoiceQ) for q in ex.qs) == 3
    pos = {}
    for line in ex.context.splitlines()[1:]:
        name, rest = line[2:].split(" (", 1)
        xy = rest.split(" at (")[1].rstrip(")").split(", ")
        pos[name] = (float(xy[0]), float(xy[1]))
    for q in ex.qs:
        if isinstance(q, ChoiceQ):
            u = q.text.split()[-1].rstrip("?")
            assert q.gold == min(range(len(q.options)), key=lambda j: math.dist(pos[u], pos[q.options[j]]))
        else:
            assert q.target is not None


def test_from_decider_keeps_gold_under_shuffle():
    class Q:  # decider.data.core.Q shape
        def __init__(s, text, options, gold): s.text, s.options, s.gold = text, options, gold
    class E:
        context, task = "ctx", "t"
        qs = [Q("a?", ["x", "y", "z"], 2), Q("b?", ["p", "q"], 0)]
    ex = from_decider(E, random.Random(1))
    assert [q.options[q.gold] for q in ex.qs] == ["z", "p"]
    E.qs = [Q("a?", ["x", "y"], -1), Q("b?", ["p", "q"], 0)]
    assert from_decider(E, random.Random(1)) is None


@pytest.mark.skipif(not TINY.exists(), reason="tiny decider checkpoint not built")
def test_answers_line_up_with_questions_in_both_modes():
    from decider.point_model import PointModel
    m = PointModel(str(TINY), dtype=torch.float32, grad_ckpt=False).eval()
    ex = multi_scene(random.Random(2), 7)
    for mode in ("packed", "independent"):
        a = answers(m, ex, mode, bs=3)
        assert [k for k, _ in a] == ["choice" if isinstance(q, ChoiceQ) else "point" for q in ex.qs]
        assert all(len(v) == len(q.options) for (k, v), q in zip(a, ex.qs) if k == "choice")
        assert len(score(ex, a)) == 7
    # independent answers equal single-question rows run one at a time
    one = answers(m, type(ex)(ex.context, [ex.qs[2]], ex.task), "packed")[0][1]
    assert torch.allclose(answers(m, ex, "independent", bs=4)[2][1], one, atol=1e-4)


def test_multi_scene_always_terminates():
    rng = random.Random(0)
    for n in (5, 20, 40):
        for _ in range(300):
            multi_scene(rng, n)

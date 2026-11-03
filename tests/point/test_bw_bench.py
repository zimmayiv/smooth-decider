from decider.data.bw_bench import PHASES, make_grouped_sweep, make_sweep
from decider.point import ChoiceQ, PointQ, question_piece


def test_per_unit_sweep_asks_every_own_entity():
    sw = make_sweep("mid", "none")
    workers, army, structs, _, _ = PHASES["mid"]
    commands = [q for q in sw.questions if isinstance(q, ChoiceQ) and ("command" in q.text or "what should" in q.text)]
    assert len(commands) == workers + army + structs


def test_group_size_one_lists_every_unit_and_larger_groups_shrink_the_state():
    fine = make_grouped_sweep("late", group_size=1)
    coarse = make_grouped_sweep("late", group_size=8)
    assert fine.meta["groups"] == PHASES["late"][1]
    assert coarse.meta["groups"] < fine.meta["groups"]
    assert len(coarse.dynamic) < len(fine.dynamic)


def test_question_budget_and_macro():
    for qb in (5, 20, 40):
        sw = make_grouped_sweep("late", group_size=2, q_budget=qb)
        assert sw.meta["micro_questions"] <= qb and sw.meta["macro_questions"] == 0
    micro = make_grouped_sweep("mid", macro=False)
    macro = make_grouped_sweep("mid", macro=True)
    per = make_grouped_sweep("mid", macro=True, macro_per_structure=True)
    assert "details on macro checks" in micro.dynamic and "Your structures:\n" in macro.dynamic
    assert 0 < macro.meta["macro_questions"] < per.meta["macro_questions"]


def test_compact_questions_refer_to_lists_stated_once():
    sw = make_grouped_sweep("mid", compact=True)
    assert "Command list C:" in sw.static and "Frame M is the map" in sw.static
    cmd = next(q for q in sw.questions if isinstance(q, ChoiceQ) and q.ref)
    pt = next(q for q in sw.questions if isinstance(q, PointQ))
    assert question_piece(cmd).endswith("Options: list C (A-J)\nAnswer: (")
    assert "Coordinate frame: M." in question_piece(pt)
    long = make_grouped_sweep("mid", compact=False)
    assert all(q.ref is None for q in long.questions if isinstance(q, ChoiceQ))

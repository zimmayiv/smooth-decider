import random

from decider.point_diag import FAMILIES, candidates, classify, summarize
from decider.data.synth_geometry import make_example


def test_perfect_predictions_never_count_as_wrong_choices():
    rng = random.Random(0)
    for fam in ("out_of_range", "waypoint", "walkable"):
        for _ in range(150):
            e = make_example(rng, fam)
            c = classify(e, e.qs[0].frame.normalize(*e.qs[0].target))
            assert c["err"] < 1e-9 and not c["wrong_choice"], (fam, c)


def test_predicting_another_candidate_is_a_wrong_choice():
    rng = random.Random(1)
    hits = 0
    for fam in ("out_of_range", "waypoint", "walkable"):
        for _ in range(150):
            e = make_example(rng, fam)
            q = e.qs[0]
            others = [p for p, _ in candidates(e) if abs(p[0] - q.target[0]) + abs(p[1] - q.target[1]) > 1e-6]
            if not others:
                continue
            c = classify(e, q.frame.normalize(*others[0]))
            assert c["wrong_choice"] and c["err_to_picked"] < 1e-9
            hits += 1
    assert hits > 200


def test_toward_splits_error_along_and_sideways():
    rng = random.Random(2)
    rows = []
    for _ in range(50):
        e = make_example(rng, "toward")
        u, v = e.qs[0].frame.normalize(*e.qs[0].target)
        rows.append(classify(e, (u + 0.01, v)))
    s = summarize(rows)["geo_toward"]
    assert abs(s["err_median"] - 0.01) < 1e-9 and "along_median" in s and "sideways_median" in s
    assert set(FAMILIES) >= {"geo_out_of_range", "geo_waypoint"}


def test_aux_questions_name_the_true_decision_and_round_trip(tmp_path):
    from decider.data.point_cont import aux_question
    from decider.data.point_build import Writer, read_jsonl
    from decider.point import ChoiceQ, PointQ
    rng = random.Random(3)
    made = 0
    w = Writer(tmp_path)
    for fam in ("out_of_range", "waypoint", "walkable"):
        for _ in range(100):
            e = make_example(rng, fam)
            aq = aux_question(e, rng)
            if aq is None:
                continue
            made += 1
            if fam == "waypoint":  # the gold option is the waypoint itself
                x, y = map(float, aq.options[aq.gold].strip("()").split(", "))
                assert (x, y) == tuple(map(float, e.qs[0].target)) or abs(x - e.qs[0].target[0]) < 1e-6
            e.qs = [aq] + e.qs
            e.meta = {"source": "synthetic"}
            w.add("train", e)
    w.close()
    back = read_jsonl(tmp_path / "train.jsonl.gz")
    assert made > 200 and len(back) == made
    assert isinstance(back[0].qs[0], ChoiceQ) and isinstance(back[0].qs[1], PointQ)


def test_mixed_row_builds_choice_and_point_slots():
    from decider.point import ChoiceQ, Frame, PointExample, PointQ, build_point, collate

    class Tok:
        def encode(self, s, add_special_tokens=False):
            return [ord(c) for c in s]
    f = Frame(0, 10, 0, 10, "tiles")
    ex = PointExample("ctx", [ChoiceQ("Which?", ["a", "b", "c"], 2), PointQ("Where?", f, (5, 5))])
    it = build_point(ex, Tok())
    assert it["golds"] == [2] and it["nopts"] == [3] and it["slots"][0] < it["point_slots"][0]
    b = collate([it], 0)
    assert b["slot_idx"].tolist() == it["slots"] and b["point_slot_idx"].tolist() == it["point_slots"]

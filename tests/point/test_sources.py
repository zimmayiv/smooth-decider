"""Converters on real sample files (downloaded into .cache/samples; skipped when absent)."""
import json
import math
import random
import re
from pathlib import Path

import pytest

S = Path(__file__).resolve().parents[2] / ".cache" / "samples"
need = lambda *names: pytest.mark.skipif(not all((S / n).exists() for n in names), reason="sample files not downloaded")


@need("sb_ev.json", "sb_360.json")
def test_statsbomb_instructed_targets_are_near_the_named_teammate():
    from decider.data.point_sources.statsbomb import match_examples
    ex = match_examples(json.load(open(S / "sb_ev.json")), json.load(open(S / "sb_360.json")), random.Random(0), match_id=1)
    inst = [e for e in ex if e.task == "sb_pass_instructed"]
    assert inst and len(ex) > len(inst)
    for e in inst:
        name = re.search(r"teammate (T\d+)", e.qs[0].text).group(1)
        x, y = map(float, re.search(rf"- {name}(?: \(goalkeeper\))? at \(([\d.]+), ([\d.]+)\)", e.context).groups())
        assert math.dist((x, y), e.qs[0].target) <= 3.0 + 0.1  # rendered at 0.1-yard precision
        assert 0 <= e.qs[0].target[0] <= 120 and 0 <= e.qs[0].target[1] <= 80


@need("me_ev.csv", "me_trk_Home.csv", "me_trk_Away.csv")
def test_metrica_pass_end_matches_the_event_file():
    from decider.data.point_sources.metrica import game_examples, read_events, read_tracking
    _, home = read_tracking(S / "me_trk_Home.csv"); _, away = read_tracking(S / "me_trk_Away.csv")
    ev = read_events(S / "me_ev.csv")
    ex = game_examples(ev, home, away, "g1", random.Random(0))
    passes = {(e.meta["frame"], e.qs[0].text): e.qs[0].target for e in ex if e.task == "metrica_pass"}
    row = next(r for r in ev if r["Type"] == "PASS" and r["To"])
    key = (int(row["Start Frame"]), f"{row['From']} passes to {row['To']}. Where is the ball received?")
    assert passes[key] == (float(row["End X"]), float(row["End Y"]))
    assert all(e.meta["source"] == "metrica" for e in ex)


@need("1886347_match.json", "1886347_dynamic_events.csv", "1886347_tracking_extrapolated.jsonl")
def test_skillcorner_runs_start_where_the_runner_is():
    from decider.data.point_sources.skillcorner import load_match, match_examples
    m, ev, trk = load_match(S, 1886347)
    ex = match_examples(m, ev, trk, random.Random(0))
    sgn = lambda r: -1.0 if r["attacking_side"] == "right_to_left" else 1.0  # events are attack-normalized
    runs = {(int(r["frame_start"]), sgn(r) * float(r["x_end"]), sgn(r) * float(r["y_end"])): r
            for r in ev if r["event_type"] == "off_ball_run"}
    near = 0
    for e in ex[:200]:
        r = runs[(e.meta["frame"], *e.qs[0].target)]
        num = re.match(r"#(\d+) ", e.qs[0].text).group(1)
        x, y = map(float, re.search(rf"- #{num} at \((-?[\d.]+), (-?[\d.]+)\)", e.context.split("Opponents:")[0]).groups())
        near += math.dist((x, y), (sgn(r) * float(r["x_start"]), sgn(r) * float(r["y_start"]))) < 3.0
    assert near / min(200, len(ex)) > 0.9  # the runner's tracked position is where the event says the run starts


@need("ac_part.gz")
def test_androidcontrol_taps_fall_inside_an_element():
    from decider.data.point_sources.androidcontrol import episode_examples, iter_records, parse_example
    ex = [e for r in iter_records(S / "ac_part.gz") for e in episode_examples(parse_example(r), "train")]
    assert ex
    for e in ex:
        x, y = e.qs[0].target
        boxes = [tuple(map(int, b)) for b in re.findall(r"at \((\d+), (\d+)\)-\((\d+), (\d+)\)", e.context)]
        assert any(l <= x <= r and t <= y <= b for l, t, r, b in boxes)
        assert e.qs[0].text.startswith("Next step: ") and ". Where should you" in e.qs[0].text


def test_jsonl_roundtrip_preserves_examples(tmp_path):
    from decider.data.point_build import Writer, read_jsonl
    from decider.data.synth_geometry import make_set
    ex = make_set(40, seed=3)
    w = Writer(tmp_path)
    for e in ex:
        e.meta = {"source": "synthetic", "group": e.task}
        w.add("train", e)
    w.close()
    back = read_jsonl(tmp_path / "train.jsonl.gz")
    assert [(b.context, b.task, b.qs[0].text, b.qs[0].frame, b.qs[0].target) for b in back] == \
           [(e.context, e.task, e.qs[0].text, e.qs[0].frame, tuple(e.qs[0].target)) for e in ex]


def test_held_out_split_is_stable_and_about_the_right_size():
    from decider.data.point_build import held_out
    assert held_out("sb123", 0.05) == held_out("sb123", 0.05)
    share = sum(held_out(f"sb{i}", 0.05) for i in range(4000)) / 4000
    assert 0.035 < share < 0.065

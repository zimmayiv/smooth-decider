"""SkillCorner open data (broadcast tracking, 10 matches) -> Point questions about off-ball runs.

Coordinates are metres with the origin at the centre of the pitch (x along the long side),
so values are negative on one half: a frame unlike the others. Players outside the broadcast
view are extrapolated and are marked "(estimated, off camera)", like last-seen positions.

Question: "#7 makes a <run type> run. Where does the run end?" -> the run's end point, from the
tracking frame at the run's start. SkillCorner's event coordinates are normalized to the attacking
direction while tracking is not, so events of a team attacking right to left are rotated 180 degrees
(checked: 323 of 328 such runs then start at the runner's tracked position, 1 without the rotation).
"""

from __future__ import annotations

import csv
import json
import random

from decider.point import Frame, PointExample, PointQ

BASE = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data"
LFS = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data"


def frame_for(match: dict) -> Frame:
    L, W = float(match["pitch_length"]), float(match["pitch_width"])
    return Frame(-L / 2, L / 2, -W / 2, W / 2, "metres from the centre spot")


def read_tracking(path: str, wanted: set) -> dict:
    """{frame: frame record} for the wanted frames only (the files are large)."""
    out = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d["frame"] in wanted and d.get("player_data"):
                out[d["frame"]] = d
    return out


def match_examples(match: dict, events: list, tracking: dict, rng: random.Random) -> list:
    fr = frame_for(match)
    info = {p["id"]: p for p in match["players"]}
    out = []
    for e in events:
        if e["event_type"] != "off_ball_run" or not e["x_end"]:
            continue
        t = tracking.get(int(e["frame_start"]))
        pid = int(e["player_id"])
        if t is None or pid not in info:
            continue
        team = info[pid]["team_id"]
        own, opp = [], []
        for p in t["player_data"]:
            if p["player_id"] not in info:
                continue
            q = info[p["player_id"]]
            line = (f"- #{q['number']} at ({p['x']:.1f}, {p['y']:.1f})"
                    + ("" if p.get("is_detected", True) else " (estimated, off camera)"))
            (own if q["team_id"] == team else opp).append(line)
        rng.shuffle(own)
        rng.shuffle(opp)
        direction = "+x" if e.get("attacking_side") == "left_to_right" else "-x"
        ball = t.get("ball_data") or {}
        text = "\n".join([f"Pitch: {fr.describe()}. Your team attacks toward {direction}."]
                         + ([f"Ball at ({ball['x']:.1f}, {ball['y']:.1f})."] if ball.get("x") is not None else [])
                         + ["Your team:"] + own + ["Opponents:"] + opp)
        kind = (e.get("event_subtype") or "off-ball").replace("_", " ")
        q = f"#{info[pid]['number']} makes a {kind} run. Where does the run end?"
        # Event coordinates are normalized so the team attacks left to right; tracking is not.
        # For a team attacking right to left, rotate the event point 180 degrees into the tracking frame.
        sgn = -1.0 if e.get("attacking_side") == "right_to_left" else 1.0
        end = (sgn * float(e["x_end"]), sgn * float(e["y_end"]))
        out.append(PointExample(text, [PointQ(q, fr, end)], "sc_run",
                                {"source": "skillcorner", "group": str(match["id"]), "frame": int(e["frame_start"])}))
    return out


def load_match(match_dir, match_id) -> tuple:
    """Read a downloaded match folder -> (match, events, tracking for the run start frames)."""
    from pathlib import Path
    d = Path(match_dir)
    match = json.load(open(d / f"{match_id}_match.json"))
    events = list(csv.DictReader(open(d / f"{match_id}_dynamic_events.csv")))
    wanted = {int(e["frame_start"]) for e in events if e["event_type"] == "off_ball_run"}
    return match, events, read_tracking(d / f"{match_id}_tracking_extrapolated.jsonl", wanted)

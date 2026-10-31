"""Metrica Sports sample data (games 1-2, CSV) -> Point questions. Held out entirely as the transfer test.

Coordinates are normalized to the pitch (0-1 on both axes, origin at the top-left, y downward).
Questions, from the tracking frame at the moment of each event:
  instructed pass  "PlayerA passes to PlayerB. Where is the ball received?" -> pass end
  movement         "Where will PlayerX be in 2 seconds?" -> tracked position 50 frames later (25 fps)
"""

from __future__ import annotations

import csv
import math
import random

from decider.point import Frame, PointExample, PointQ

BASE = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
FRAME = Frame(0, 1, 0, 1, "fraction of pitch length and width", y_down=True)
FPS = 25


def read_tracking(path: str) -> tuple[str, dict]:
    """-> (team, {frame: {player: (x, y)}}), dropping NaN positions (players off the pitch)."""
    with open(path) as f:
        rows = list(csv.reader(f))
    team = next(c for c in rows[0] if c)
    cols = rows[2]
    players = [(cols[i], i) for i in range(3, len(cols) - 2, 2) if cols[i].startswith("Player")]
    out = {}
    for r in rows[3:]:
        pos = {}
        for name, i in players:
            try:
                x, y = float(r[i]), float(r[i + 1])
            except (ValueError, IndexError):
                continue
            if not (math.isnan(x) or math.isnan(y)):
                pos[name] = (x, y)
        out[int(r[1])] = pos
    return team, out


def read_events(path: str) -> list:
    with open(path) as f:
        return list(csv.DictReader(f))


def render(own: dict, opp: dict, rng) -> str:
    lines = [f"Pitch: {FRAME.describe()}."]
    for label, pos in (("Your team", own), ("Opponents", opp)):
        names = sorted(pos)
        rng.shuffle(names)
        lines.append(f"{label}:")
        lines += [f"- {n} at ({pos[n][0]:.3f}, {pos[n][1]:.3f})" for n in names]
    return "\n".join(lines)


def game_examples(events: list, home: dict, away: dict, game: str, rng: random.Random, horizon_s: float = 2.0) -> list:
    out = []
    ahead = int(horizon_s * FPS)
    for ev in events:
        try:
            fr, x1, y1 = int(ev["Start Frame"]), float(ev["End X"]), float(ev["End Y"])
        except ValueError:
            continue
        h, a = home.get(fr), away.get(fr)
        if not h or not a:
            continue
        own = ev["Team"]
        mine, theirs = (h, a) if own == "Home" else (a, h)
        text = render(mine, theirs, rng)
        meta = {"source": "metrica", "group": game, "frame": fr}
        if ev["Type"] == "PASS" and ev["To"] and not (math.isnan(x1) or math.isnan(y1)):
            q = f"{ev['From']} passes to {ev['To']}. Where is the ball received?"
            out.append(PointExample(text, [PointQ(q, FRAME, (x1, y1))], "metrica_pass", dict(meta)))
        later = (home if own == "Home" else away).get(fr + ahead, {})
        movers = [p for p in mine if p in later]
        if movers and rng.random() < 0.5:
            p = rng.choice(movers)
            q = f"Where will {p} be in {horizon_s:g} seconds?"
            out.append(PointExample(text, [PointQ(q, FRAME, later[p])], "metrica_move", dict(meta)))
    return out

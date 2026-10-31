"""StatsBomb open data, 360 freeze frames -> Point questions about where a pass ends.

StatsBomb pitch: 120 x 80 yards, y increasing downward on their diagrams, with every
event oriented so the team in possession attacks toward x = 120. A freeze frame lists
only players inside the camera's visible area (a polygon), much like fog of war.

Two question kinds per pass (completed or not):
  instructed  "Pass to teammate T3." -> pass end location; T3 is the teammate nearest the
              end location, used only when that teammate is within `near` yards (completed passes)
  open        "The ball carrier plays a <height> pass. Where does the ball end up?" -> end location;
              genuinely uncertain, so it trains the mixture's spread
"""

from __future__ import annotations

import json
import math
import random

from decider.point import Frame, PointExample, PointQ

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
FRAME = Frame(0, 120, 0, 80, "yards", y_down=True)


def fmt(v: float) -> str:
    return f"{v:.1f}"


def render(freeze, visible_area, rng) -> tuple[str, dict]:
    """Text view of a freeze frame; returns (text, label -> location for teammates)."""
    mates = [p for p in freeze if p["teammate"] and not p["actor"]]
    opps = [p for p in freeze if not p["teammate"]]
    actor = next((p for p in freeze if p["actor"]), None)
    rng.shuffle(mates)
    rng.shuffle(opps)
    labels = {f"T{i + 1}": tuple(p["location"]) for i, p in enumerate(mates)}
    poly = visible_area or []
    corners = [(poly[i], poly[i + 1]) for i in range(0, len(poly) - 1, 2)]
    lines = [f"Pitch: {FRAME.describe()}. Your team attacks toward x = 120.",
             "Visible area (only players inside it are shown): polygon "
             + ", ".join(f"({fmt(x)}, {fmt(y)})" for x, y in corners) + "."]
    if actor:
        lines.append(f"Ball carrier (your team) at ({fmt(actor['location'][0])}, {fmt(actor['location'][1])}).")
    lines.append("Your teammates:")
    lines += [f"- {k}{' (goalkeeper)' if p['keeper'] else ''} at ({fmt(p['location'][0])}, {fmt(p['location'][1])})"
              for k, p in zip(labels, mates)]
    lines.append("Opponents:")
    lines += [f"- O{i + 1}{' (goalkeeper)' if p['keeper'] else ''} at ({fmt(p['location'][0])}, {fmt(p['location'][1])})"
              for i, p in enumerate(opps)]
    return "\n".join(lines), labels


def match_examples(events: list, frames: list, rng: random.Random, near: float = 3.0, match_id=None) -> list:
    ff = {f["event_uuid"]: f for f in frames}
    out = []
    for e in events:
        if e["type"]["name"] != "Pass" or e["id"] not in ff or "end_location" not in e.get("pass", {}):
            continue
        f = ff[e["id"]]
        if not any(p["actor"] for p in f["freeze_frame"]):
            continue
        end = tuple(e["pass"]["end_location"][:2])
        text, labels = render(f["freeze_frame"], f.get("visible_area"), rng)
        complete = "outcome" not in e["pass"]
        meta = {"source": "statsbomb", "group": str(match_id), "event": e["id"], "complete": complete}
        nearest = min(labels.items(), key=lambda kv: math.dist(kv[1], end), default=None)
        if complete and nearest and math.dist(nearest[1], end) <= near:
            q = f"Pass the ball to teammate {nearest[0]}. Where should the ball be played to?"
            out.append(PointExample(text, [PointQ(q, FRAME, end)], "sb_pass_instructed", dict(meta)))
        height = e["pass"].get("height", {}).get("name", "Ground Pass").lower()
        q = f"The ball carrier plays a {height}. Where does the ball end up?"
        out.append(PointExample(text, [PointQ(q, FRAME, end)], "sb_pass_open", dict(meta)))
    return out


def competitions_with_360(competitions: list) -> list:
    return [(c["competition_id"], c["season_id"]) for c in competitions if c.get("match_available_360")]


def fetch_json(url: str):
    import urllib.request
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def iter_matches(max_matches: int | None = None, seed: int = 0):
    """Yield (match_id, events, frames) for 360 matches, downloading one match at a time."""
    comps = competitions_with_360(fetch_json(f"{BASE}/competitions.json"))
    ids = []
    for cid, sid in comps:
        ids += [m["match_id"] for m in fetch_json(f"{BASE}/matches/{cid}/{sid}.json")]
    random.Random(seed).shuffle(ids)
    for mid in ids[:max_matches]:
        try:
            frames = fetch_json(f"{BASE}/three-sixty/{mid}.json")
        except Exception:
            continue
        yield mid, fetch_json(f"{BASE}/events/{mid}.json"), frames

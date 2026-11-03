"""Brood War-shaped synthetic sweeps for latency benchmarking (not training; answers are meaningless).

A sweep is the plan's Observation rendered as text, static part first (map, bases, terrain,
command catalogue: identical every sweep, so it can be cached) and dynamic part after it
(resources, production, own units and structures, visible and last-seen enemies, the current
plan), plus questions for every controlled entity:
  a Choice for the command type (always), a Choice for the target (some), a Point (some).

Sizes are knobs, not facts: unit counts per phase and line formats are assumptions from general
Brood War knowledge, and the real representation is not designed yet.

Two layouts:
  make_sweep          every unit listed and asked about (the first benchmark)
  make_grouped_sweep  what the LLM planner would set up: army units in groups of `group_size`
                      (1 = every unit on its own, for the finest micro), only the highest-priority
                      groups asked about (up to `q_budget` questions), workers summarized, and
                      structures listed and asked about only on macro sweeps; `compact` states the
                      command list and map frame once and refers to them from each question.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from decider.point import ChoiceQ, Frame, PointQ

MAP = Frame(0, 4096, 0, 4096, "pixels", y_down=True)  # a 128 x 128 tile map
PHASES = {  # own workers, army, structures; visible enemies; last-seen enemies
    "early": (12, 3, 4, 3, 1),
    "mid": (40, 50, 30, 50, 20),
    "late": (60, 110, 50, 80, 40),
}
TERRAIN = ("none", "regions", "grid")
ARMY = ["Zealot", "Dragoon", "High Templar", "Dark Templar", "Archon", "Reaver", "Observer", "Shuttle", "Corsair"]
STRUCT = {"Nexus": ["Train Probe"], "Gateway": ["Train Zealot", "Train Dragoon", "Train High Templar"],
          "Cybernetics Core": ["Research Singularity Charge"], "Forge": ["Upgrade Ground Weapons", "Upgrade Armor"],
          "Robotics Facility": ["Train Reaver", "Train Observer", "Train Shuttle"], "Pylon": [], "Photon Cannon": ["Attack"]}
ENEMY = ["Zergling", "Hydralisk", "Mutalisk", "Lurker", "Ultralisk", "Drone", "Overlord", "Sunken Colony", "Hatchery"]
COMMANDS = ["Continue current order", "Move", "Attack-move", "Attack unit", "Hold position", "Patrol",
            "Gather", "Build", "Return cargo", "Use ability"]
CATALOGUE = [
    "Move(point)", "AttackMove(point)", "Attack(unit)", "HoldPosition()", "Patrol(point)", "Stop()",
    "Gather(mineral|geyser)", "ReturnCargo()", "Build(structure type, tile point)", "Train(unit type)",
    "Research(tech)", "Upgrade(upgrade)", "UseTech(tech, point|unit)", "Load(unit)", "Unload(point)",
    "Cancel()", "Continue()",
]


@dataclass
class Sweep:
    static: str
    dynamic: str
    questions: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def context(self) -> str:
        return self.static + "\n" + self.dynamic


def p(rng):
    return (rng.randrange(32, 4064), rng.randrange(32, 4064))


def fmt(q):
    return f"({q[0]}, {q[1]})"


def static_part(rng, terrain: str) -> str:
    lines = [f"Map: {MAP.describe()}. 128 x 128 tiles of 32 pixels. Tileset: jungle."]
    bases = [p(rng) for _ in range(10)]
    lines.append("Bases (resource clusters): " + ", ".join(f"B{i} {fmt(b)}" for i, b in enumerate(bases)) + ".")
    if terrain == "regions":
        lines.append("Regions (centre; neighbouring regions):")
        for i in range(25):
            nb = sorted(j for j in rng.sample(range(25), 3) if j != i)
            lines.append(f"- R{i} {fmt(p(rng))}; " + ", ".join(f"R{j}" for j in nb))
    elif terrain == "grid":
        lines.append("Walkability, 32 x 32 cells of 128 pixels (row 0 at y = 0; '.' walkable, '#' blocked):")
        for _ in range(32):
            lines.append("".join("#" if rng.random() < 0.25 else "." for _ in range(32)))
    lines.append("Commands: " + ", ".join(CATALOGUE) + ".")
    return "\n".join(lines)


def dynamic_part(rng, phase: str) -> tuple[str, list]:
    workers, army, structs, vis, seen = PHASES[phase]
    own = []
    lines = [f"Frame {rng.randrange(1000, 40000):,}. Minerals {rng.randrange(0, 2000)}, gas {rng.randrange(0, 1200)}, "
             f"supply {min(200, workers + 2 * army)}/{min(200, workers + 2 * army + 16)}."]
    lines.append("Your units:")
    nid = 10
    for _ in range(workers):
        nid += 1; q = p(rng)
        own.append((nid, "Probe", q)); lines.append(f"- #{nid} Probe at {fmt(q)}, hp 20/20, order Gather mineral at B{rng.randrange(10)}")
    for _ in range(army):
        nid += 1; q = p(rng); t = rng.choice(ARMY)
        own.append((nid, t, q)); lines.append(f"- #{nid} {t} at {fmt(q)}, hp {rng.randrange(20, 100)}/100, order AttackMove to {fmt(p(rng))}")
    lines.append("Your structures:")
    for _ in range(structs):
        nid += 1; q = p(rng); t = rng.choice(list(STRUCT))
        own.append((nid, t, q))
        trains = [a.split(" ", 1)[1] for a in STRUCT[t] if a.startswith("Train ")]
        work = f", training {trains[0]} {rng.randrange(0, 100)}%" if trains and rng.random() < 0.5 else ""
        lines.append(f"- #{nid} {t} at {fmt(q)}{work}")
    enemies = []
    lines.append("Visible enemies:")
    for _ in range(vis):
        nid += 1; q = p(rng); t = rng.choice(ENEMY)
        enemies.append((nid, t, q)); lines.append(f"- E#{nid} {t} at {fmt(q)}, hp {rng.randrange(10, 400)}")
    lines.append("Last seen enemies:")
    for _ in range(seen):
        nid += 1
        lines.append(f"- E#{nid} {rng.choice(ENEMY)} last seen at {fmt(p(rng))}, frame {rng.randrange(1000, 40000):,}")
    lines.append('Plan: "Hold the natural with Dragoons; take the third base when supply passes 100; '
                 'scout the enemy main with one Observer; keep building Probes to 60."')
    return "\n".join(lines), own + [("enemy", e) for e in enemies]


def questions_for(rng, own_and_enemies) -> list:
    own = [x for x in own_and_enemies if x[0] != "enemy"]
    enemies = [x[1] for x in own_and_enemies if x[0] == "enemy"]
    qs = []
    for nid, t, q in own:
        if t in STRUCT:
            opts = (["Continue current order"] + STRUCT[t] + ["Cancel"])[:10]
            qs.append(ChoiceQ(f"Unit #{nid} ({t}): what should it do now?", opts, 0))
            continue
        qs.append(ChoiceQ(f"Unit #{nid} ({t}): which command now?", COMMANDS, 0))
        if enemies and rng.random() < 0.5:
            near = sorted(enemies, key=lambda e: (e[2][0] - q[0]) ** 2 + (e[2][1] - q[1]) ** 2)[:8]
            qs.append(ChoiceQ(f"Unit #{nid} ({t}): which target?", [f"E#{e[0]} {e[1]}" for e in near], 0))
        if rng.random() < 0.4:
            qs.append(PointQ(f"Unit #{nid} ({t}): where should it move?", MAP))
    return qs


def make_sweep(phase: str, terrain: str, seed: int = 0) -> Sweep:
    rng = random.Random(seed)
    static = static_part(rng, terrain)
    dynamic, ents = dynamic_part(rng, phase)
    return Sweep(static, dynamic, questions_for(rng, ents), {"phase": phase, "terrain": terrain})


# ---- grouped layout

ROLES = ["hold the natural", "push the third", "defend the main", "harass the mineral line", "escort the Shuttle",
         "contain the ramp", "clear the watchtower"]
COMBAT_RANGE = 800  # pixels: groups this close to an enemy may be asked for a target


def blobs(rng, n, per_blob, sd=90):
    """n positions in clusters of about per_blob around random centres."""
    k = max(1, round(n / per_blob))
    centres = [p(rng) for _ in range(k)]
    out = []
    for i in range(n):
        c = centres[i % k]
        out.append((i % k, (min(4064, max(32, int(rng.gauss(c[0], sd)))), min(4064, max(32, int(rng.gauss(c[1], sd)))))))
    return out


def chunk_groups(units, size, sizes: dict | None = None):
    """units: (blob, id, type, pos); split each blob (squad) into groups of at most `size`, or sizes[blob]
    for squads given their own granularity (sorted by x)."""
    groups = []
    for b in sorted({u[0] for u in units}):
        members = sorted((u for u in units if u[0] == b), key=lambda u: u[3])
        k = (sizes or {}).get(b, size)
        groups += [members[i:i + k] for i in range(0, len(members), k)]
    return groups


def centre(g):
    return (sum(u[3][0] for u in g) // len(g), sum(u[3][1] for u in g) // len(g))


def spread(g):
    c = centre(g)
    return int(max(((u[3][0] - c[0]) ** 2 + (u[3][1] - c[1]) ** 2) ** 0.5 for u in g))


def composition(g):
    counts = {}
    for u in g:
        counts[u[2]] = counts.get(u[2], 0) + 1
    return ", ".join(f"{n} {t}" for t, n in sorted(counts.items(), key=lambda x: -x[1]))


def dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def make_grouped_sweep(phase: str, terrain: str = "regions", seed: int = 0, *, group_size: int = 8,
                       enemy_group_size: int | None = None, q_budget: int = 30, macro: bool = False,
                       compact: bool = True, macro_per_structure: bool = False,
                       fine_squads: int = 0, fine_size: int = 1) -> Sweep:
    """fine_squads: how many squads the planner judges most important right now (here: closest to an enemy)
    are split finely, into groups of fine_size (1 = every unit on its own); the rest use group_size."""
    rng = random.Random(seed)
    egs = group_size if enemy_group_size is None else enemy_group_size
    workers, army, structs, vis, seen = PHASES[phase]
    static = static_part(rng, terrain)
    if compact:
        static += ("\nCommand list C: " + " ".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(COMMANDS)) + "."
                   + f"\nFrame M is the map: {MAP.describe()}.")

    nid = 10
    def ids(n):
        nonlocal nid
        out = list(range(nid + 1, nid + 1 + n)); nid += n
        return out
    own = [(b, i, rng.choice(ARMY), q) for i, (b, q) in zip(ids(army), blobs(rng, army, 12))]
    foes = [(b, i, rng.choice(ENEMY), q) for i, (b, q) in zip(ids(vis), blobs(rng, vis, 10))]
    sizes = {}
    if fine_squads:
        def squad_threat(b):
            members = [u for u in own if u[0] == b]
            c = centre(members)
            return min((dist2(c, e[3]) for e in foes), default=10 ** 9)
        for b in sorted({u[0] for u in own}, key=squad_threat)[:fine_squads]:
            sizes[b] = fine_size
    groups = chunk_groups(own, group_size, sizes)
    egroups = chunk_groups(foes, egs)
    bases = rng.sample(range(10), 3)

    lines = [f"Frame {rng.randrange(1000, 40000):,}. Minerals {rng.randrange(0, 2000)}, gas {rng.randrange(0, 1200)}, "
             f"supply {min(200, workers + 2 * army)}/{min(200, workers + 2 * army + 16)}."]
    split = [workers // 3 + (1 if i < workers % 3 else 0) for i in range(3)]
    lines.append("Probes: " + ", ".join(f"{n} mining at B{b}" for n, b in zip(split, bases)) + ".")
    lines.append("Your army:")
    labels, ng = [], 0
    for gi, g in enumerate(groups):
        role = ROLES[gi % len(ROLES)]
        if len(g) == 1:
            _, i, t, q = g[0]
            label = f"#{i} {t}"
            lines.append(f"- {label} at {fmt(q)}, hp {rng.randrange(20, 100)}%, squad S{g[0][0]} ({role}), order AttackMove {fmt(p(rng))}")
        else:
            ng += 1
            label = f"G{ng}"
            lines.append(f"- {label} ({role}): {len(g)} units, {composition(g)}, centre {fmt(centre(g))}, spread {spread(g)}, "
                         f"hp {rng.randrange(40, 100)}%, order AttackMove {fmt(p(rng))}")
        labels.append(label)
    sq = []
    if macro:  # one question per production structure, or (default) one per structure type
        lines.append("Your structures:")
        asked_types = set()
        for i in ids(structs):
            t = rng.choice(list(STRUCT)); q = p(rng)
            trains = [a.split(" ", 1)[1] for a in STRUCT[t] if a.startswith("Train ")]
            work = f", training {trains[0]} {rng.randrange(0, 100)}%" if trains and rng.random() < 0.5 else ""
            lines.append(f"- #{i} {t} at {fmt(q)}{work}")
            if STRUCT[t] and t != "Photon Cannon" and (macro_per_structure or t not in asked_types):
                who = f"#{i} {t}" if macro_per_structure else f"Idle {t}s"
                sq.append(ChoiceQ(f"{who}: what next?", (["Continue current order"] + STRUCT[t] + ["Cancel"])[:10], 0))
                asked_types.add(t)
    else:
        counts = {}
        for _ in range(structs):
            t = rng.choice(list(STRUCT)); counts[t] = counts.get(t, 0) + 1
        lines.append("Your structures: " + ", ".join(f"{n} {t}" for t, n in counts.items()) + " (details on macro checks).")
    lines.append("Visible enemies:")
    elabels = []
    for g in egroups:
        if len(g) == 1:
            _, i, t, q = g[0]
            label = f"E#{i} {t}"
            lines.append(f"- {label} at {fmt(q)}, hp {rng.randrange(10, 400)}")
        else:
            label = f"E{len(elabels) + 1}"
            lines.append(f"- {label}: {len(g)} units, {composition(g)}, centre {fmt(centre(g))}, spread {spread(g)}")
        elabels.append((label, centre(g)))
    lines.append("Last seen enemies:")
    for b in range(max(1, seen // 8)):
        lines.append(f"- {rng.randrange(2, 10)} {rng.choice(ENEMY)} near {fmt(p(rng))}, frame {rng.randrange(1000, 40000):,}")
    lines.append('Plan: "Hold the natural with Dragoons; take the third base when supply passes 100; '
                 'scout the enemy main with one Observer; keep building Probes to 60."')

    # priority: groups closest to an enemy first (in play the LLM sets this)
    def threat(g):
        c = centre(g)
        return min((dist2(c, e[1]) for e in elabels), default=10 ** 9)
    order = sorted(range(len(groups)), key=lambda gi: threat(groups[gi]))
    qs, asked = [], 0
    for gi in order:
        g, label, c = groups[gi], labels[gi], centre(groups[gi])
        new = [ChoiceQ(f"{label}: command?" if compact else f"{label} ({composition(g)}): which command now?",
                       COMMANDS, 0, ref="list C" if compact else None)]
        near = sorted(elabels, key=lambda e: dist2(c, e[1]))[:8]
        if near and dist2(c, near[0][1]) < COMBAT_RANGE ** 2:
            new.append(ChoiceQ(f"{label}: target?" if compact else f"{label}: which target?", [e[0] for e in near], 0))
        if rng.random() < 0.5:
            new.append(PointQ(f"{label}: move where?" if compact else f"{label}: where should it move?", MAP,
                              frame_ref="M" if compact else None))
        if len(qs) + len(new) > q_budget:
            break
        qs += new; asked += 1
    if macro:
        sq.append(ChoiceQ("Next building?", ["None"] + [t for t in STRUCT if t != "Nexus"] + ["Nexus"], 0))
        sq.append(PointQ("Where to build it?" if compact else "Where should the next building go?", MAP,
                         frame_ref="M" if compact else None))
    meta = {"phase": phase, "terrain": terrain, "layout": "grouped", "group_size": group_size, "q_budget": q_budget,
            "macro": macro, "compact": compact, "macro_per_structure": macro_per_structure,
            "fine_squads": fine_squads, "fine_size": fine_size, "squads": len({u[0] for u in own}), "groups": len(groups), "groups_asked": asked,
            "micro_questions": len(qs), "macro_questions": len(sq)}
    return Sweep(static, "\n".join(lines), qs + sq, meta)

"""AndroidControl (Google, Apache 2.0) -> Point questions: step instruction + element list -> tap point.

Records are GZIP TFRecords of tf.train.Example. Each episode has, per step, a screenshot
(ignored), a serialized AndroidAccessibilityForest (android_env proto), a step instruction
and an action JSON. Only tap-like actions with coordinates (click, long_press) become questions.
The accessibility trees are decoded here with a minimal protobuf reader for the few fields
used, so neither TensorFlow nor android_env is needed.
"""

from __future__ import annotations

import gzip
import json
import struct

from decider.point import Frame, PointExample, PointQ

BASE = "https://storage.googleapis.com/gresearch/android_control"


# ---------------------------------------------------------------- protobuf wire format (minimal)

def _varint(b, i):
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return r, i


def _fields(b):
    """Yield (field_number, wire_type, value) for one message; length-delimited values as bytes."""
    i = 0
    while i < len(b):
        key, i = _varint(b, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 2:
            n, i = _varint(b, i)
            v, i = b[i:i + n], i + n
        elif wt == 5:
            v, i = b[i:i + 4], i + 4
        elif wt == 1:
            v, i = b[i:i + 8], i + 8
        else:
            raise ValueError(f"unsupported wire type {wt}")
        yield f, wt, v


def _rect(b):
    r = {1: 0, 2: 0, 3: 0, 4: 0}  # ProtoRect: left, top, right, bottom
    for f, _, v in _fields(b):
        r[f] = v
    return r[1], r[2], r[3], r[4]


# AndroidAccessibilityNodeInfo field numbers (android_env/proto/a11y/android_accessibility_node_info.proto)
_NODE = {2: "bounds", 3: "class_name", 4: "content_description", 5: "hint_text", 7: "text",
         10: "view_id", 14: "clickable", 15: "editable", 16: "enabled", 21: "scrollable", 23: "visible"}


def decode_forest(b: bytes) -> list:
    """AndroidAccessibilityForest -> flat list of node dicts (windows in order, nodes in tree order)."""
    nodes = []
    for f, _, win in _fields(b):
        if f != 1:
            continue
        for wf, _, tree in _fields(win):
            if wf != 11:
                continue
            for tf, _, nb in _fields(tree):
                if tf != 1:
                    continue
                node = {}
                for nf, wt, v in _fields(nb):
                    name = _NODE.get(nf)
                    if name == "bounds":
                        node[name] = _rect(v)
                    elif name and wt == 2:
                        node[name] = v.decode("utf-8", "replace")
                    elif name:
                        node[name] = bool(v)
                nodes.append(node)
    return nodes


# ---------------------------------------------------------------- TFRecord / tf.train.Example

def iter_records(path: str):
    """Records of a GZIP TFRecord file (length-prefixed framing; CRCs not checked)."""
    with gzip.open(path, "rb") as f:
        while True:
            try:  # a truncated file ends the shard instead of failing the job
                head = f.read(12)
                if len(head) < 12:
                    return
                (n,) = struct.unpack("<Q", head[:8])
                data = f.read(n)
                f.read(4)
            except (EOFError, OSError):
                return
            if len(data) < n:
                return
            yield data


def parse_example(record: bytes) -> dict:
    from tfrecord import example_pb2
    ex = example_pb2.Example()
    ex.ParseFromString(record)
    out = {}
    for k, v in ex.features.feature.items():
        kind = v.WhichOneof("kind")
        out[k] = list(getattr(v, kind).value)
    return out


# ---------------------------------------------------------------- rendering

def element_lines(nodes: list, max_elements: int = 80) -> list:
    lines = []
    for n in nodes:
        if not n.get("visible") or "bounds" not in n:
            continue
        l, t, r, b = n["bounds"]
        if r <= l or b <= t:
            continue
        label = (n.get("text") or n.get("content_description") or n.get("hint_text") or "").strip().replace("\n", " ")
        flags = [w for w, k in (("clickable", "clickable"), ("editable", "editable"), ("scrollable", "scrollable")) if n.get(k)]
        if not (label or flags):
            continue
        cls = (n.get("class_name") or "View").split(".")[-1]
        rid = (n.get("view_id") or "").split("/")[-1]
        desc = f"{cls}" + (f" '{label[:50]}'" if label else "") + (f" id={rid}" if rid else "") + (f" [{', '.join(flags)}]" if flags else "")
        lines.append(f"- {desc} at ({l}, {t})-({r}, {b})")
        if len(lines) >= max_elements:
            break
    return lines


def episode_examples(ep: dict, split: str) -> list:
    if not ep.get("actions") or not ep.get("step_instructions") or not ep.get("goal"):
        return []  # protobuf omits empty lists: some episodes have no actions at all
    goal = ep["goal"][0].decode()
    widths, heights = ep["screenshot_widths"], ep["screenshot_heights"]
    out = []
    for i, (act_b, instr_b) in enumerate(zip(ep["actions"], ep["step_instructions"])):
        act = json.loads(act_b)
        if act.get("action_type") not in ("click", "long_press") or "x" not in act:
            continue
        frame = Frame(0, int(widths[i]), 0, int(heights[i]), "screen pixels", y_down=True)
        lines = element_lines(decode_forest(ep["accessibility_trees"][i]))
        if not lines:
            continue
        x, y = float(act["x"]), float(act["y"])
        if not (0 <= x <= frame.x1 and 0 <= y <= frame.y1):
            continue
        text = "\n".join([f"Screen: {frame.describe()}.", f"Overall goal: {goal}", "On-screen elements (bounds left, top - right, bottom):"] + lines)
        verb = "long-press" if act["action_type"] == "long_press" else "tap"
        step = instr_b.decode().strip()
        step = step if step.endswith((".", "!", "?")) else step + "."
        q = f"Next step: {step} Where should you {verb}?"
        out.append(PointExample(text, [PointQ(q, frame, (x, y))], "android_tap",
                                {"source": "androidcontrol", "group": str(ep["episode_id"][0]), "split": split}))
    return out


def episode_split(splits: dict, episode_id: int) -> str:
    for name, ids in splits.items():
        if episode_id in ids:
            return "val" if name == "validation" else name
    return "train"

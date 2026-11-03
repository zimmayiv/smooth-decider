"""Download and convert every Point source into fixed train / val / test splits (run on the GPU box).

  python -m decider.data.point_build --out data/point

Splits:
  synthetic      train, val and test from different seeds (every family in each)
  statsbomb      whole matches held out for val (stable hash of the match id, 5%)
  skillcorner    whole matches held out for val (1 of 10)
  androidcontrol the dataset's own episode splits (train / validation / test)
  metrica        test only: the source held out entirely, to test transfer to an unseen provider and frame
Large downloads are streamed one match or shard at a time and deleted after conversion.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path

from decider.point import ChoiceQ, Frame, PointExample, PointQ


def to_json(e: PointExample) -> dict:
    def q_json(q):
        if isinstance(q, ChoiceQ):
            d = {"type": "choice", "text": q.text, "options": list(q.options), "gold": q.gold}
            return {**d, "ref": q.ref} if q.ref else d
        d = {"text": q.text, "frame": vars(q.frame), "target": list(q.target)}
        return {**d, "frame_ref": q.frame_ref} if q.frame_ref else d
    return {"context": e.context, "task": e.task, "meta": e.meta, "qs": [q_json(q) for q in e.qs]}


def from_json(d: dict) -> PointExample:
    def q_obj(q):
        if q.get("type") == "choice":
            return ChoiceQ(q["text"], q["options"], q["gold"], ref=q.get("ref"))
        return PointQ(q["text"], Frame(**q["frame"]), tuple(q["target"]), frame_ref=q.get("frame_ref"))
    return PointExample(d["context"], [q_obj(q) for q in d["qs"]], d["task"], d.get("meta", {}))


def read_jsonl(path) -> list:
    with gzip.open(path, "rt") as f:
        return [from_json(json.loads(line)) for line in f]


class Writer:
    def __init__(self, out: Path):
        out.mkdir(parents=True, exist_ok=True)
        self.files = {s: gzip.open(out / f"{s}.jsonl.gz", "wt") for s in ("train", "val", "test")}
        self.counts = Counter()

    def add(self, split: str, e: PointExample):
        self.files[split].write(json.dumps(to_json(e)) + "\n")
        self.counts[(e.meta.get("source", "synthetic"), e.task, split)] += 1

    def close(self):
        for f in self.files.values():
            f.close()


def held_out(group: str, share: float) -> bool:
    return int(hashlib.md5(group.encode()).hexdigest(), 16) % 10_000 < share * 10_000


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=600) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
    return dest


def build_synthetic(w: Writer, n_train: int, n_eval_per_family: int):
    from decider.data.synth_geometry import FAMILIES, make_set
    for split, n, seed in (("train", n_train, 0), ("val", n_eval_per_family * len(FAMILIES), 1),
                           ("test", n_eval_per_family * len(FAMILIES), 2)):
        for e in make_set(n, seed=seed):
            e.meta = {"source": "synthetic", "group": e.task}
            w.add(split, e)


def build_statsbomb(w: Writer, max_matches: int, per_match: int, seed: int = 0):
    from decider.data.point_sources import statsbomb
    rng = random.Random(seed)
    for mid, events, frames in statsbomb.iter_matches(max_matches, seed):
        ex = statsbomb.match_examples(events, frames, rng, match_id=mid)
        rng.shuffle(ex)
        split = "val" if held_out(f"sb{mid}", 0.05) else "train"
        for e in ex[:per_match]:
            w.add(split, e)


def build_skillcorner(w: Writer, tmp: Path, seed: int = 0):
    from decider.data.point_sources import skillcorner as sc
    rng = random.Random(seed)
    ids = sorted(m["id"] for m in json.load(urllib.request.urlopen(f"{sc.BASE}/matches.json", timeout=120)))
    val_id = ids[0]
    for mid in ids:
        d = tmp / f"sc{mid}"
        download(f"{sc.BASE}/matches/{mid}/{mid}_match.json", d / f"{mid}_match.json")
        download(f"{sc.BASE}/matches/{mid}/{mid}_dynamic_events.csv", d / f"{mid}_dynamic_events.csv")
        download(f"{sc.LFS}/matches/{mid}/{mid}_tracking_extrapolated.jsonl", d / f"{mid}_tracking_extrapolated.jsonl")
        for e in sc.match_examples(*sc.load_match(d, mid), rng):
            w.add("val" if mid == val_id else "train", e)
        for f in d.iterdir():
            f.unlink()


def build_metrica(w: Writer, tmp: Path, seed: int = 0):
    from decider.data.point_sources import metrica
    rng = random.Random(seed)
    for g in (1, 2):
        base = f"{metrica.BASE}/Sample_Game_{g}/Sample_Game_{g}"
        paths = {k: download(f"{base}_{k}.csv", tmp / f"me{g}_{k}.csv")
                 for k in ("RawEventsData", "RawTrackingData_Home_Team", "RawTrackingData_Away_Team")}
        _, home = metrica.read_tracking(paths["RawTrackingData_Home_Team"])
        _, away = metrica.read_tracking(paths["RawTrackingData_Away_Team"])
        for e in metrica.game_examples(metrica.read_events(paths["RawEventsData"]), home, away, f"game{g}", rng):
            w.add("test", e)
        for p in paths.values():
            p.unlink()


def build_android(w: Writer, tmp: Path, shards: int, max_train_per_shard: int, seed: int = 0):
    from decider.data.point_sources import androidcontrol as ac
    rng = random.Random(seed)
    splits = json.load(urllib.request.urlopen(f"{ac.BASE}/splits.json", timeout=120))
    splits = {k: set(v) for k, v in splits.items()}
    for s in range(shards):
        path = download(f"{ac.BASE}/android_control-{s:05d}-of-00020", tmp / "ac_shard.gz")
        kept = 0
        for rec in ac.iter_records(path):
            ep = ac.parse_example(rec)
            split = ac.episode_split(splits, int(ep["episode_id"][0]))
            for e in ac.episode_examples(ep, split):
                if split == "train":
                    if kept >= max_train_per_shard or rng.random() > 0.5:
                        continue
                    kept += 1
                w.add(split, e)
        path.unlink()
        print(f"[android] shard {s}: done", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/point")
    ap.add_argument("--tmp", default="data/tmp")
    ap.add_argument("--sources", default="synthetic,statsbomb,skillcorner,metrica,androidcontrol")
    ap.add_argument("--synth_train", type=int, default=120_000)
    ap.add_argument("--synth_eval_per_family", type=int, default=400)
    ap.add_argument("--sb_matches", type=int, default=160)
    ap.add_argument("--sb_per_match", type=int, default=500)
    ap.add_argument("--android_shards", type=int, default=20)
    ap.add_argument("--android_train_per_shard", type=int, default=1500)
    a = ap.parse_args()
    out, tmp = Path(a.out), Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    w = Writer(out)
    builders = {
        "synthetic": lambda: build_synthetic(w, a.synth_train, a.synth_eval_per_family),
        "statsbomb": lambda: build_statsbomb(w, a.sb_matches, a.sb_per_match),
        "skillcorner": lambda: build_skillcorner(w, tmp),
        "metrica": lambda: build_metrica(w, tmp),
        "androidcontrol": lambda: build_android(w, tmp, a.android_shards, a.android_train_per_shard),
    }
    try:
        for name in a.sources.split(","):
            print(f"[build] {name}", flush=True)
            builders[name]()
            print(f"[build] {name} done: " + json.dumps({f"{t}/{s}": n for (src, t, s), n in w.counts.items()
                                                       if src == name or (name == "synthetic" and src == "synthetic")}), flush=True)
    finally:
        w.close()
        summary = {f"{src}|{t}|{s}": n for (src, t, s), n in sorted(w.counts.items())}
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("[build] wrote", out, flush=True)


if __name__ == "__main__":
    main()

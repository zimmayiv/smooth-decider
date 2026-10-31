"""Replace one source's rows in a built dataset with a separately built copy.

  python -m decider.data.point_merge --main data/point --part data/point_android --source androidcontrol
"""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--part", required=True)
    ap.add_argument("--source", required=True)
    a = ap.parse_args()
    counts = Counter()
    for split in ("train", "val", "test"):
        main, part = Path(a.main) / f"{split}.jsonl.gz", Path(a.part) / f"{split}.jsonl.gz"
        tmp = main.with_suffix(".tmp")
        with gzip.open(main, "rt") as fin, gzip.open(tmp, "wt") as fout:
            for line in fin:
                if json.loads(line).get("meta", {}).get("source") != a.source:
                    fout.write(line)
                else:
                    counts[f"dropped/{split}"] += 1
            if part.exists():
                with gzip.open(part, "rt") as fp:
                    for line in fp:
                        fout.write(line)
                        counts[f"added/{split}"] += 1
        tmp.replace(main)
    summ = json.loads((Path(a.main) / "summary.json").read_text())
    summ = {k: v for k, v in summ.items() if not k.startswith(a.source + "|")}
    summ.update({k: v for k, v in json.loads((Path(a.part) / "summary.json").read_text()).items() if k.startswith(a.source + "|")})
    (Path(a.main) / "summary.json").write_text(json.dumps(summ, indent=1))
    print(dict(counts))


if __name__ == "__main__":
    main()

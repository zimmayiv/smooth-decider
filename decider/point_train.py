"""Fine-tune decider for coordinates: LoRA on the backbone plus the Point head, with decider replay.

  point mode  Point rows (mixture head, NLL) + replay of decider's own mixture (decider's loss)
  grid mode   the same Point rows as coarse+fine Choice questions (decider's loss) + the same replay

Recipe follows decider-4b v2.1's own stage: LoRA rank 64, alpha 128, on attention, GatedDeltaNet
and MLP projections, LR 1e-4; the new head trains from scratch at --head_lr. Batches come from
decider's token-bucketed batcher over the combined rows. Saves a merged decider-format folder
(+ point_head.pt), so the result loads like any decider checkpoint.

  python -m decider.point_train --model Mapika/decider-4b --point_data data/point \
      --replay data/mixture_delta.pkl --mode point --out results/point_run
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch

from decider.data.point_build import read_jsonl
from decider.grid import to_grid_example
from decider.point_model import PointModel
from decider.point import build_point, collate, point_nll

from decider import data as D
from decider.prompt import build
from decider.train import batches_by_tokens, loss_fn, make_items


def take_tokens(items, budget):
    out, n = [], 0
    for it in items:
        if n >= budget:
            break
        out.append(it)
        n += len(it["ids"])
    return out, n


def point_items(examples, tok, mode, rng, max_ctx):
    out = []
    for e in examples:
        if mode == "point":
            it = build_point(e, tok, max_ctx_tokens=max_ctx)
        else:
            it = build(to_grid_example(e), tok, rng, max_options=255, max_ctx_tokens=max_ctx)
        it["task"] = e.task
        out.append(it)
    return out


def replay_items(path, tok, rng, budget, max_ctx):
    train, _ = D.load_cache(path)
    rng.shuffle(train)
    items, n, chunk = [], 0, 20000
    for c in range(0, len(train), chunk):
        part = make_items(train[c:c + chunk], tok, rng, max_ctx, none_prob=0.1, max_options=255)
        part, k = take_tokens(part, budget - n)
        items += part
        n += k
        if n >= budget:
            break
    return items, n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--point_data", required=True)
    ap.add_argument("--replay", required=True, help="decider mixture pickle (decider.data.mixture output)")
    ap.add_argument("--mode", choices=["point", "grid"], default="point")
    ap.add_argument("--out", required=True)
    ap.add_argument("--point_tokens", type=float, default=45e6)
    ap.add_argument("--point_data2", default=None, help="a second Point dataset mixed in (e.g. replay of the first run's data)")
    ap.add_argument("--point_tokens2", type=float, default=0)
    ap.add_argument("--init_adapter", default=None, help="continue training this adapter folder instead of a fresh LoRA")
    ap.add_argument("--replay_tokens", type=float, default=35e6)
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--max_tokens", type=int, default=16384)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max_ctx", type=int, default=4096)
    ap.add_argument("--point_weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a")

    def log(*s):
        msg = " ".join(str(x) for x in s)
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log("[args]", json.dumps(vars(a)))
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    if a.init_adapter:
        model = PointModel(a.init_adapter, dtype=getattr(torch, a.dtype), grad_ckpt=True, trainable_adapter=True)
        targets = sorted(model.dm.lm.peft_config["default"].target_modules)
    else:
        model = PointModel(a.model, dtype=getattr(torch, a.dtype), grad_ckpt=True)
        targets = model.add_lora(a.lora_r, a.lora_alpha)
    model.to(a.device)
    tok = model.tok
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[model] LoRA on {targets}; trainable parameters {n_train / 1e6:.1f}M")

    t0 = time.time()
    pts = read_jsonl(Path(a.point_data) / "train.jsonl.gz")
    rng.shuffle(pts)
    pitems, pn = take_tokens(point_items(pts, tok, a.mode, rng, a.max_ctx), a.point_tokens)
    if a.point_data2 and a.point_tokens2 > 0:
        pts2 = read_jsonl(Path(a.point_data2) / "train.jsonl.gz")
        rng.shuffle(pts2)
        more, n2 = take_tokens(point_items(pts2, tok, a.mode, rng, a.max_ctx), a.point_tokens2)
        pitems, pn = pitems + more, pn + n2
    ritems, rn = replay_items(a.replay, tok, rng, a.replay_tokens, a.max_ctx)
    items = pitems + ritems
    by_task = {}
    for it in pitems:
        by_task[it["task"]] = by_task.get(it["task"], 0) + 1
    log(f"[data] point rows {len(pitems)} ({pn / 1e6:.1f}M tokens), replay rows {len(ritems)} ({rn / 1e6:.1f}M tokens), "
        f"built in {time.time() - t0:.0f}s; point rows by task {json.dumps(by_task)}")

    batches = batches_by_tokens(items, a.max_tokens, random.Random(a.seed))
    total = len(batches) // a.accum
    lora = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("point_head")]
    head = list(model.point_head.parameters())
    opt = torch.optim.AdamW([{"params": lora, "lr": a.lr}, {"params": head, "lr": a.head_lr}],
                            weight_decay=0.0, betas=(0.9, 0.95))
    base_lrs = [a.lr, a.head_lr]

    def scale(s):
        if s < a.warmup:
            return s / a.warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - a.warmup) / max(1, total - a.warmup))))

    log(f"[sched] {total} optimizer steps, {len(batches)} micro-batches")
    model.train()
    step, t0, t_last = 0, time.time(), time.time()
    acc = {"choice": 0.0, "point": 0.0, "n_c": 0, "n_p": 0, "tok": 0}
    for micro, idx in enumerate(batches[: total * a.accum]):
        b = collate([items[i] for i in idx], tok.pad_token_id)
        b = {k: (v.to(a.device) if torch.is_tensor(v) else v) for k, v in b.items()}
        o = model(b)
        loss = 0.0
        if "choice_logits" in o:
            lc, ce = loss_fn(o["choice_logits"], b["golds"], b["nopts"])
            loss = loss + lc
            acc["choice"] += float(ce); acc["n_c"] += 1
        if "point" in o:
            lp = point_nll(o["point"], b["point_targets"])
            loss = loss + a.point_weight * lp
            acc["point"] += float(lp.detach()); acc["n_p"] += 1
        (loss / a.accum).backward()
        acc["tok"] += b["input_ids"].numel()
        if (micro + 1) % a.accum:
            continue
        for g_, base in zip(opt.param_groups, base_lrs):
            g_["lr"] = base * scale(step)
        gn = torch.nn.utils.clip_grad_norm_(lora + head, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 25 == 0 or step == total:
            now = time.time()
            mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            log(f"[train] step {step}/{total} choice_ce {acc['choice'] / max(1, acc['n_c']):.4f} "
                f"point_nll {acc['point'] / max(1, acc['n_p']):.4f} gn {float(gn):.2f} lr {a.lr * scale(step):.2e} "
                f"{(now - t0) / 60:.1f}min eta {(total - step) * (now - t_last) / 25 / 60:.0f}min "
                f"{acc['tok'] / (now - t_last):.0f}tok/s mem {mem:.0f}GB")
            acc = {"choice": 0.0, "point": 0.0, "n_c": 0, "n_p": 0, "tok": 0}
            t_last = now

    model.eval()
    model.save(str(out / "model"), base=str(Path(a.model).resolve()))  # adapter + head (small)
    for f in ("decider_config.json", "chat_template.jinja"):
        if (Path(a.model) / f).exists():
            shutil.copy(Path(a.model) / f, out / "model" / f)
    (out / "model" / "point.json").write_text(json.dumps({"mode": a.mode, "base": str(Path(a.model).resolve()),
                                                               "lora_r": a.lora_r, "init_adapter": a.init_adapter}))
    log(f"[save] {out / 'model'}")


if __name__ == "__main__":
    main()

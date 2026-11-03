"""Latency of one Brood War-shaped sweep under different ways of running it (eager PyTorch).

  a  independent  one row per question (state + that question); timed on a sample of questions and
                  scaled to the whole sweep (marked "est")
  b  packed       one row: the state, then every question (later questions can read earlier ones)
  c  shared       the state once with a cache; the cache is copied per question and only the
                  question tokens run (same answers as a, checked by --check)
  e  static       as c, but the static part (map, terrain, catalogue) is cached across sweeps and
                  only the dynamic part plus the questions run each sweep

Every mode sees identical tokens: "Context:\\n" + static, then "\\n" + dynamic, then each question.

  python -m decider.point_latency --model path/to/point/model --out results/latency.json
  # grouped sweeps (bw_bench.make_grouped_sweep): every combination of the knobs, packed and shared prefix
  python -m decider.point_latency --model ... --layout grouped --group_sizes 1,2,4,8,12 --q_budgets 20,40 --macro 0,1
  # token counts only (no model, runs anywhere with the tokenizer)
  python -m decider.point_latency --tokens_only --tokenizer path/to/qwen3.5 --layout grouped ...

Each sweep is compared with the per-sweep compute budget at several decision cadences:
cadence frames x 42 ms (Brood War "fastest" game speed) minus the network round trip (--rtt).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from decider.data.bw_bench import PHASES, TERRAIN, make_grouped_sweep, make_sweep
from decider.point_model import PointModel
from decider.point import ChoiceQ, collate, question_piece

from decider.model import cap_logits
from decider.prompt import MAX_OPTIONS


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Runner:
    def __init__(self, model: PointModel, mem_budget_gb: float = 6.0):
        self.m = model
        self.tok = model.tok
        self.dev = next(model.parameters()).device
        self.mem_budget = mem_budget_gb * 2 ** 30

    def enc(self, s):
        return self.tok.encode(s, add_special_tokens=False)

    def sweep_ids(self, sw):
        static, dynamic = self.enc("Context:\n" + sw.static), self.enc("\n" + sw.dynamic)
        qs = [(self.enc(question_piece(q)), q) for q in sw.questions]
        return static, dynamic, qs

    def readout(self, h_slots, kinds, nopts):
        """Answers for slot hidden states: Choice -> option probabilities, Point -> mixture params."""
        out = {}
        ci = [i for i, k in enumerate(kinds) if k == "choice"]
        pi = [i for i, k in enumerate(kinds) if k == "point"]
        if ci:
            W = self.m.lm_head_weight()[self.m.dm.letters]
            lg = cap_logits(self.m.dm, F.linear(h_slots[ci], W).float())
            no = torch.tensor([nopts[i] for i in ci], device=lg.device)
            lg = lg.masked_fill(torch.arange(MAX_OPTIONS, device=lg.device)[None] >= no[:, None], float("-inf"))
            out["choice"] = (ci, torch.softmax(lg, -1))
        if pi:
            out["point"] = (pi, self.m.point_head(h_slots[pi]))
        return out

    # ---- a: independent rows
    @torch.no_grad()
    def independent(self, ctx_ids, qs, rows=8):
        items = []
        for ids, q in qs:
            full = ctx_ids + ids
            it = {"ids": full}
            if isinstance(q, ChoiceQ):
                it.update(slots=[len(full) - 1], golds=[0], nopts=[len(q.options)])
            else:
                it.update(point_slots=[len(full) - 1], point_targets=[(0.5, 0.5)])
            items.append(it)
        outs = []
        for c in range(0, len(items), rows):
            b = {k: (v.to(self.dev) if torch.is_tensor(v) else v) for k, v in collate(items[c:c + rows], self.tok.pad_token_id).items()}
            outs.append(self.m(b))
        return outs

    # ---- b: packed
    @torch.no_grad()
    def packed(self, ctx_ids, qs_text):
        ids = list(ctx_ids)
        slots, kinds, nopts = [], [], []
        for k, q in enumerate(qs_text):
            ids += self.enc(question_piece(q, f" {k + 1}"))
            slots.append(len(ids) - 1)
            kinds.append("choice" if isinstance(q, ChoiceQ) else "point")
            nopts.append(len(q.options) if isinstance(q, ChoiceQ) else 0)
        x = torch.tensor([ids], device=self.dev)
        h = self.m.backbone()(input_ids=x).last_hidden_state[0]
        return self.readout(h[slots], kinds, nopts), len(ids)

    # ---- c / e: shared prefix
    @torch.no_grad()
    def prefix(self, ids, cache=None):
        x = torch.tensor([ids], device=self.dev)
        out = self.m.backbone()(input_ids=x, past_key_values=cache, use_cache=True)
        return out.past_key_values

    @staticmethod
    def cache_bytes(cache) -> int:
        tot = 0
        for t in _tensors(cache):
            tot += t.numel() * t.element_size()
        return tot

    @torch.no_grad()
    def questions(self, cache, P, qs):
        """Run each question against a copy of the prefix cache, in chunks that fit the memory budget."""
        per_q = max(1, self.cache_bytes(cache))
        chunk = max(1, min(64, int(self.mem_budget // per_q)))
        h_all, kinds, nopts = [], [], []
        for c in range(0, len(qs), chunk):
            part = qs[c:c + chunk]
            B, T = len(part), max(len(ids) for ids, _ in part)
            x = torch.full((B, T), self.tok.pad_token_id, dtype=torch.long, device=self.dev)
            mask = torch.zeros((B, P + T), dtype=torch.long, device=self.dev)
            mask[:, :P] = 1
            for i, (ids, _) in enumerate(part):
                x[i, :len(ids)] = torch.tensor(ids, device=self.dev)
                mask[i, P:P + len(ids)] = 1
            cc = fork_cache(cache, B)
            h = self.m.backbone()(input_ids=x, attention_mask=mask, past_key_values=cc, use_cache=True).last_hidden_state
            h_all.append(h[torch.arange(B, device=self.dev), torch.tensor([len(ids) - 1 for ids, _ in part], device=self.dev)])
            for _, q in part:
                kinds.append("choice" if isinstance(q, ChoiceQ) else "point")
                nopts.append(len(q.options) if isinstance(q, ChoiceQ) else 0)
            del cc
        return self.readout(torch.cat(h_all), kinds, nopts), chunk


def _tensors(cache):
    """Every tensor held by a cache: attention keys/values, and linear-attention (GatedDeltaNet)
    conv/recurrent states, which transformers keeps in dicts keyed by state index."""
    seen = []
    for layer in getattr(cache, "layers", []):
        for v in vars(layer).values():
            if torch.is_tensor(v):
                seen.append(v)
            elif isinstance(v, (list, tuple)):
                seen += [t for t in v if torch.is_tensor(t)]
            elif isinstance(v, dict):
                seen += [t for t in v.values() if torch.is_tensor(t)]
    return seen


def fork_cache(cache, n: int):
    """A copy of a batch-1 cache repeated to batch n. transformers' batch_repeat_interleave does not
    cover linear-attention layers, so every tensor (all have batch as dim 0) is repeated here."""
    cc = copy.deepcopy(cache)
    for layer in cc.layers:
        for name, v in vars(layer).items():
            if torch.is_tensor(v) and v.dim() > 0:
                setattr(layer, name, v.repeat_interleave(n, dim=0))
            elif isinstance(v, dict):
                for k, t in v.items():
                    if torch.is_tensor(t) and t.dim() > 0:
                        v[k] = t.repeat_interleave(n, dim=0)
    return cc


def kernel_report() -> dict:
    """Which implementation transformers will run for the GatedDeltaNet pieces."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    name = lambda f: getattr(f, "__module__", "?") + "." + getattr(f, "__name__", "?")
    return {"conv": name(mq.causal_conv1d_fn), "delta_rule": name(getattr(mq, "chunk_gated_delta_rule", None))}


def timed(fn, reps):
    sync()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = fn()
        sync()
        times.append(1000 * (time.perf_counter() - t0))
    times.sort()
    return {"ms_p50": times[len(times) // 2], "ms_max": times[-1], "reps": reps}, r


FRAME_MS = 42  # one game frame at "fastest" speed


def budget_ms(cadence_frames: int, rtt_ms: float) -> float:
    return cadence_frames * FRAME_MS - rtt_ms


def token_counts(enc, sw) -> dict:
    static = enc("Context:\n" + sw.static)
    dynamic = enc("\n" + sw.dynamic)
    q = [len(enc(question_piece(x))) for x in sw.questions]
    return {**sw.meta, "tokens_static": len(static), "tokens_dynamic": len(dynamic), "n_questions": len(q),
            "tokens_questions": sum(q), "tokens_total": len(static) + len(dynamic) + sum(q)}


def bench_sweep(R: Runner, sw, reps=5, sample=8, modes="abce"):
    static, dynamic, qs = R.sweep_ids(sw)
    ctx = static + dynamic
    res = token_counts(R.enc, sw)
    R.packed(ctx, sw.questions[:2]); R.questions(R.prefix(ctx), len(ctx), qs[:2])  # warm-up
    if "a" in modes:  # sample, scaled
        t, _ = timed(lambda: R.independent(ctx, qs[:sample], rows=sample), max(2, reps // 2))
        res["a_independent_est_ms"] = t["ms_p50"] * len(qs) / min(sample, len(qs))
    if "b" in modes:
        t, (_, n) = timed(lambda: R.packed(ctx, sw.questions), reps)
        res["b_packed_ms"] = t["ms_p50"]; res["b_tokens"] = n
    if "c" in modes or "e" in modes:  # state each sweep + questions
        tp, cache = timed(lambda: R.prefix(ctx), reps)
        tq, (_, chunk) = timed(lambda: R.questions(cache, len(ctx), qs), reps)
        res.update(c_prefix_ms=tp["ms_p50"], c_questions_ms=tq["ms_p50"], c_total_ms=tp["ms_p50"] + tq["ms_p50"], c_chunk=chunk,
                   cache_mb=R.cache_bytes(cache) / 2 ** 20)
    if "e" in modes:  # static cached across sweeps; per sweep = copy static cache + dynamic + questions
        static_cache = R.prefix(static)
        te, _ = timed(lambda: R.prefix(dynamic, copy.deepcopy(static_cache)), reps)
        res.update(e_dynamic_ms=te["ms_p50"], e_total_ms=te["ms_p50"] + tq["ms_p50"])
    return res


def best_ms(r):
    return min(v for k, v in r.items() if k in ("b_packed_ms", "c_total_ms", "e_total_ms"))


def sweeps(a):
    """(phase, terrain, sweep) for every requested combination."""
    ints = lambda s: [int(x) for x in s.split(",")]
    for ph in a.phases.split(","):
        for te in a.terrain.split(","):
            if a.layout == "per_unit":
                yield make_sweep(ph, te)
                continue
            for gs in ints(a.group_sizes):
                for fq in ints(a.fine_squads):
                    for fs in (ints(a.fine_sizes) if fq else [1]):
                        for qb in ints(a.q_budgets):
                            for mac in ints(a.macro):
                                yield make_grouped_sweep(ph, te, group_size=gs, q_budget=qb, macro=bool(mac),
                                                         compact=bool(a.compact), fine_squads=fq, fine_size=fs)


@torch.no_grad()
def check_equivalence(R: Runner, sw, n=6, atol=2e-2):
    """Shared-prefix answers must match independent rows (same tokens, same attention)."""
    static, dynamic, qs = R.sweep_ids(sw)
    ctx = static + dynamic
    qs = qs[:n]
    ind = R.independent(ctx, qs, rows=n)[0]
    sh, _ = R.questions(R.prefix(ctx), len(ctx), qs)
    diffs = {}
    if "choice" in sh:
        idx, prob = sh["choice"]
        ref = torch.softmax(ind["choice_logits"], -1)
        diffs["choice_max_abs"] = float((prob - ref[: len(idx)]).abs().max())
    if "point" in sh:
        idx, p = sh["point"]
        diffs["point_mu_max_abs"] = float((p["mu"] - ind["point"]["mu"][: len(idx)]).abs().max())
    diffs["ok"] = all(v < atol for k, v in diffs.items() if k != "ok")
    return diffs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--phases", default=",".join(PHASES))
    ap.add_argument("--terrain", default=",".join(TERRAIN))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--mem_gb", type=float, default=6.0)
    ap.add_argument("--check", action="store_true", help="only verify shared-prefix answers equal independent rows")
    ap.add_argument("--out", default=None)
    ap.add_argument("--layout", choices=["per_unit", "grouped"], default="per_unit")
    ap.add_argument("--group_sizes", default="1,4,8")
    ap.add_argument("--q_budgets", default="30")
    ap.add_argument("--fine_squads", default="0", help="most important squads split finely (mixed granularity)")
    ap.add_argument("--fine_sizes", default="1", help="group size inside those squads")
    ap.add_argument("--macro", default="0,1", help="0 = micro-only sweep, 1 = sweep with the macro check")
    ap.add_argument("--compact", type=int, default=1)
    ap.add_argument("--modes", default=None, help="subset of abce (default: abce per_unit, bc grouped)")
    ap.add_argument("--cadences", default="6,8,10,12", help="decision every N frames")
    ap.add_argument("--rtt", type=float, default=72.0, help="network round trip, ms")
    ap.add_argument("--no_conv_patch", action="store_true", help="keep transformers' reference causal conv")
    ap.add_argument("--tokens_only", action="store_true")
    ap.add_argument("--tokenizer", default=None, help="for --tokens_only (default: the model's base)")
    a = ap.parse_args()
    if a.layout == "grouped" and a.terrain == ",".join(TERRAIN):
        a.terrain = "regions"
    cad = [int(x) for x in a.cadences.split(",")]
    if a.tokens_only:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model)
        enc = lambda s: tok.encode(s, add_special_tokens=False)
        for sw in sweeps(a):
            print(json.dumps(token_counts(enc, sw)), flush=True)
        return
    if not a.no_conv_patch:
        from decider.engine import patch_conv  # decider's fusable causal conv (transformers' fallback is slow)
        patch_conv()
    model = PointModel(a.model, dtype=getattr(torch, a.dtype), grad_ckpt=False).to(a.device).eval()
    print("kernels:", kernel_report(), flush=True)
    R = Runner(model, a.mem_gb)
    if a.check:
        for ph in ("early", "mid"):
            print(ph, json.dumps(check_equivalence(R, make_sweep(ph, "regions"))), flush=True)
            print(ph, "grouped compact", json.dumps(check_equivalence(R, make_grouped_sweep(ph, group_size=1, macro=True))), flush=True)
        return
    modes = a.modes or ("abce" if a.layout == "per_unit" else "bc")
    rows = []
    for sw in sweeps(a):
        r = bench_sweep(R, sw, a.reps, modes=modes)
        r["fits_cadence"] = {c: best_ms(r) <= budget_ms(c, a.rtt) for c in cad}
        rows.append(r)
        print(json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps({"model": a.model, "device": torch.cuda.get_device_name() if torch.cuda.is_available() else a.device,
                                           "conv_patch": not a.no_conv_patch, "rtt_ms": a.rtt,
                                           "budget_ms": {c: budget_ms(c, a.rtt) for c in cad}, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()

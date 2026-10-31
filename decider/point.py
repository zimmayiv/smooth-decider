"""The `Point` question type: a declared coordinate frame, a prompt slot, and a
mixture-of-Gaussians head over (x, y).

Coordinates are always learned in the frame's normalized space u = (x - x0) / (x1 - x0),
v = (y - y0) / (y1 - y0), so a 0-1 field, a 105 m pitch and an 8192-pixel map are
the same problem for the head. The frame is stated in the prompt, so the model
knows what the numbers mean, and `Frame.denormalize` maps predictions back.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class Frame:
    x0: float
    x1: float
    y0: float
    y1: float
    units: str = "units"
    y_down: bool = False  # True when y grows downward (screens, Brood War maps)

    def __post_init__(self):
        if not (self.x1 > self.x0 and self.y1 > self.y0):
            raise ValueError(f"degenerate frame {self}")

    def normalize(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.x0) / (self.x1 - self.x0), (y - self.y0) / (self.y1 - self.y0)

    def denormalize(self, u: float, v: float) -> tuple[float, float]:
        return self.x0 + u * (self.x1 - self.x0), self.y0 + v * (self.y1 - self.y0)

    def describe(self) -> str:
        fmt = lambda a: f"{a:g}"
        down = "y increases downward" if self.y_down else "y increases upward"
        return (f"x from {fmt(self.x0)} to {fmt(self.x1)}, y from {fmt(self.y0)} to {fmt(self.y1)} "
                f"({self.units}; {down})")


@dataclass
class PointQ:
    text: str
    frame: Frame
    target: tuple | None = None  # (x, y) in frame units; None at inference


@dataclass
class ChoiceQ:
    """A decider-style multiple-choice question that can share a row with Point questions."""
    text: str
    options: list
    gold: int = 0


@dataclass
class PointExample:
    context: str
    qs: list = field(default_factory=list)  # PointQ and/or ChoiceQ, answered in order in one pass
    task: str = "point"
    meta: dict = field(default_factory=dict)


def build_point(example: PointExample, tok, max_ctx_tokens: int = 4096) -> dict:
    """Token ids and answer slots for a row of Point (and optionally Choice) questions, in decider's plain
    state-first layout. Choice questions use decider's own option format, in the given option order."""
    from decider.prompt import LETTERS, NARROW

    ids = tok.encode("Context:\n" + example.context, add_special_tokens=False)[:max_ctx_tokens]
    slots, golds, nopts, p_slots, targets, frames = [], [], [], [], [], []
    multi = len(example.qs) > 1
    for k, q in enumerate(example.qs):
        n = f" {k + 1}" if multi else ""
        if isinstance(q, ChoiceQ):
            if len(q.options) > NARROW:
                raise ValueError(f"mixed rows support up to {NARROW} options, got {len(q.options)}")
            piece = (f"\n\nQuestion{n}: {q.text}\nOptions:" + "".join(f"\n({LETTERS[j]}) {o}" for j, o in enumerate(q.options))
                     + f"\nAnswer{n}: (")
            ids = ids + tok.encode(piece, add_special_tokens=False)
            slots.append(len(ids) - 1); golds.append(q.gold); nopts.append(len(q.options))
        else:
            piece = (f"\n\nQuestion{n}: {q.text}\nCoordinate frame: {q.frame.describe()}."
                     f"\nAnswer{n} as a point: (")
            ids = ids + tok.encode(piece, add_special_tokens=False)
            p_slots.append(len(ids) - 1)
            targets.append(q.frame.normalize(*q.target) if q.target is not None else (float("nan"), float("nan")))
            frames.append(q.frame)
    return {"ids": ids, "point_slots": p_slots, "point_targets": targets, "frames": frames, "task": example.task,
            "slots": slots, "golds": golds, "nopts": nopts}


def collate(items: list, pad_id: int, bucket: int = 64) -> dict:
    """Batch rows that may hold choice questions (decider's `slots/golds/nopts`) and/or
    point questions (`point_slots/point_targets`). Right-pads like decider."""
    T = max(len(it["ids"]) for it in items)
    T = ((T + bucket - 1) // bucket) * bucket
    B = len(items)
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.long)
    c_idx, c_b, golds, nopts, p_idx, p_b, p_t, tasks = [], [], [], [], [], [], [], []
    for b, it in enumerate(items):
        n = len(it["ids"])
        input_ids[b, :n] = torch.tensor(it["ids"])
        attn[b, :n] = 1
        for k, s in enumerate(it.get("slots", [])):
            c_idx.append(s); c_b.append(b); golds.append(it["golds"][k]); nopts.append(it["nopts"][k])
        for s, t in zip(it.get("point_slots", []), it.get("point_targets", [])):
            p_idx.append(s); p_b.append(b); p_t.append(t); tasks.append(it.get("task", ""))
    lt = lambda x: torch.tensor(x, dtype=torch.long)
    return dict(input_ids=input_ids, attention_mask=attn,
                slot_idx=lt(c_idx), slot_batch=lt(c_b), golds=lt(golds), nopts=lt(nopts),
                point_slot_idx=lt(p_idx), point_slot_batch=lt(p_b),
                point_targets=torch.tensor(p_t, dtype=torch.float32).reshape(-1, 2), point_tasks=tasks)


class PointHead(nn.Module):
    """Answer-slot hidden state -> mixture of K bivariate Gaussians over normalized (u, v)."""

    def __init__(self, hidden: int, k: int = 8, inner: int = 1024, min_sigma: float = 1e-3):
        super().__init__()
        self.k, self.min_sigma = k, min_sigma
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, inner)
        self.fc2 = nn.Linear(inner, k * 6)
        nn.init.normal_(self.fc2.weight, std=1e-3)
        with torch.no_grad():  # start as a broad mixture spread over the frame
            g = math.ceil(math.sqrt(k))
            centres = [((i % g + 0.5) / g, (i // g + 0.5) / g) for i in range(k)]
            b = torch.zeros(k, 6)
            b[:, 1:3] = torch.logit(torch.tensor(centres).clamp(0.05, 0.95))
            b[:, 3:5] = math.log(0.25)
            self.fc2.bias.copy_(b.reshape(-1))

    def forward(self, h: torch.Tensor) -> dict:
        z = self.fc2(F.silu(self.fc1(self.norm(h.float())))).view(-1, self.k, 6)
        return {
            "logit_pi": z[..., 0],
            "mu": 1.2 * torch.sigmoid(z[..., 1:3]) - 0.1,  # a little room outside [0, 1]
            "sigma": F.softplus(z[..., 3:5]) + self.min_sigma,
            "rho": 0.95 * torch.tanh(z[..., 5]),
        }


def component_logpdf(p: dict, uv: torch.Tensor) -> torch.Tensor:
    """log N(uv | component k) for each row and component: [N, K]."""
    d = (uv[:, None, :] - p["mu"]) / p["sigma"]
    rho = p["rho"]
    one_m = 1 - rho ** 2
    quad = (d[..., 0] ** 2 + d[..., 1] ** 2 - 2 * rho * d[..., 0] * d[..., 1]) / one_m
    return -math.log(2 * math.pi) - p["sigma"].log().sum(-1) - 0.5 * one_m.log() - 0.5 * quad


def mixture_logpdf(p: dict, uv: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(F.log_softmax(p["logit_pi"], -1) + component_logpdf(p, uv), -1)


def point_nll(p: dict, uv: torch.Tensor) -> torch.Tensor:
    """Mean negative log-likelihood of the targets in normalized space (per row: -log density)."""
    return -mixture_logpdf(p, uv).mean()


@torch.no_grad()
def sample(p: dict, n: int, generator=None) -> torch.Tensor:
    """n samples per row: [N, n, 2]."""
    N, K = p["logit_pi"].shape
    comp = torch.multinomial(F.softmax(p["logit_pi"], -1), n, replacement=True, generator=generator)  # [N, n]
    gather = lambda t: torch.gather(t, 1, comp[..., None].expand(-1, -1, t.shape[-1]) if t.dim() == 3 else comp)
    mu, sig, rho = gather(p["mu"]), gather(p["sigma"]), gather(p["rho"])
    e = torch.randn(N, n, 2, generator=generator, device=mu.device)
    x = e[..., 0]
    y = rho * e[..., 0] + torch.sqrt(1 - rho ** 2) * e[..., 1]
    return mu + sig * torch.stack([x, y], -1)


@torch.no_grad()
def point_estimate(p: dict, n: int = 256) -> torch.Tensor:
    """A single best point per row: the sample (or component mean) with the highest mixture density."""
    cands = torch.cat([p["mu"], sample(p, n)], 1)  # [N, K + n, 2]
    N, M, _ = cands.shape
    flat = {k: v.repeat_interleave(M, 0) for k, v in p.items()}
    dens = mixture_logpdf(flat, cands.reshape(-1, 2)).view(N, M)
    return cands[torch.arange(N), dens.argmax(1)]


@torch.no_grad()
def hpd_level(p: dict, uv: torch.Tensor, n: int = 512, generator=None) -> torch.Tensor:
    """Smallest highest-density region of the prediction that contains the target, as a probability
    (0 = at the mode, 1 = outside everything). Uniform on [0, 1] when the mixture is calibrated."""
    s = sample(p, n, generator)  # [N, n, 2]
    N = s.shape[0]
    flat = {k: v.repeat_interleave(n, 0) for k, v in p.items()}
    dens_s = mixture_logpdf(flat, s.reshape(-1, 2)).view(N, n)
    dens_t = mixture_logpdf(p, uv)
    return (dens_s > dens_t[:, None]).float().mean(1)

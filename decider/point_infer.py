"""Point answers from a fine-tuned checkpoint, one forward pass per batch of rows.

    from decider.point_infer import PointDecider
    d = PointDecider("runs/point/model")            # adapter folder or merged folder with point_head.pt
    d.point("Map: ...\\n- Alpha (ally) at (120, 340)\\n- Bravo (enemy) at (400, 90)",
            [{"question": "Move Alpha halfway to Bravo.",
              "frame": {"x0": 0, "x1": 512, "y0": 0, "y1": 512, "units": "pixels", "y_down": True}}])
    # -> [{"type": "point", "x": 260.1, "y": 215.3, "radius90": 6.2,
    #      "components": [{"weight": 0.97, "x": 260.1, "y": 215.3, "sx": 2.9, "sy": 3.1, "rho": 0.02}, ...]}]

Coordinates are in each question's declared frame. `radius90` is the radius around the best point that
holds 90% of the predicted probability (from samples), a single-number confidence. Choice questions keep
going through decider's own `Decider`; the two can share a checkpoint.
"""

from __future__ import annotations

import torch

from decider.point import Frame, PointExample, PointQ, build_point, collate, point_estimate, sample
from decider.point_model import PointModel


class PointDecider:
    def __init__(self, path: str, device: str | None = None, dtype=None):
        device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        dtype = dtype or (torch.bfloat16 if device == "cuda" else torch.float32)
        self.model = PointModel(path, dtype=dtype, grad_ckpt=False).to(device).eval()
        self.device = device

    @torch.no_grad()
    def point(self, state: str, questions: list, max_state_tokens: int = 4096, samples: int = 512) -> list:
        """questions: [{"question": str, "frame": {x0, x1, y0, y1, units?, y_down?}}] about one state.
        All questions are answered in one pass (later questions can read earlier question texts)."""
        qs = [PointQ(q["question"], Frame(**q["frame"])) for q in questions]
        item = build_point(PointExample(state, qs), self.model.tok, max_ctx_tokens=max_state_tokens)
        b = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in collate([item], self.model.tok.pad_token_id).items()}
        p = self.model(b)["point"]
        best = point_estimate(p)
        draws = sample(p, samples)
        out = []
        for i, q in enumerate(qs):
            f = q.frame
            w = torch.tensor([f.x1 - f.x0, f.y1 - f.y0], device=best.device)
            dist = ((draws[i] - best[i]) * w).norm(dim=-1)
            x, y = f.denormalize(*best[i].tolist())
            pi = torch.softmax(p["logit_pi"][i], -1)
            comps = []
            for k in torch.argsort(pi, descending=True).tolist():
                if pi[k] < 0.01:
                    continue
                cx, cy = f.denormalize(*p["mu"][i, k].tolist())
                comps.append({"weight": round(float(pi[k]), 4), "x": cx, "y": cy,
                              "sx": float(p["sigma"][i, k, 0]) * (f.x1 - f.x0), "sy": float(p["sigma"][i, k, 1]) * (f.y1 - f.y0),
                              "rho": float(p["rho"][i, k])})
            out.append({"type": "point", "x": x, "y": y, "radius90": float(torch.quantile(dist.float(), 0.9)),
                        "components": comps})
        return out

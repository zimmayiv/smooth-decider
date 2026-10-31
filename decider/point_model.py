"""decider plus a Point head: one backbone pass answers Choice/Score/Noul questions with
decider's own letter readout and Point questions with the mixture head."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from decider.model import DecisionModel, cap_logits
from decider.point import PointHead
from decider.prompt import MAX_OPTIONS

# Linear layers LoRA adapts in Qwen3.5: full attention, GatedDeltaNet and MLP projections.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
                "out_proj", "gate_proj", "up_proj", "down_proj"]


class PointModel(nn.Module):
    def __init__(self, path: str, dtype=torch.bfloat16, grad_ckpt: bool = True, k: int = 8, trainable_adapter: bool = False):
        super().__init__()
        p = Path(path)
        base = path
        adapter = p / "adapter_config.json"
        if adapter.exists():  # an adapter folder: load its base model, then the adapter
            base = json.loads((p / "point.json").read_text())["base"]
        self.dm = DecisionModel(base, dtype=dtype, grad_ckpt=grad_ckpt)
        if adapter.exists():
            from peft import PeftModel
            if trainable_adapter:  # continue training the same adapter
                self.dm.lm = PeftModel.from_pretrained(self.dm.lm, str(p), is_trainable=True)
                self.dm.lm.enable_input_require_grads()
            else:  # merged in memory: decider's own code paths expect a plain model, and it is faster
                self.dm.lm = PeftModel.from_pretrained(self.dm.lm, str(p)).merge_and_unload()
        self.point_head = PointHead(self.dm.lm.config.hidden_size, k=k)
        head_file = p / "point_head.pt"
        if head_file.exists():
            self.point_head.load_state_dict(torch.load(head_file, map_location="cpu"))

    @property
    def tok(self):
        return self.dm.tok

    def add_lora(self, r: int = 64, alpha: int = 128, dropout: float = 0.0):
        """Wrap the backbone in a LoRA (decider v2.1's stage recipe); only adapters and the head train."""
        from peft import LoraConfig, get_peft_model

        names = {n.split(".")[-1] for n, m in self.dm.lm.named_modules() if isinstance(m, nn.Linear)}
        targets = [t for t in LORA_TARGETS if t in names]
        cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=targets, bias="none")
        self.dm.lm = get_peft_model(self.dm.lm, cfg)
        self.dm.lm.enable_input_require_grads()  # needed for gradient checkpointing with a frozen embedding
        return targets

    def merge_lora(self):
        self.dm.lm = self.dm.lm.merge_and_unload()

    def backbone(self):
        lm = self.dm.lm
        return lm.get_base_model().model if hasattr(lm, "get_base_model") else lm.model

    def lm_head_weight(self):
        lm = self.dm.lm
        base = lm.get_base_model() if hasattr(lm, "get_base_model") else lm
        return base.lm_head.weight

    def forward(self, batch: dict) -> dict:
        h = self.backbone()(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
        out = {}
        if batch["slot_idx"].numel():
            hs = h[batch["slot_batch"], batch["slot_idx"]]
            logits = cap_logits(self.dm, F.linear(hs, self.lm_head_weight()[self.dm.letters]).float())
            ar = torch.arange(MAX_OPTIONS, device=logits.device)[None, :]
            out["choice_logits"] = logits.masked_fill(ar >= batch["nopts"][:, None], float("-inf"))
        if batch["point_slot_idx"].numel():
            out["point"] = self.point_head(h[batch["point_slot_batch"], batch["point_slot_idx"]])
        return out

    @torch.no_grad()
    def interpolate_(self, other: "PointModel", alpha: float):
        """In place: every weight becomes (1 - alpha) * self + alpha * other. Both models must share the base
        and be loaded merged (the default for adapter folders); then the backbone is exactly
        W0 + (1 - alpha) * delta_self + alpha * delta_other. The point heads are averaged weight by weight."""
        mine, theirs = dict(self.named_parameters()), dict(other.named_parameters())
        if mine.keys() != theirs.keys():
            raise ValueError("models have different parameters; load both as merged adapter folders")
        for n, w in mine.items():
            w.mul_(1 - alpha).add_(theirs[n].to(w.device, w.dtype), alpha=alpha)
        return self

    def save(self, out_dir: str, merged: bool = False, base: str | None = None):
        """Save the trained parts. Default: the LoRA adapter + point_head.pt + point.json naming the
        base model (small; PointModel(out_dir) reloads it). merged=True instead writes a full
        decider-format folder with the LoRA merged in (for serving); merging is final."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        lm = self.dm.lm
        if merged:
            if hasattr(lm, "merge_and_unload"):
                lm = lm.merge_and_unload()
            lm.save_pretrained(out)
            self.dm.tok.save_pretrained(out)
        else:
            if not hasattr(lm, "peft_config"):
                raise ValueError("no adapter to save; use merged=True")
            lm.save_pretrained(out)  # peft writes adapter_config.json + adapter weights only
            (out / "point.json").write_text(json.dumps({"base": base}))
        torch.save(self.point_head.state_dict(), out / "point_head.pt")

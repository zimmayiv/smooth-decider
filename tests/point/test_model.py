"""PointModel on a tiny random decider-shaped checkpoint (decider's real tokenizer)."""
import random
from pathlib import Path

import pytest
import torch

from decider.point_model import PointModel
from decider.point import Frame, PointExample, PointQ, build_point, collate

from decider.data.core import Example, Q
from decider.prompt import build

TINY = Path(__file__).resolve().parents[2] / ".cache" / "tiny_decider"
pytestmark = pytest.mark.skipif(not TINY.exists(), reason="tiny decider checkpoint not built")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def model():
    torch.manual_seed(0)
    m = PointModel(str(TINY), dtype=torch.float32, grad_ckpt=False).to(DEV).eval()
    return m


def rows(tok):
    c = build(Example("The sky is blue.", [Q("What colour is the sky?", ["red", "blue", "green"], 1)], "toy"), tok,
              random.Random(0), max_options=255)
    p = build_point(PointExample("Unit A at (10, 20). Enemy at (60, 40).",
                                 [PointQ("Move A halfway to the enemy.", Frame(0, 100, 0, 50, "tiles"), (35, 30))]), tok)
    return c, p


def to(b):
    return {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in b.items()}


def test_choice_logits_equal_deciders_own_readout(model):
    c, p = rows(model.tok)
    b = to(collate([c, p], model.tok.pad_token_id))
    with torch.no_grad():
        ours = model(b)["choice_logits"]
        ref = model.dm.slot_logits(b["input_ids"], b["attention_mask"], b["slot_idx"], b["slot_batch"], b["nopts"])
    assert torch.allclose(ours, ref, atol=1e-5)


def test_fresh_lora_changes_nothing_and_trains_only_adapters_and_head(model):
    c, p = rows(model.tok)
    b = to(collate([c, p], model.tok.pad_token_id))
    with torch.no_grad():
        before = model(b)
    targets = model.add_lora(r=4, alpha=8)
    model.to(DEV).eval()
    assert "in_proj_qkv" in targets and "q_proj" in targets and "gate_proj" in targets
    with torch.no_grad():
        after = model(b)
    assert torch.allclose(before["choice_logits"], after["choice_logits"], atol=1e-5)
    assert torch.allclose(before["point"]["mu"], after["point"]["mu"], atol=1e-5)
    trainable = {n.split(".")[0] if n.startswith("point_head") else "lora"
                 for n, q in model.named_parameters() if q.requires_grad and ("lora_" in n or n.startswith("point_head"))}
    frozen_base = [n for n, q in model.named_parameters() if q.requires_grad and "lora_" not in n and not n.startswith("point_head")]
    assert trainable == {"point_head", "lora"} and frozen_base == []


def _nontrivial_lora(model):
    model.add_lora(r=4, alpha=8)
    with torch.no_grad():  # make the adapter non-trivial
        for n, q in model.named_parameters():
            if "lora_B" in n:
                q.normal_(std=0.02)
    model.to(DEV).eval()


def test_adapter_save_reloads_identically_and_is_small(model, tmp_path):
    c, p = rows(model.tok)
    b = to(collate([c, p], model.tok.pad_token_id))
    _nontrivial_lora(model)
    with torch.no_grad():
        ref = model(b)
    model.save(str(tmp_path / "ad"), base=str(TINY))
    files = {f.name for f in (tmp_path / "ad").iterdir()}
    assert {"adapter_config.json", "point_head.pt", "point.json"} <= files and "model.safetensors" not in files
    re = PointModel(str(tmp_path / "ad"), dtype=torch.float32, grad_ckpt=False).to(DEV).eval()
    with torch.no_grad():
        got = re(b)
    assert torch.allclose(ref["choice_logits"], got["choice_logits"], atol=1e-4)
    assert torch.allclose(ref["point"]["mu"], got["point"]["mu"], atol=1e-4)
    with torch.no_grad():  # decider's own readout (used by its regression eval) must work on the reloaded model
        own = re.dm.slot_logits(b["input_ids"], b["attention_mask"], b["slot_idx"], b["slot_batch"], b["nopts"])
    assert torch.allclose(own, ref["choice_logits"], atol=1e-4)


def test_merged_save_reloads_identically(model, tmp_path):
    import shutil
    c, p = rows(model.tok)
    b = to(collate([c, p], model.tok.pad_token_id))
    _nontrivial_lora(model)
    with torch.no_grad():
        ref = model(b)
    model.save(str(tmp_path / "m"), merged=True)
    for f in ("decider_config.json", "chat_template.jinja"):
        if (TINY / f).exists():
            shutil.copy(TINY / f, tmp_path / "m" / f)
    re = PointModel(str(tmp_path / "m"), dtype=torch.float32, grad_ckpt=False).to(DEV).eval()
    with torch.no_grad():
        got = re(b)
    assert torch.allclose(ref["choice_logits"], got["choice_logits"], atol=1e-4)
    assert torch.allclose(ref["point"]["mu"], got["point"]["mu"], atol=1e-4)


def test_interpolating_merged_adapters_averages_the_fine_tunes(tmp_path):
    torch.manual_seed(0)
    c, p = rows(PointModel(str(TINY), dtype=torch.float32, grad_ckpt=False).tok)
    paths = []
    for seed in (1, 2):
        m = PointModel(str(TINY), dtype=torch.float32, grad_ckpt=False)
        torch.manual_seed(seed)
        _nontrivial_lora(m)
        with torch.no_grad():
            for n, q in m.named_parameters():
                if n.startswith("point_head"):
                    q.add_(0.01 * torch.randn_like(q))
        m.save(str(tmp_path / f"a{seed}"), base=str(TINY))
        paths.append(str(tmp_path / f"a{seed}"))
    load = lambda path: PointModel(path, dtype=torch.float32, grad_ckpt=False).to(DEV).eval()
    m1, m2, base = load(paths[0]), load(paths[1]), load(str(TINY))
    w = lambda m: dict(m.dm.lm.named_parameters())
    soup = load(paths[0]).interpolate_(m2, 0.5)
    for n, q in w(soup).items():  # merged soup == W0 + (delta1 + delta2) / 2
        exp = w(base)[n] + 0.5 * (w(m1)[n] - w(base)[n]) + 0.5 * (w(m2)[n] - w(base)[n])
        assert torch.allclose(q, exp, atol=1e-6), n
    b = to(collate([c, p], m1.tok.pad_token_id))
    with torch.no_grad():
        assert torch.allclose(load(paths[0]).interpolate_(m2, 0.0)(b)["point"]["mu"], m1(b)["point"]["mu"], atol=1e-6)
        assert torch.allclose(load(paths[0]).interpolate_(m2, 1.0)(b)["point"]["mu"], m2(b)["point"]["mu"], atol=1e-5)


def test_point_decider_returns_points_in_the_declared_frame(tmp_path):
    from decider.point_infer import PointDecider
    m = PointModel(str(TINY), dtype=torch.float32, grad_ckpt=False)
    _nontrivial_lora(m)
    m.save(str(tmp_path / "ad"), base=str(TINY))
    d = PointDecider(str(tmp_path / "ad"), device=DEV, dtype=torch.float32)
    frame = {"x0": -50, "x1": 50, "y0": 0, "y1": 8192, "units": "pixels", "y_down": True}
    out = d.point("Alpha at (0, 100). Bravo at (40, 7000).",
                  [{"question": "Move Alpha to Bravo.", "frame": frame}, {"question": "And back.", "frame": frame}])
    assert len(out) == 2
    for o in out:
        assert o["type"] == "point" and -60 <= o["x"] <= 60 and -900 <= o["y"] <= 9100
        assert o["radius90"] > 0 and abs(sum(c["weight"] for c in o["components"]) - 1) < 0.1

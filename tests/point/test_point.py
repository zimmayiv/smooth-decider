import math

import pytest
import torch

from decider.point import (Frame, PointExample, PointHead, PointQ, build_point, collate, hpd_level,
                              mixture_logpdf, point_estimate, point_nll, sample)


def test_frame_roundtrip_and_description():
    f = Frame(0, 8192, 0, 8192, "pixels", y_down=True)
    assert f.normalize(4096, 2048) == (0.5, 0.25)
    assert f.denormalize(0.5, 0.25) == (4096, 2048)
    assert "x from 0 to 8192" in f.describe() and "downward" in f.describe()
    with pytest.raises(ValueError):
        Frame(1, 1, 0, 1)


def fixed_params(mu, sigma, rho=0.0, pi=None):
    k = len(mu)
    return {"logit_pi": torch.log(torch.tensor([pi or [1 / k] * k])),
            "mu": torch.tensor([mu], dtype=torch.float32),
            "sigma": torch.tensor([sigma], dtype=torch.float32),
            "rho": torch.tensor([[rho] * k], dtype=torch.float32)}


def test_mixture_density_integrates_to_one():
    p = fixed_params([[0.3, 0.4], [0.7, 0.6]], [[0.05, 0.08], [0.1, 0.04]], rho=0.5)
    g = torch.linspace(-0.5, 1.5, 801)
    uu, vv = torch.meshgrid(g, g, indexing="ij")
    pts = torch.stack([uu.reshape(-1), vv.reshape(-1)], 1)
    rep = {k: v.expand(len(pts), *v.shape[1:]) for k, v in p.items()}
    mass = mixture_logpdf(rep, pts).exp().sum() * (g[1] - g[0]) ** 2
    assert abs(float(mass) - 1) < 1e-3


def test_hpd_levels_are_uniform_when_targets_come_from_the_prediction():
    torch.manual_seed(0)
    p = fixed_params([[0.2, 0.3], [0.8, 0.7], [0.5, 0.5]], [[0.05, 0.1], [0.08, 0.03], [0.2, 0.2]], rho=-0.3)
    n = 3000
    targets = sample(p, n)[0]
    rep = {k: v.expand(n, *v.shape[1:]) for k, v in p.items()}
    levels = hpd_level(rep, targets, n=1024)
    for q in (0.1, 0.5, 0.9):  # coverage at level q should be about q
        assert abs(float((levels <= q).float().mean()) - q) < 0.04


def test_point_estimate_finds_the_dominant_mode():
    p = fixed_params([[0.2, 0.2], [0.8, 0.9]], [[0.02, 0.02], [0.2, 0.2]], pi=[0.8, 0.2])
    est = point_estimate(p)
    assert torch.allclose(est, torch.tensor([[0.2, 0.2]]), atol=0.02)


def test_head_starts_broad_and_learns_a_bimodal_target():
    torch.manual_seed(0)
    head = PointHead(hidden=16, k=4, inner=32)
    h = torch.randn(512, 16)
    side = (h[:, 0] > 0).float()  # the hidden state says which of two spots is right
    targets = torch.stack([0.2 + 0.6 * side, 0.5 + 0.0 * side], 1) + 0.01 * torch.randn(512, 2)
    start = float(point_nll(head(h), targets).detach())
    assert abs(start) < 5  # broad, finite starting mixture
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    for _ in range(400):
        opt.zero_grad(); loss = point_nll(head(h), targets); loss.backward(); opt.step()
    assert float(loss) < start - 4
    est = point_estimate(head(h))
    assert float((est - targets).norm(dim=1).median()) < 0.05


def test_build_point_slot_is_the_last_token_and_targets_are_normalized():
    class Tok:
        def encode(self, s, add_special_tokens=False):
            return [ord(c) for c in s]
    f = Frame(0, 100, 0, 50, "metres")
    ex = PointExample("a at (10, 20)", [PointQ("Where?", f, (25, 10)), PointQ("And?", f, (100, 50))])
    it = build_point(ex, Tok())
    assert it["point_slots"][1] == len(it["ids"]) - 1
    assert it["ids"][it["point_slots"][0]] == ord("(")
    assert it["point_targets"] == [(0.25, 0.2), (1.0, 1.0)]


def test_collate_mixes_choice_and_point_rows():
    choice_row = {"ids": [1, 2, 3], "slots": [2], "golds": [1], "nopts": [3]}
    point_row = {"ids": [4, 5, 6, 7, 8], "point_slots": [2, 4], "point_targets": [(0.1, 0.2), (0.3, 0.4)], "task": "t"}
    b = collate([choice_row, point_row], pad_id=0)
    assert b["input_ids"].shape == (2, 64)
    assert b["slot_batch"].tolist() == [0] and b["golds"].tolist() == [1]
    assert b["point_slot_batch"].tolist() == [1, 1] and b["point_slot_idx"].tolist() == [2, 4]
    assert torch.allclose(b["point_targets"], torch.tensor([[0.1, 0.2], [0.3, 0.4]]))


def test_fork_cache_repeats_every_layer_type():
    from pathlib import Path
    import pytest
    tiny = Path(__file__).resolve().parents[2] / ".cache" / "tiny_decider"
    if not tiny.exists():
        pytest.skip("tiny decider checkpoint not built")
    from decider.point_latency import _tensors, fork_cache
    from decider.point_model import PointModel
    m = PointModel(str(tiny), dtype=torch.float32, grad_ckpt=False).eval()
    with torch.no_grad():
        c = m.backbone()(input_ids=torch.randint(0, 1000, (1, 9)), use_cache=True).past_key_values
    f = fork_cache(c, 5)
    assert all(t.shape[0] == 5 for t in _tensors(f)) and all(t.shape[0] == 1 for t in _tensors(c))
    kinds = {type(L).__name__ for L in f.layers}
    assert {"LinearAttentionLayer", "DynamicLayer"} <= kinds

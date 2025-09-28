import torch

from qham import hamilton, left_broadcast, normalize


def test_left_broadcast_orientation():
    q = torch.tensor([1.0, 0.0, 0.0, 0.0])  # identity
    W = torch.randn(7, 4)
    Y = left_broadcast(q, W, backend="auto")
    assert torch.allclose(Y, W)


def test_normalize_finite_and_unit_on_nonzero():
    q = torch.randn(16, 4, dtype=torch.float32)
    z = torch.zeros(4)
    out = normalize(torch.cat([q, z[None, :]], dim=0), eps=1e-6)
    assert torch.isfinite(out).all()
    nonzero = torch.norm(out[:-1], dim=-1)
    assert torch.allclose(nonzero, torch.ones_like(nonzero), rtol=1e-5, atol=1e-6)


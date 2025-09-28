from __future__ import annotations

from typing import Optional

import torch

from ._backend import select_backend


def _ensure_last_dim4(t: torch.Tensor, name: str) -> None:
    if t.shape[-1] != 4:
        raise ValueError(f"qham expects last dimension == 4 for {name}, got shape {tuple(t.shape)}.")


def _validate_inputs(a: torch.Tensor, b: Optional[torch.Tensor] = None) -> None:
    _ensure_last_dim4(a, "a")
    if b is not None:
        _ensure_last_dim4(b, "b")
        if a.device != b.device:
            raise RuntimeError(
                f"Device mismatch: {a.device} vs {b.device}. Place inputs on the same device."
            )
        if a.dtype != b.dtype:
            raise TypeError(
                f"Dtype mismatch: {a.dtype} vs {b.dtype}. Cast to a common dtype explicitly."
            )


def hamilton(a: torch.Tensor, b: torch.Tensor, *, backend: str = "auto") -> torch.Tensor:
    """
    Hamilton product a ⊗ b with full broadcasting.

    Both inputs must be real tensors whose last dimension is 4, representing [w, x, y, z].

    Device & dtype policy: inputs must be on the same device and same dtype.
    """
    chosen = select_backend(backend)
    _validate_inputs(a, b)

    # Torch-only path for now
    if chosen == "torch":
        w1, x1, y1, z1 = a.unbind(dim=-1)
        w2, x2, y2, z2 = b.unbind(dim=-1)

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack((w, x, y, z), dim=-1)

    # Future backends (do not enable yet)
    # if chosen == "cuda":
    #     # TODO: call CUDA kernel
    #     raise NotImplementedError("CUDA backend not enabled yet.")
    # if chosen == "triton":
    #     # TODO: call Triton kernel
    #     raise NotImplementedError("Triton backend not enabled yet.")

    raise AssertionError(f"Unexpected backend selection: {chosen}")


def left_broadcast(q: torch.Tensor, W: torch.Tensor, *, backend: str = "auto") -> torch.Tensor:
    """
    Apply left multiplication by q to every row of W.

    Computes Y[i] = q ⊗ W[i], for i in [0, Nq).

    Shapes:
      q:  (4,)
      W:  (Nq, 4)
      out:(Nq, 4)
    """
    # Validate orientation explicitly
    _ensure_last_dim4(q, "q")
    _ensure_last_dim4(W, "W")

    # Torch-only path for now; rely on broadcasting (q:(4,), W:(Nq,4)) → (Nq,4)
    return hamilton(q, W, backend=backend)


def conj(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate: [w, x, y, z] → [w, -x, -y, -z]."""
    _ensure_last_dim4(q, "q")
    wxyz = q
    w, x, y, z = wxyz.unbind(dim=-1)
    return torch.stack((w, -x, -y, -z), dim=-1)


def normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Unit-normalize along the last axis using:
        q / clamp_min(||q||, eps)
    where ||q|| = sqrt(w^2 + x^2 + y^2 + z^2).
    Using clamp_min avoids division by zero in low precision.
    """
    _ensure_last_dim4(q, "q")
    norm = q.norm(dim=-1, keepdim=True).clamp_min(eps)
    return q / norm


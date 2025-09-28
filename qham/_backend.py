from __future__ import annotations

import os
from typing import Literal

# For now, only "torch" is implemented.
_SUPPORTED = ("torch",)
# Future plan (do not enable yet):
# _SUPPORTED = ("torch", "triton", "cuda")


def select_backend(backend: str = "auto") -> Literal["torch"]:
    """
    Resolve the execution backend.

    Precedence (Torch-only for now):
    1) Explicit kwarg wins if not "auto". Currently only "torch" is supported.
       Passing "triton" or "cuda" raises NotImplementedError (for now).
    2) If backend == "auto", consult env QHAM_BACKEND.
       - "torch" -> "torch"
       - "triton" or "cuda" -> NotImplementedError (for now)
    3) Otherwise default to "torch".

    # TODO (when enabling more backends):
    # If backend == "auto" and no env override:
    #   Prefer fastest available: CUDA -> Triton -> Torch.
    #   Example sketch (commented out):
    #
    #   import torch
    #   if torch.cuda.is_available():
    #       return "cuda"
    #   try:
    #       import triton  # noqa: F401
    #       return "triton"
    #   except Exception:
    #       pass
    #   return "torch"
    """
    if backend != "auto":
        if backend == "torch":
            return "torch"
        # Keep explicit, friendly messages for future users
        if backend in {"triton", "cuda"}:
            raise NotImplementedError(
                f"Backend '{backend}' not yet implemented. Use backend='torch' for now."
            )
        raise ValueError(f"Unknown backend '{backend}'. Supported: {_SUPPORTED} or 'auto'.")

    # backend == "auto"
    env = os.getenv("QHAM_BACKEND", "").strip().lower()
    if env == "torch":
        return "torch"
    if env in {"triton", "cuda"}:
        raise NotImplementedError(
            f"QHAM_BACKEND={env!r} requested but not implemented yet. Use 'torch' for now."
        )

    # Default
    return "torch"


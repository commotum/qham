# qham — Unified Hamilton‑product API (Torch ↔ Triton ↔ CUDA)

`qham` is a small library that does one thing well: fast quaternion math for machine learning, with a single API that runs on:

- Pure PyTorch (reference path; autograd works out of the box)
- Triton (GPU fast path; same API)
- CUDA/C++ (maximum performance; same API)

You write the same Python code and pick a backend when you need more speed.

---

## Install (dev)

```bash
pip install qham           # once published
# or from source while developing:
# uv sync && uv build
```

---

## Quick start

```python
import torch
from qham import hamilton, left_broadcast, normalize, conj

# Pairwise product (broadcastable tensors ending in 4)
a = torch.tensor([1., 2., 3., 4.])      # [w, x, y, z]
b = torch.tensor([0.5, -1., 0.25, 2.])
c = hamilton(a, b)                       # shape (4,)

# One quaternion across many rows (RGB -> embedding)
q = torch.tensor([0., 0.2, -0.1, 0.7], device="cuda", dtype=torch.float16)  # pure imaginary RGB quaternion
embed_dim = 4096
Nq = embed_dim // 4
W = torch.randn(Nq, 4, device="cuda", dtype=torch.float16)                 # learned weight quaternions
Y = left_broadcast(q, W, backend="auto")  # (Nq, 4)
embed = Y.flatten()                       # (embed_dim,)
```

> Backend selection
>
> - Passing `backend="torch"` uses the PyTorch reference path.
> - Passing `backend="auto"` consults the environment:
>   - If `QHAM_BACKEND=torch`, selects `"torch"`.
>   - `QHAM_BACKEND=triton|cuda` are recognized but not implemented yet (will raise a helpful error).
> - If no env override is set, `backend="auto"` defaults to `"torch"`.
>
> <!-- FUTURE (uncomment when implemented):
> When Triton/CUDA are enabled, `backend="auto"` with no env override will prefer the fastest available:
> CUDA → Triton → Torch.
> -->

---

## API

```python
hamilton(a, b, *, backend="auto") -> Tensor[... , 4]
left_broadcast(q, W, *, backend="auto") -> Tensor[Nq, 4]
conj(q) -> Tensor[..., 4]
normalize(q, eps=1e-8) -> Tensor[..., 4]
```

- `hamilton(a, b)` — Hamilton product $a \otimes b$ with full broadcasting. `a` and `b` are real tensors whose last dimension is 4, representing `[w, x, y, z]`.
- `left_broadcast(q, W)` — Apply left multiplication by `q` to every row in `W`.
  - Computes `Y[i] = q ⊗ W[i]`.
  - `q`: shape `(4,)`
  - `W`: shape `(Nq, 4)` (learned weight quaternions)
  - returns: `(Nq, 4)`
- `conj(q)` — Quaternion conjugate: `[w, x, y, z] → [w, -x, -y, -z]`.
- `normalize(q)` — Unit‑normalize along the last axis using `q / clamp_min(‖q‖, eps)` to avoid division by zero in low precision.

---

## Quaternions and shapes

- A quaternion is a 4‑tuple of reals
  $$q = w + x\,\mathbf{i} + y\,\mathbf{j} + z\,\mathbf{k}$$
  stored as a length‑4 real vector `[w, x, y, z]` (scalar first).

- Shape `(..., 4)`: the last dimension is exactly 4 for `[w, x, y, z]`. The leading `...` is any number of batch/structural dims.
  - `(4,)` — a single quaternion
  - `(N, 4)` — a batch of N quaternions
  - `(B, H, L, 4)` — quaternions per batch/head/position (common in attention models)

- Device & dtype: real tensors (`float32`, `float16`, `bfloat16`) on exactly one device (CPU xor a single CUDA device). Inputs must share the same device and same dtype; no implicit promotion.

---

## Hamilton product

For $a=(w_1,x_1,y_1,z_1)$ and $b=(w_2,x_2,y_2,z_2)$,

$$\begin{aligned}
w &= w_1 w_2 - x_1 x_2 - y_1 y_2 - z_1 z_2,\\
x &= w_1 x_2 + x_1 w_2 + y_1 z_2 - z_1 y_2,\\
y &= w_1 y_2 - x_1 z_2 + y_1 w_2 + z_1 x_2,\\
z &= w_1 z_2 + x_1 y_2 - y_1 x_2 + z_1 w_2.
\end{aligned}$$

Properties:

- Associative: $(a\otimes b)\otimes c = a\otimes(b\otimes c)$
- Not commutative: $a\otimes b \ne b\otimes a$
- Identity: $1=(1,0,0,0)$, so $a\otimes1=1\otimes a=a$
- Conjugate: $a^*=(w,-x,-y,-z)$
- Norm: $\lVert a\rVert^2=w^2+x^2+y^2+z^2$
- Inverse (if $\lVert a\rVert\ne 0$): $a^{-1}=a^*/\lVert a\rVert^2$
- Rotations: unit quaternions ($\lVert a\rVert=1$) represent 3D rotations

---

## Broadcasting

PyTorch broadcasting rules apply to all leading dims (the last dim is fixed at 4):

- Match from right to left (ignoring the last `4`).
- A dim matches if equal, or if one of them is `1` (the `1` expands).
- The result has the elementwise max of the matched dims.

Examples:

- Apply one weight to everything: `a.shape=(B,H,L,4)`, `b.shape=(1,1,1,4)` → result `(B,H,L,4)`.
- Per‑head weights: `a=(B,H,L,4)`, `b=(1,H,1,4)` → each head gets its own `b`.
- Per‑token weights: `a=(B,H,L,4)`, `b=(1,1,L,4)` → each position gets its own `b`.
- One‑to‑one batch: `a=(N,4)`, `b=(N,4)` → result `(N,4)`.

---

## Examples

```python
import torch
from qham import hamilton

# 1) Single multiply
q = torch.tensor([1., 2., 3., 4.])
r = torch.tensor([0.5, -1., 0.25, 2.])
qr = hamilton(q, r)                  # (4,)

# 2) Batch multiply
Q = torch.randn(32, 4)
R = torch.randn(32, 4)
QR = hamilton(Q, R)                  # (32,4)

# 3) Broadcast a single weight across a batch
W = torch.tensor([1., 0., 0., 0.])   # identity quaternion
QR = hamilton(Q, W)                  # (32,4) → equals Q

# 4) Multi-dim with per-head weights (GPU + AMP)
B, H, L = 2, 8, 128
Q = torch.randn(B, H, L, 4, device="cuda", dtype=torch.float16)
R = torch.randn(1, H, 1, 4, device="cuda", dtype=torch.float16)
with torch.cuda.amp.autocast(dtype=torch.float16):
    QR = hamilton(Q, R)              # (B,H,L,4)
```

**RGB → embedding (left‑broadcast)**

```python
from qham import left_broadcast

# q: one RGB mapped to a pure imaginary quaternion [0, r, g, b] in [-1,1]
q = torch.tensor([0., 0.2, -0.1, 0.7], device="cuda", dtype=torch.bfloat16)

# W: one learned quaternion per 4-d slice of the embedding
embed_dim = 4096
Nq = embed_dim // 4
W = torch.randn(Nq, 4, device="cuda", dtype=torch.bfloat16)

# Apply q to every row of W
Y = left_broadcast(q, W, backend="auto")   # (Nq, 4)
emb = Y.flatten()                           # (embed_dim,)
```

---

## Autograd (training)

- The PyTorch path is built from standard tensor ops; gradients flow automatically through `hamilton` and `left_broadcast`.
- The Triton and CUDA paths compute backward analytically and match the reference implementation.
- Compose `hamilton` inside models and train end‑to‑end.

---

## Precision & performance tips

- Dtypes: `float32` is safest; `bfloat16`/`float16` are fine on modern GPUs (prefer AMP with fp32 accumulation during training).
- Speed:
  - Start with `backend="auto"`; on CUDA devices, `qham` selects the fastest available backend.
  - For the one‑`q` across many `W` pattern, `left_broadcast` loads `q` once and streams through `W`—minimal bytes per output, no temporary 4×4 matrices.
  - If you need more, try `backend="triton"` (no compile toolchain) or `backend="cuda"` (prebuilt kernels).

---

## Common pitfalls

- Last dim must be 4. If you see a size mismatch, print shapes: the trailing axis must be `4`.
- Device mismatch. Inputs must live on the same device.
- Layout confusion. This library always uses `[w, x, y, z]`. If you have `[x, y, z, w]`, convert:

```python
def xyzw_to_wxyz(t): return torch.stack([t[...,3], t[...,0], t[...,1], t[...,2]], -1)
def wxyz_to_xyzw(t): return torch.stack([t[...,1], t[...,2], t[...,3], t[...,0]], -1)
```

---

## (Optional) Matrix view

Left multiplication can be represented as a $4\times4$ matrix $L(q)$ so that $q\otimes r = L(q)\,r$. This is convenient on paper, but no 4×4 matrices are built per row in this library.

---

## Why this design?

- Tiny, stable API — `hamilton` and `left_broadcast` cover the hot paths, so you can plug them into embedding layers and attention stacks.
- Backend‑agnostic — start with PyTorch; switch to Triton or CUDA when profiling says it’s worth it.
- No surprises — shapes are explicit (`(...,4)`), broadcasting is standard PyTorch, and autograd is guaranteed.

---

## Requirements

- Python: 3.9–3.12
- PyTorch: ≥ 2.x (Torch backend only for now)
- CUDA (optional): to run on GPU via the Torch backend
- Triton/CUDA backends: coming soon (commented out in code/docs for now)

## Install

```bash
pip install qham           # when published

# From source (editable dev install):
pip install -e .
# or (if you expose extras):
# pip install -e .[dev]
```

To get a recommended PyTorch wheel channel for your system (CPU, cu121, cu118), run:

```bash
python scripts/recommend_install.py
```

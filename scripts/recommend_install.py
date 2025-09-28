#!/usr/bin/env python3
import re
import subprocess
from textwrap import dedent


def _run(cmd: str) -> str:
    try:
        return subprocess.check_output(
            cmd, shell=True, text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception:
        return ""


def detect_cuda_version() -> str | None:
    nsmi = _run("nvidia-smi")
    nvcc = _run("nvcc --version")

    # Try to find 'CUDA Version: X.Y' in nvidia-smi
    m = re.search(r"CUDA Version:\s*([0-9]+)\.([0-9]+)", nsmi)
    if m:
        return f"{m.group(1)}.{m.group(2)}"

    # Try to find 'release X.Y' in nvcc --version
    m = re.search(r"release\s+([0-9]+)\.([0-9]+)", nvcc)
    if m:
        return f"{m.group(1)}.{m.group(2)}"

    return None


def choose_index(cuda_version: str | None) -> tuple[str, str]:
    # Map a few common CUDA runtimes to PyTorch wheel channels.
    if not cuda_version:
        return "cpu", "https://download.pytorch.org/whl/cpu"

    try:
        major_minor = tuple(map(int, cuda_version.split(".")[:2]))
    except Exception:
        return "cpu", "https://download.pytorch.org/whl/cpu"

    if major_minor >= (12, 1):
        return "cu121", "https://download.pytorch.org/whl/cu121"
    if major_minor >= (11, 8):
        return "cu118", "https://download.pytorch.org/whl/cu118"
    # Fallback to CPU if older/unknown
    return "cpu", "https://download.pytorch.org/whl/cpu"


def main() -> int:
    cuda_version = detect_cuda_version()
    channel, url = choose_index(cuda_version)

    print(
        dedent(
            f"""
            Detected CUDA runtime: {cuda_version or "not found"}
            Recommended PyTorch wheel channel: {channel}

            Suggested pip command:
                pip install --index-url {url} torch torchvision torchaudio

            Notes:
              - If the channel 404s, check https://pytorch.org/get-started/locally/
              - Triton/CUDA backends in qham are not required yet; Torch backend works on CPU or CUDA.
            """
        ).strip()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


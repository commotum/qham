import os

import pytest

from qham._backend import select_backend


def test_explicit_torch():
    assert select_backend("torch") == "torch"


def test_auto_default_is_torch(monkeypatch):
    monkeypatch.delenv("QHAM_BACKEND", raising=False)
    assert select_backend("auto") == "torch"


def test_env_torch(monkeypatch):
    monkeypatch.setenv("QHAM_BACKEND", "torch")
    assert select_backend("auto") == "torch"


@pytest.mark.parametrize("value", ["triton", "cuda"])
def test_not_implemented_backends_explicit(value):
    with pytest.raises(NotImplementedError):
        select_backend(value)


@pytest.mark.parametrize("value", ["triton", "cuda"])
def test_not_implemented_backends_env(monkeypatch, value):
    monkeypatch.setenv("QHAM_BACKEND", value)
    with pytest.raises(NotImplementedError):
        select_backend("auto")


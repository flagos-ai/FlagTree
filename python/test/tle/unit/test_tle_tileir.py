"""Explicit token chains for TileIR TLE pointer, atomic, and view operations."""

import pytest

import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@pytest.fixture(autouse=True)
def tileir_backend(monkeypatch):
    if "tileir" not in triton.backends.backends:
        pytest.skip("TileIR backend is not installed")
    monkeypatch.setenv("FLAGTREE_USE_TILEIR", "1")


@pytest.fixture
def cuda_torch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    return torch


@triton.jit
def _pointer_token_chain(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    token = tle.create_mem_token()
    value, token = tle.load_tko(X + offsets, mask, other=0, memToken=token, has_result_token=True)
    token = tle.store_tko(Y + offsets, value + 1, mask, memToken=token, has_result_token=True)
    tle.store_tko(Y + offsets, value + 2, mask, memToken=token)


def test_pointer_token_chain(cuda_torch):
    torch = cuda_torch
    n = 259
    x = torch.arange(n, device="cuda", dtype=torch.float32)
    y = torch.full((n + 16, ), -1, device="cuda", dtype=x.dtype)
    kernel = _pointer_token_chain[(triton.cdiv(n, 128), )](x, y, n, 128)
    torch.testing.assert_close(y[:n], x + 2, atol=0, rtol=0)
    assert torch.all(y[n:] == -1)
    assert kernel.metadata.target.backend == "tileir"


@triton.jit
def _atomic_token_chain(X, Y):
    pid = tl.program_id(0)
    token = tle.create_mem_token()
    first, token = tle.atomic_cas_tko(X + pid, 0, 1, memToken=token, semantics="acquire", has_result_token=True)
    second, token = tle.atomic_cas_tko(X + pid, 0, 2, memToken=token, has_result_token=True)
    old, token = tle.atomic_xchg_tko(X + pid, 3, memToken=token, semantics="release", has_result_token=True)
    final = tle.load_tko(X + pid, memToken=token)
    tl.store(Y + pid * 4, first)
    tl.store(Y + pid * 4 + 1, second)
    tl.store(Y + pid * 4 + 2, old)
    tl.store(Y + pid * 4 + 3, final)


def test_atomic_token_chain(cuda_torch):
    torch = cuda_torch
    x = torch.zeros(8, device="cuda", dtype=torch.int32)
    y = torch.full((8, 4), -1, device="cuda", dtype=torch.int32)
    kernel = _atomic_token_chain[(8, )](x, y)
    expected = torch.tensor([0, 1, 1, 3], device="cuda", dtype=torch.int32).expand_as(y)
    torch.testing.assert_close(y, expected, atol=0, rtol=0)
    assert torch.all(x == 3)
    assert kernel.metadata.target.backend == "tileir"


@triton.jit
def _view_token_chain(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    x_view = tle.make_view(X, [N], [1], [BLOCK], [0])
    y_view = tle.make_view(Y, [N], [1], [BLOCK], [0])
    start = tle.create_mem_token()
    value, token = tle.load_view_tko(x_view, [pid], memToken=start, semantics="acquire", scope="device",
                                     has_result_token=True)
    token = tle.join_mem_tokens(start, token)
    token = tle.store_view_tko(y_view, value + 1, [pid], memToken=token, semantics="release", scope="device",
                               has_result_token=True)
    tle.store_view_tko(y_view, value + 2, [pid], memToken=token)


def test_view_token_chain(cuda_torch):
    torch = cuda_torch
    n = 259
    x = torch.arange(n, device="cuda", dtype=torch.float32)
    y = torch.full((n + 16, ), -1, device="cuda", dtype=x.dtype)
    kernel = _view_token_chain[(triton.cdiv(n, 128), )](x, y, n, 128)
    torch.testing.assert_close(y[:n], x + 2, atol=0, rtol=0)
    assert torch.all(y[n:] == -1)
    assert kernel.metadata.target.backend == "tileir"

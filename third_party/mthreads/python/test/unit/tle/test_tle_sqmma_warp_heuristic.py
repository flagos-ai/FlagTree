"""MUSA SQMMA warp selection preserves explicit and chained layouts."""

import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from test_tle_warp_specialize import _compile_ws_ir


@triton.jit
def dot_chain(Q, K, Out, M: tl.constexpr, N: tl.constexpr, DTYPE: tl.constexpr, MODE: tl.constexpr):
    offsets = tl.arange(0, 64)[None, :]
    query = tl.load(Q + tl.arange(0, M)[:, None] * 64 + offsets).to(DTYPE)
    key = tl.load(K + tl.arange(0, N)[:, None] * 64 + offsets).to(DTYPE)
    q = tle.gpu.alloc((M, 64), DTYPE, init_value=query)
    k = tle.gpu.alloc((N, 64), DTYPE, init_value=key)
    tl.debug_barrier()
    scores = tle.gpu.wgmma(q, k, trans_b=True)
    if MODE == "async_acc":
        scores = tle.gpu.wgmma(q, k, acc=scores, trans_b=True)
    scores = tle.gpu.wgmma_wait(0, scores)
    if MODE == "none" or MODE == "async_acc":
        result = scores
    elif MODE == "c_operand":
        result = tl.dot(tl.full((M, 64), 1.0, DTYPE), tl.full((64, N), 1.0, DTYPE), scores)
    else:
        if MODE == "row":
            scores = tl.exp2(scores - tl.max(scores, axis=1)[:, None])
            scores = scores / tl.sum(scores, axis=1)[:, None]
        elif MODE == "column":
            scores = scores - tl.sum(scores, axis=0)[None, :]
        elif MODE == "gather":
            indices = tl.broadcast_to((tl.arange(0, N) ^ 1)[None, :], (M, N))
            scores = tl.gather(scores, indices, axis=1)
        if MODE == "transpose":
            result = tl.dot(tl.trans(scores).to(DTYPE), tl.full((M, 128), 1.0, DTYPE))
        elif MODE == "b_operand":
            result = tl.dot(tl.full((128, M), 1.0, DTYPE), scores.to(DTYPE))
        elif MODE == "shared":
            p = tle.gpu.alloc((M, N), DTYPE, init_value=scores.to(DTYPE))
            v = tle.gpu.alloc((N, 128), DTYPE, init_value=tl.full((N, 128), 1.0, DTYPE))
            tl.debug_barrier()
            result = tle.gpu.wgmma(p, v)
            result = tle.gpu.wgmma_wait(0, result)
        else:
            result = tl.dot(scores.to(DTYPE), tl.full((N, 128), 1.0, DTYPE))
    rows = tl.arange(0, result.shape[0])[:, None]
    cols = tl.arange(0, result.shape[1])[None, :]
    tl.store(Out + rows * result.shape[1] + cols, result)


@pytest.mark.parametrize("dtype", [tl.float8e4nv, tl.bfloat16])
@pytest.mark.parametrize("mode", ["raw", "row", "column", "gather", "transpose", "b_operand", "c_operand", "shared"])
def test_chained_dot_prefers_m_split(dtype, mode):
    _, ttgir = _compile_ws_ir(dot_chain, {"Q": "*fp32", "K": "*fp32", "Out": "*fp32"},
                              {"M": 32, "N": 64, "DTYPE": dtype, "MODE": mode}, consumer_warps=8)
    dot = next(line for line in ttgir.splitlines() if " = ttmg.squad_dot " in line)
    encoding = re.search(r"-> tensor<32x64xf32, (#[^>]+)>", dot).group(1)
    layout = next(line for line in ttgir.splitlines() if line.startswith(encoding + " = "))
    assert "warpsPerCTA = [8, 1], instrShape = [16, 64," in layout


@pytest.mark.parametrize("m,n,warps,mode,expected", [
    (32, 64, 4, "none", "warpsPerCTA = [4, 1], instrShape = [32, 64,"),
    (32, 64, 8, "none", "warpsPerCTA = [4, 2], instrShape = [32, 32,"),
    (64, 64, 8, "none", "warpsPerCTA = [4, 2], instrShape = [64, 32,"),
    (128, 128, 8, "none", "warpsPerCTA = [8, 1], instrShape = [64, 128,"),
    (128, 128, 16, "none", "warpsPerCTA = [8, 2], instrShape = [64, 64,"),
    (32, 128, 8, "raw", "warpsPerCTA = [8, 1], instrShape = [16, 64,"),
    (16, 128, 8, "row", "warpsPerCTA = [4, 2], instrShape = [16, 64,"),
])
def test_default_growth_and_legal_fallback(m, n, warps, mode, expected):
    _, ttgir = _compile_ws_ir(dot_chain, {"Q": "*fp32", "K": "*fp32", "Out": "*fp32"},
                              {"M": m, "N": n, "DTYPE": tl.bfloat16, "MODE": mode}, consumer_warps=warps)
    dot = next(line for line in ttgir.splitlines() if " = ttmg.squad_dot " in line)
    encoding = re.search(rf"-> tensor<{m}x{n}xf32, (#[^>]+)>", dot).group(1)
    assert expected in next(line for line in ttgir.splitlines() if line.startswith(encoding + " = "))


def test_async_c_chain_preserves_warp_split():
    _, ttgir = _compile_ws_ir(dot_chain, {"Q": "*fp32", "K": "*fp32", "Out": "*fp32"},
                              {"M": 32, "N": 64, "DTYPE": tl.bfloat16, "MODE": "async_acc"}, consumer_warps=8)
    dots = [line for line in ttgir.splitlines() if " = ttmg.squad_dot " in line]
    assert len(dots) == 2
    encodings = [re.search(r"-> tensor<32x64xf32, (#[^>]+)>", dot).group(1) for dot in dots]
    assert encodings[0] == encodings[1]
    layout = next(line for line in ttgir.splitlines() if line.startswith(encodings[0] + " = "))
    assert "warpsPerCTA = [8, 1], instrShape = [16, 64," in layout


@pytest.mark.parametrize("m,n,warps", [(32, 64, 8), (64, 128, 8), (128, 128, 16), (16, 128, 8)])
def test_last_sqmma_preserves_warp_split(m, n, warps):
    _, ttgir = _compile_ws_ir(dot_chain, {"Q": "*fp32", "K": "*fp32", "Out": "*fp32"},
                              {"M": m, "N": n, "DTYPE": tl.bfloat16, "MODE": "shared"}, consumer_warps=warps)
    dots = [line for line in ttgir.splitlines() if " = ttmg.squad_dot " in line]
    assert len(dots) == 2
    splits = []
    for dot in dots:
        encoding = re.search(r"-> tensor<[^>]+, (#[^>]+)>", dot).group(1)
        layout = next(line for line in ttgir.splitlines() if line.startswith(encoding + " = "))
        splits.append(re.search(r"warpsPerCTA = (\[[^]]+\])", layout).group(1))
    assert splits[0] == splits[1]


@pytest.mark.parametrize("dtype,torch_dtype", [(tl.float8e4nv, torch.float8_e4m3fn), (tl.bfloat16, torch.bfloat16)])
@pytest.mark.parametrize("mode", ["none", "raw", "row", "async_acc", "shared"])
def test_warp_heuristic_runtime(dtype, torch_dtype, mode):
    import torch_musa

    if not torch_musa.is_available():
        pytest.skip("MUSA device is not available")
    torch.manual_seed(1451)
    query = torch.randn((32, 64)) * 0.25
    key = torch.randn((64, 64)) * 0.25
    scores = query.to(torch_dtype).double() @ key.to(torch_dtype).double().T
    if mode == "none":
        expected = scores.float()
    elif mode == "async_acc":
        expected = (2 * scores).float()
    else:
        probabilities = scores.float()
        if mode == "row":
            probabilities = torch.exp2(probabilities - probabilities.max(dim=1, keepdim=True).values)
            probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
        expected = probabilities.to(torch_dtype).double() @ torch.ones((64, 128), dtype=torch.float64)
        expected = expected.float()
    output = torch.empty(expected.shape, dtype=torch.float32, device="musa")
    dot_chain[(1, )](query.to("musa"), key.to("musa"), output, 32, 64, dtype, mode, num_warps=8, num_stages=1)
    torch.testing.assert_close(output.cpu(), expected, rtol=1e-3, atol=2e-3)

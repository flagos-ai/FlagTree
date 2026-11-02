"""NVIDIA tl.dot and TLE WGMMA share the MMAv3 warp-selection heuristic."""

import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

pytestmark = pytest.mark.skip_if_tle_unsupported("gpu.wgmma", "gpu.wgmma_wait")


@pytest.fixture(scope="module", autouse=True)
def require_hopper():
    if triton.runtime.driver.active.get_current_target().backend != "cuda":
        pytest.skip("NVIDIA WGMMA regression")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Hopper WGMMA regression")


@triton.jit(do_not_specialize=["ITERATIONS"])
def warp_heuristic_kernel(A, B, V, Out, M: tl.constexpr, FIRST_TLE: tl.constexpr, MODE: tl.constexpr, ITERATIONS):
    rows = tl.arange(0, M)[:, None]
    k = tl.arange(0, 64)
    a = tl.load(A + rows * 64 + k[None, :])
    b = tl.load(B + k[:, None] * 64 + k[None, :])
    if FIRST_TLE:
        shared_a = tle.gpu.alloc((M, 64), a.dtype, init_value=a)
        shared_b = tle.gpu.alloc((64, 64), b.dtype, init_value=b)
        tl.debug_barrier()
        scores = tle.gpu.wgmma(shared_a, shared_b)
        if MODE == "acc":
            scores = tle.gpu.wgmma(a, shared_b, scores)
        elif MODE == "loop":
            for _ in range(ITERATIONS):
                scores = tle.gpu.wgmma(a, shared_b, scores)
                scores = tle.gpu.wgmma_wait(1, scores)
        scores = tle.gpu.wgmma_wait(0, scores)
    else:
        scores = tl.dot(a, b)
        if MODE == "acc":
            scores = tl.dot(a, b, scores)
        elif MODE == "loop":
            for _ in range(ITERATIONS):
                scores = tl.dot(a, b, scores)
    if MODE == "raw" or MODE == "softmax" or MODE == "to_tle":
        if MODE == "softmax":
            scores = tl.exp2(scores - tl.max(scores, axis=1)[:, None])
            scores = scores / tl.sum(scores, axis=1)[:, None]
        v = tl.load(V + k[:, None] * 64 + k[None, :])
        if MODE == "to_tle":
            shared_v = tle.gpu.alloc((64, 64), v.dtype, init_value=v)
            tl.debug_barrier()
            result = tle.gpu.wgmma(scores.to(a.dtype), shared_v)
            result = tle.gpu.wgmma_wait(0, result)
        else:
            result = tl.dot(scores.to(a.dtype), v)
    else:
        result = scores
    tl.store(Out + rows * 64 + k[None, :], result)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("first_tle", [False, True], ids=["tl_dot", "tle_wgmma"])
@pytest.mark.parametrize("m,warps,mode,expected", [
    (64, 4, "none", [4, 1]),
    (64, 8, "none", [4, 2]),
    (128, 8, "none", [8, 1]),
    (128, 16, "none", [8, 2]),
    (256, 16, "none", [16, 1]),
    (64, 8, "raw", [8, 1]),
    (64, 8, "softmax", [8, 1]),
    (64, 8, "to_tle", [8, 1]),
    (64, 8, "acc", [8, 1]),
    (64, 8, "loop", [8, 1]),
])
def test_nvidia_warp_heuristic(dtype, first_tle, m, warps, mode, expected):
    torch.manual_seed(1451)
    a = (torch.randn((m, 64)) * 0.25).to(dtype)
    b = (torch.randn((64, 64)) * 0.25).to(dtype)
    v = (torch.randn((64, 64)) * 0.25).to(dtype)
    scores = (a.double() @ b.double()).float()
    if mode == "acc":
        reference = 2 * scores
    elif mode == "loop":
        reference = 3 * scores
    elif mode in ("raw", "softmax", "to_tle"):
        if mode == "softmax":
            scores = torch.exp2(scores - scores.max(dim=1, keepdim=True).values)
            scores = scores / scores.sum(dim=1, keepdim=True)
        reference = (scores.to(dtype).double() @ v.double()).float()
    else:
        reference = scores
    output = torch.empty((m, 64), dtype=torch.float32, device="cuda")
    compiled = warp_heuristic_kernel[(1, )](a.cuda(), b.cuda(), v.cuda(), output, m, first_tle, mode, 2,
                                            num_warps=warps, num_stages=1)
    torch.testing.assert_close(output.cpu(), reference, rtol=1e-3, atol=2e-3)
    ttgir = compiled.asm["ttgir"]
    dot = next(line for line in ttgir.splitlines() if " = ttng.warp_group_dot " in line)
    encoding = re.search(rf"-> tensor<{m}x64xf32, (#[^>]+)>", dot).group(1)
    layout = next(line for line in ttgir.splitlines() if line.startswith(encoding + " = "))
    assert f"warpsPerCTA = {expected}" in layout
    if mode == "loop":
        assert "scf.for" in ttgir

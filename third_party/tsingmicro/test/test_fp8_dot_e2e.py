"""End-to-end checks for the emulated fp8 ``tl.dot`` on the <PII:company:公司名称> backend.

``LinalgToMK`` upcasts fp8 operands to bf16 before ``mk.dot`` because the GEMM
engine has no FP8 data format.  These tests cover the shapes the upcast has to
survive: plain dot (including mixed e4m3 x e5m2, which the frontend allows),
transposed operands, accumulation inside a loop, a K-loop GEMM, plus
``tl.dot_scaled`` as the regression guard for the shared ``arith.extf`` helper.

Device execution is opt-in via ``TX81_RUN_DEVICE_TESTS`` (default on) and the
card can be pinned with ``TX81_TEST_DEVICE=<n>``.

Run with::

    cd third_party/<backend>
    TX81_TEST_DEVICE=16 python -m pytest test/test_fp8_dot_e2e.py -x -v

Note: ``test_fp8_dot_loop_accumulation`` requires the InsertBarrierPass fix
that treats FP8/FP4 conversions as CPU-executed (Change-Id I67b3ffc7...).
Without it the second and later iterations race with the pending ``tx.gemm``
and one output row comes back wrong; see the loop test's docstring.
"""

import os

import pytest
import triton
import triton.language as tl

M = N = K = 64
BK = 64
GEMM_K = 256
SCALE_BLOCK = 32


def _torch_device():
    try:
        import torch

        import torch_txda  # noqa: F401 - registers torch.txda
        pin = os.getenv("TX81_TEST_DEVICE")
        if pin is not None:
            torch.txda.set_device(int(pin))
        device = triton.runtime.driver.active.get_active_torch_device()
        torch.zeros(1, device=device)
        return device
    except Exception:
        return None


RUN_DEVICE = os.getenv("TX81_RUN_DEVICE_TESTS", "1").lower() in ("1", "true", "yes")
DEVICE = _torch_device() if RUN_DEVICE else None

pytestmark = pytest.mark.skipif(DEVICE is None, reason="no <PII:product:硬件型号编码> device available")


@triton.jit
def dot_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    om = tl.arange(0, M)
    ok = tl.arange(0, K)
    on = tl.arange(0, N)
    a = tl.load(A + om[:, None] * K + ok[None, :])
    b = tl.load(B + ok[:, None] * N + on[None, :])
    c = tl.dot(a, b)
    tl.store(C + om[:, None] * N + on[None, :], c)


@triton.jit
def dot_trans_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    # A is stored (K, M) and B is stored (N, K); both are transposed in-kernel.
    om = tl.arange(0, M)
    ok = tl.arange(0, K)
    on = tl.arange(0, N)
    a = tl.load(A + ok[:, None] * M + om[None, :])
    b = tl.load(B + on[:, None] * K + ok[None, :])
    c = tl.dot(tl.trans(a), tl.trans(b))
    tl.store(C + om[:, None] * N + on[None, :], c)


@triton.jit
def dot_loop_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, STEPS: tl.constexpr):
    om = tl.arange(0, M)
    ok = tl.arange(0, K)
    on = tl.arange(0, N)
    acc = tl.zeros((M, N), tl.float32)
    for _ in tl.range(0, STEPS):
        a = tl.load(A + om[:, None] * K + ok[None, :])
        b = tl.load(B + ok[:, None] * N + on[None, :])
        acc += tl.dot(a, b)
    tl.store(C + om[:, None] * N + on[None, :], acc)


@triton.jit
def fp8_gemm_kloop_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
    om = tl.arange(0, M)
    on = tl.arange(0, N)
    ok = tl.arange(0, BK)
    acc = tl.zeros((M, N), tl.float32)
    for k0 in tl.range(0, K, BK):
        a = tl.load(A + om[:, None] * K + (k0 + ok)[None, :])
        b = tl.load(B + (k0 + ok)[:, None] * N + on[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(C + om[:, None] * N + on[None, :], acc)


@triton.jit
def scaled_dot_kernel(A, SA, B, SB, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    om = tl.arange(0, M)
    ok = tl.arange(0, K)
    on = tl.arange(0, N)
    sg: tl.constexpr = K // 32
    a = tl.load(A + om[:, None] * K + ok[None, :])
    b = tl.load(B + ok[:, None] * N + on[None, :])
    sa = tl.load(SA + om[:, None] * sg + tl.arange(0, sg)[None, :])
    sb = tl.load(SB + on[:, None] * sg + tl.arange(0, sg)[None, :])
    acc = tl.zeros((M, N), tl.float32)
    c = tl.dot_scaled(a, sa, "e4m3", b, sb, "e4m3", acc)
    tl.store(C + om[:, None] * N + on[None, :], c)


TORCH_FP8 = {
    "e4m3": "float8_e4m3fn",
    "e5m2": "float8_e5m2",
}


def _quant(t, dtype):
    # Multiples of 0.25 are exactly representable in both e4m3 and e5m2, so the
    # bf16 upcast in LinalgToMK is lossless and the reference can be exact.
    return (t.float() / 4).to(dtype)


def _fp8(dtype):
    import torch
    return getattr(torch, TORCH_FP8[dtype])


def _assert_close(res, ref, atol=1e-2):
    import torch
    torch.testing.assert_close(res, ref, atol=atol, rtol=1e-3)


@pytest.mark.parametrize("a_dtype,b_dtype", [
    ("e4m3", "e4m3"),
    ("e5m2", "e5m2"),
    ("e4m3", "e5m2"),
])
def test_fp8_dot(a_dtype, b_dtype):
    """Plain fp8 tl.dot; mixed fp8 x fp8 combinations are allowed by semantic.py."""
    import torch

    torch.manual_seed(0)
    a_cpu = _quant(torch.randint(-8, 8, (M, K)), _fp8(a_dtype))
    b_cpu = _quant(torch.randint(-8, 8, (K, N)), _fp8(b_dtype))
    c = torch.empty((M, N), device=DEVICE, dtype=torch.float32)
    dot_kernel[(1, )](a_cpu.to(DEVICE), b_cpu.to(DEVICE), c, M, N, K)
    torch.txda.synchronize()
    _assert_close(c.cpu(), a_cpu.float() @ b_cpu.float())


@pytest.mark.parametrize("dtype", ["e4m3", "e5m2"])
def test_fp8_dot_transposed(dtype):
    """Transpose detection must run on the original operands, before the upcast."""
    import torch

    torch.manual_seed(2)
    a_cpu = _quant(torch.randint(-8, 8, (M, K)), _fp8(dtype))
    b_cpu = _quant(torch.randint(-8, 8, (K, N)), _fp8(dtype))
    c = torch.empty((M, N), device=DEVICE, dtype=torch.float32)
    dot_trans_kernel[(1, )](a_cpu.T.contiguous().to(DEVICE), b_cpu.T.contiguous().to(DEVICE), c, M, N, K)
    torch.txda.synchronize()
    _assert_close(c.cpu(), a_cpu.float() @ b_cpu.float())


@pytest.mark.parametrize("steps", [4, 8])
@pytest.mark.parametrize("num_stages", [None, 1, 2])
def test_fp8_dot_loop_accumulation(steps, num_stages):
    """fp8 upcast inside a loop, the pattern that exposed the missing barrier.

    The conversion lowers to a scalar CRT loop executed by the KCORE, so the
    InsertBarrierPass must treat it as CPU-side: otherwise the next iteration's
    conversion overwrites the SPM buffer a still-pending ``tx.gemm`` is reading
    and the first output row comes back wrong.  Requires the barrier fix
    (Change-Id I67b3ffc7...); fails deterministically without it at >= 4
    iterations and at any num_stages.
    """
    import torch

    torch.manual_seed(3)
    a_cpu = _quant(torch.randint(-8, 8, (M, K)), _fp8("e4m3"))
    b_cpu = _quant(torch.randint(-8, 8, (K, N)), _fp8("e4m3"))
    c = torch.empty((M, N), device=DEVICE, dtype=torch.float32)
    launch = {} if num_stages is None else {"num_stages": num_stages}
    dot_loop_kernel[(1, )](a_cpu.to(DEVICE), b_cpu.to(DEVICE), c, M, N, K, steps, **launch)
    torch.txda.synchronize()
    _assert_close(c.cpu(), (a_cpu.float() @ b_cpu.float()) * steps)


@pytest.mark.parametrize("dtype", ["e4m3", "e5m2"])
def test_fp8_gemm_kloop(dtype):
    """K-loop GEMM with psum accumulation: the realistic fp8 matmul shape."""
    import torch

    torch.manual_seed(5)
    a_cpu = _quant(torch.randint(-8, 8, (M, GEMM_K)), _fp8(dtype))
    b_cpu = _quant(torch.randint(-8, 8, (GEMM_K, N)), _fp8(dtype))
    c = torch.empty((M, N), device=DEVICE, dtype=torch.float32)
    fp8_gemm_kloop_kernel[(1, )](a_cpu.to(DEVICE), b_cpu.to(DEVICE), c, M, N, GEMM_K, BK)
    torch.txda.synchronize()
    _assert_close(c.cpu(), a_cpu.float() @ b_cpu.float(), atol=2e-2)


def test_dot_scaled_mxfp8():
    """tl.dot_scaled must stay identical after sharing createExtFGeneric."""
    import torch

    torch.manual_seed(1)
    sg = K // SCALE_BLOCK
    a_cpu = _quant(torch.randint(-8, 8, (M, K)), _fp8("e4m3"))
    b_cpu = _quant(torch.randint(-8, 8, (K, N)), _fp8("e4m3"))
    # MXFP8 scales are E8M0 stored in uint8: value = 2 ** (scale - 127).
    sa_cpu = torch.randint(124, 130, (M, sg), dtype=torch.uint8)
    sb_cpu = torch.randint(124, 130, (N, sg), dtype=torch.uint8)
    c = torch.empty((M, N), device=DEVICE, dtype=torch.float32)
    scaled_dot_kernel[(1, )](a_cpu.to(DEVICE), sa_cpu.to(DEVICE), b_cpu.to(DEVICE), sb_cpu.to(DEVICE), c, M, N, K)
    torch.txda.synchronize()
    # A's scale expands along K.  B's scale is indexed by the output column and
    # applies to B's rows, i.e. it is transposed relative to the (K, N) layout.
    scale_a = torch.pow(2.0, sa_cpu.float() - 127).repeat_interleave(SCALE_BLOCK, 1)
    scale_b = torch.pow(2.0, sb_cpu.float() - 127).repeat_interleave(SCALE_BLOCK, 1).T
    ref = (a_cpu.float() * scale_a) @ (b_cpu.float() * scale_b)
    _assert_close(c.cpu(), ref, atol=2e-2)

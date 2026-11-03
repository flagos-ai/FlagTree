"""gfx936 async LDS copies and legacy MMAC layout regressions."""
import re
import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton import knobs


def _require_gfx936():
    if not torch.cuda.is_available():
        pytest.skip("requires an HCU")
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.arch != "gfx936":
        pytest.skip("gfx936-specific code generation")


@triton.jit
def _copy(A, C, WIDTH: tl.constexpr):
    r = tl.broadcast_to(tl.arange(0, 16)[:, None], (16, WIDTH))
    c = tl.broadcast_to(tl.arange(0, WIDTH)[None, :], (16, WIDTH))
    buf = tle.gpu.alloc((16, WIDTH), A.dtype.element_ty, nv_mma_shared_layout=False)
    buf.store(tle.load(A + r * WIDTH + c, is_async=True))
    tl.store(C + r * WIDTH + c, buf.load())


class _LLIRCaptured(Exception):
    pass


@pytest.mark.parametrize("dtype,width", [
    ("fp16", 64),
    ("bf16", 64),
    ("bf16", 128),
    ("fp32", 64),
])
def test_global_to_lds_uses_supported_width(dtype, width, monkeypatch, tmp_path):
    # Run real make_llir; stop before DTK's known GLOBAL PostRA hazard crash.
    # This regression covers GLOBAL opcode legality, not vendor codegen/runtime.
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    backend = pytest.importorskip("triton.backends.hcu.compiler_hcu")
    captured = []

    def capture(llir, metadata, options):
        captured.append(llir)
        raise _LLIRCaptured

    monkeypatch.setattr(knobs.cache, "dir", str(tmp_path / "cache"))
    monkeypatch.setattr(knobs.hcu, "use_buffer_ops", False)
    monkeypatch.setattr(backend.HIPBackend, "make_amdgcn", staticmethod(capture))
    source = ASTSource(
        _copy, signature={"A": "*" + dtype, "C": "*" + dtype},
        constexprs={"WIDTH": width},
        attrs={(i,): [("tt.divisibility", 16), ("tt.pointer_range", 32)]
               for i in range(2)},
    )
    with pytest.raises(_LLIRCaptured):
        triton.compile(source, target=GPUTarget("hip", "gfx936", 64),
                       options={"num_warps": 4, "num_stages": 1})
    assert len(captured) == 1
    calls = [line for line in captured[0].splitlines()
             if "call " in line and "@llvm.amdgcn.global.load.lds" in line]
    assert calls
    assert all(re.search(r", i32 4, i32 \d+, i32 \d+\)", line) for line in calls)
    assert "@llvm.amdgcn.raw.ptr.buffer.load.lds" not in captured[0]


@triton.jit
def _mm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        AS0: tl.constexpr, AS1: tl.constexpr, BS0: tl.constexpr, BS1: tl.constexpr,
        AA: tl.constexpr, AB: tl.constexpr, BK: tl.constexpr = 32):
    m = tl.program_id(0) * 64 + tl.arange(0, 64)
    n = tl.program_id(1) * 64 + tl.arange(0, 64)
    k = tl.arange(0, BK)
    acc = tl.zeros((64, 64), tl.float32)
    for i in tl.range(0, tl.cdiv(K, BK), num_stages=2):
        rk = i * BK + k
        a = tle.load(A + m[:, None] * AS0 + rk[None, :] * AS1,
                     (m[:, None] < M) & (rk[None, :] < K), 0, is_async=AA)
        b = tle.load(B + rk[:, None] * BS0 + n[None, :] * BS1,
                     (rk[:, None] < K) & (n[None, :] < N), 0, is_async=AB)
        acc = tl.dot(a, b, acc)
    tl.store(C + m[:, None] * N + n[None, :], acc,
             (m[:, None] < M) & (n[None, :] < N))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(64, 64, 128), (65, 97, 67)])
@pytest.mark.parametrize("layout", ["normal", "transposed_a", "transposed_b"])
@pytest.mark.parametrize("policies", [(False, False), (True, True), (True, False), (False, True)])
def test_legacy_mmac_layout(dtype, shape, layout, policies):
    _require_gfx936()
    m, n, k = shape
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((k, n), device="cuda", dtype=dtype)
    if layout == "transposed_a":
        a = a.T.contiguous().T
    if layout == "transposed_b":
        b = b.T.contiguous().T
    c = torch.empty((m, n), device="cuda", dtype=dtype)
    kernel = _mm[(triton.cdiv(m, 64), triton.cdiv(n, 64))](
        a, b, c, m, n, k, *a.stride(), *b.stride(), *policies,
        num_warps=4, num_stages=2, optimize_epilogue=False)
    torch.testing.assert_close(c.float(), (a @ b).float(), rtol=0.02, atol=0.125)
    has_matrix_read = "llvm.hcu.ds.read.m32x16.b16" in kernel.asm["llir"]
    assert has_matrix_read == (layout != "transposed_b")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("k", [16, 272])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_kpack_exceeds_logical_k_tile(dtype, k, asynchronous):
    _require_gfx936()
    m = n = 64
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((k, n), device="cuda", dtype=dtype)
    c = torch.empty((m, n), device="cuda", dtype=dtype)
    _mm[(1, 1)](a, b, c, m, n, k, *a.stride(), *b.stride(),
                 asynchronous, asynchronous, 16, kpack=2,
                 num_warps=4, num_stages=2, optimize_epilogue=False)
    torch.testing.assert_close(c.float(), (a @ b).float(), rtol=0.02, atol=0.125)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mmac_wave_partition_runtime(dtype):
    _require_gfx936()
    m, n, k = 65, 97, 67
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((k, n), device="cuda", dtype=dtype)
    c = torch.empty((m, n), device="cuda", dtype=dtype)
    kernel = _mm[(2, 2)](a, b, c, m, n, k, *a.stride(), *b.stride(), False, False,
                          num_warps=8, num_stages=2, mmac_warps_m=4,
                          optimize_epilogue=False)
    torch.testing.assert_close(c.float(), (a @ b).float(), rtol=0.02, atol=0.125)
    assert '#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 2]' in kernel.asm['ttgir']


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape,warps,warp_m", [
    ((64, 64, 128), 4, None),
    ((65, 97, 67), 8, 4),
    ((65, 97, 17), 4, None),
])
def test_local_prefetch_runtime(dtype, shape, warps, warp_m):
    _require_gfx936()
    m, n, k = shape
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((k, n), device="cuda", dtype=dtype)
    c = torch.empty((m, n), device="cuda", dtype=dtype)
    options = {"num_warps": warps, "num_stages": 2,
               "optimize_epilogue": False, "schedule_hint": "local-prefetch"}
    if warp_m is not None:
        options["mmac_warps_m"] = warp_m
    kernel = _mm[(triton.cdiv(m, 64), triton.cdiv(n, 64))](
        a, b, c, m, n, k, *a.stride(), *b.stride(), False, False, **options)
    torch.testing.assert_close(c.float(), (a @ b).float(), rtol=0.02, atol=0.125)
    if warp_m is not None:
        assert f"warpsPerCTA = [{warp_m}, {warps // warp_m}]" in kernel.asm["ttgir"]

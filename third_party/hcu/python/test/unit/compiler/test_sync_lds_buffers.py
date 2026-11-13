# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""gfx936 synchronous LDS ring: standalone block-pointer GEMM and codegen guards."""
import re
import subprocess
import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton import knobs
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

# Keep the production block-pointer loop shape without a FlagGems dependency.
# Only its synchronous, 16-bit path is needed for these compiler regressions.
@triton.jit(do_not_specialize=['K'])
def _sync_mm(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr, IS_FP64: tl.constexpr=False, MASK_EDGES: tl.constexpr=False, USE_TLE: tl.constexpr=False, LARGE_TILE_CONFIGS: tl.constexpr=False, ASYNC_LOAD: tl.constexpr=False, LOAD_MODE: tl.constexpr=-1, ASYNC_LOAD_MASK: tl.constexpr=0):
    tl.static_assert(LOAD_MODE == -1 or (LOAD_MODE >= 0 and LOAD_MODE <= 3))
    tl.static_assert(LOAD_MODE == -1 or LOAD_MODE & ~ASYNC_LOAD_MASK == 0)
    async_a: tl.constexpr = ASYNC_LOAD if LOAD_MODE == -1 else LOAD_MODE & 1 != 0
    async_b: tl.constexpr = ASYNC_LOAD if LOAD_MODE == -1 else LOAD_MODE & 2 != 0
    if LARGE_TILE_CONFIGS or ASYNC_LOAD_MASK:
        k_bound = tl.multiple_of(K.to(tl.int64), 16)
    else:
        k_bound = K.to(tl.int64)
    pid = tl.program_id(0).to(tl.int64)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    group_size_m = min(num_pid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    if MASK_EDGES:
        offs_am_cont = offs_am
        offs_bn_cont = offs_bn
    else:
        offs_am_cont = tl.max_contiguous(tl.multiple_of(offs_am % M, BLOCK_M), BLOCK_M)
        offs_bn_cont = tl.max_contiguous(tl.multiple_of(offs_bn % N, BLOCK_N), BLOCK_N)
    prev_k_mult = tl.maximum(tl.cdiv(K, BLOCK_K) * BLOCK_K - BLOCK_K, 0)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, k_bound), strides=(stride_am, stride_ak), offsets=((pid_m * BLOCK_M).to(tl.int32), 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(k_bound, N), strides=(stride_bk, stride_bn), offsets=(0, (pid_n * BLOCK_N).to(tl.int32)), block_shape=(BLOCK_K, BLOCK_N), order=(1, 0))
    for start_k in range(0, prev_k_mult, BLOCK_K):
        rk = start_k + offs_k
        if not MASK_EDGES and USE_TLE:
            a = tle.load(a_block_ptr, is_async=async_a)
            b = tle.load(b_block_ptr, is_async=async_b)
            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
        elif MASK_EDGES and USE_TLE:
            a = tle.load(a_block_ptr, boundary_check=(0,), padding_option='zero', is_async=async_a)
            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
            b = tle.load(b_ptr + (rk[:, None] * stride_bk + offs_bn_cont[None, :] * stride_bn), mask=(offs_bn_cont < N)[None, :], other=0.0, is_async=async_b)
        elif MASK_EDGES:
            a = tl.load(a_ptr + (offs_am_cont[:, None] * stride_am + rk[None, :] * stride_ak), mask=(offs_am_cont < M)[:, None], other=0.0)
            b = tl.load(b_ptr + (rk[:, None] * stride_bk + offs_bn_cont[None, :] * stride_bn), mask=(offs_bn_cont < N)[None, :], other=0.0)
        else:
            a = tl.load(a_ptr + (offs_am_cont[:, None] * stride_am + rk[None, :] * stride_ak))
            b = tl.load(b_ptr + (rk[:, None] * stride_bk + offs_bn_cont[None, :] * stride_bn))
        if a.dtype != b.dtype:
            a = a.to(c_ptr.dtype.element_ty)
            b = b.to(c_ptr.dtype.element_ty)
        accumulator += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
    rk = prev_k_mult + offs_k
    mask_k = rk < K
    if not MASK_EDGES and USE_TLE:
        if LARGE_TILE_CONFIGS and BLOCK_M == 256 and (BLOCK_N == 256) and (BLOCK_K == 16):
            a = tle.load(a_block_ptr, is_async=async_a)
            b = tle.load(b_block_ptr, is_async=async_b)
        else:
            a = tle.load(a_block_ptr, boundary_check=(0, 1), padding_option='zero', is_async=async_a)
            b = tle.load(b_block_ptr, boundary_check=(0, 1), padding_option='zero', is_async=async_b)
    elif MASK_EDGES and USE_TLE:
        a = tle.load(a_block_ptr, boundary_check=(0, 1), padding_option='zero', is_async=async_a)
        b = tle.load(b_ptr + (rk[:, None] * stride_bk + offs_bn_cont[None, :] * stride_bn), mask=mask_k[:, None] & ((offs_bn_cont < N)[None, :] if MASK_EDGES else True), other=0.0, is_async=async_b)
    else:
        a = tl.load(a_ptr + (offs_am_cont[:, None] * stride_am + rk[None, :] * stride_ak), mask=mask_k[None, :] & ((offs_am_cont < M)[:, None] if MASK_EDGES else True), other=0.0)
        b = tl.load(b_ptr + (rk[:, None] * stride_bk + offs_bn_cont[None, :] * stride_bn), mask=mask_k[:, None] & ((offs_bn_cont < N)[None, :] if MASK_EDGES else True), other=0.0)
    if a.dtype != b.dtype:
        a = a.to(c_ptr.dtype.element_ty)
        b = b.to(c_ptr.dtype.element_ty)
    accumulator += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
    accumulator = accumulator.to(c_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptr = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    mask_store = (offs_cm < M)[:, None] & (offs_cn < N)[None, :]
    if LARGE_TILE_CONFIGS and (not MASK_EDGES) and (BLOCK_M == 256) and (BLOCK_N == 256):
        mask_store = None
    tl.store(c_ptr, accumulator, mask=mask_store)


@pytest.fixture
def native_ring_options(monkeypatch, tmp_path):
    backend = pytest.importorskip("triton.backends.hcu.compiler_hcu")
    try:
        clang = backend.HIPBackend.path_to_rocm_clang()
        p = subprocess.run([clang, "--target=amdgcn-amd-amdhsa", "-nogpulib", "--print-supported-cpus"],
                           capture_output=True, text=True)
    except (OSError, RuntimeError):
        pytest.skip("requires a native gfx936 toolchain")
    if "gfx936" not in p.stdout + p.stderr:
        pytest.skip("native compiler does not support gfx936")
    monkeypatch.setattr(knobs.cache, "dir", str(tmp_path / "cache"))
    monkeypatch.setattr(knobs.hcu, "use_buffer_ops", True)
    monkeypatch.setattr(knobs.hcu, "use_async_copy", False)
    monkeypatch.setattr(knobs.hcu, "use_block_pingpong", False)
    monkeypatch.setattr(knobs.hcu, "use_in_thread_transpose", False)
    return dict(num_warps=8, num_stages=2, schedule_hint="local-prefetch",
                mmac_warps_m=2, mmac_transposed=True, optimize_epilogue=True,
                kpack=1, waves_per_eu=1)


def _source(dtype):
    names = _sync_mm.arg_names
    constants = dict(BLOCK_M=256, BLOCK_N=256, BLOCK_K=16, GROUP_M=4,
                     IS_FP64=False, MASK_EDGES=False, USE_TLE=True,
                     LARGE_TILE_CONFIGS=True, ASYNC_LOAD=False, LOAD_MODE=0,
                     ASYNC_LOAD_MASK=3, stride_ak=1, stride_bn=1, stride_cn=1)
    signature = {n: "*" + dtype if n.endswith("_ptr") else "i32"
                 for n in names if n not in constants}
    attrs = {(i,): [("tt.divisibility", 16)] for i in [0, 1, 2, 3, 4, 6, 8, 10]}
    for i in [0, 1, 2]:
        attrs[(i,)].append(("tt.pointer_range", 32))
    return ASTSource(_sync_mm, signature=signature,
                     constexprs=constants, attrs=attrs)


def _main_loop(asm):
    lines = asm.splitlines()
    labels = {m.group(1): i for i, line in enumerate(lines)
              if (m := re.match(r"^\.LBB0_([0-9]+):", line))}
    loops = []
    for end, line in enumerate(lines):
        m = re.search(r"\bs_(?:cbranch_\w+|branch)\s+\.LBB0_([0-9]+)", line)
        if not m or m.group(1) not in labels or labels[m.group(1)] >= end:
            continue
        body = "\n".join(lines[labels[m.group(1)]:end + 1])
        if (len(re.findall(r"^\.LBB0_\d+:", body, re.M)) == 1
                and len(re.findall(r"\bv_mmac_f32_16x16x16_(?:bf16|f16)\b", body)) == 32):
            loops.append(body)
    assert len(loops) == 1
    return loops[0]


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_sync_two_slot_codegen_and_missing_proof_guard(dtype, native_ring_options, tmp_path):
    target = GPUTarget("hip", "gfx936", 64)
    src = _source(dtype)
    default = triton.compile(src, target=target, options=native_ring_options)
    one = triton.compile(src, target=target,
                         options=dict(native_ring_options, sync_lds_buffers=1))
    two = triton.compile(src, target=target,
                         options=dict(native_ring_options, sync_lds_buffers=2))
    assert default.asm["hsaco"] == one.asm["hsaco"]
    assert one.metadata.shared == 16384
    assert two.metadata.shared == 32768
    assert len(re.findall(r"\bs_barrier\b", _main_loop(one.asm["amdgcn"]))) == 2
    assert len(re.findall(r"\bs_barrier\b", _main_loop(two.asm["amdgcn"]))) == 1
    for kernel in [one, two]:
        assert re.search(r"\.private_segment_fixed_size:\s*0\b", kernel.asm["amdgcn"])

    # Keep the already constructed two-slot ring but remove its proof request.
    # Generic membar must preserve the WAR CTA; an option alone is no proof.
    text, n = re.subn(r"ttg\.hcu\.sync_lds_buffers\s*=\s*2\s*:\s*i32\s*,\s*", "",
                      two.asm["ttgir"])
    assert n == 1
    source = tmp_path / "no_proof_request.ttgir"
    source.write_text(text)
    guarded = triton.compile(str(source), target=target,
                            options=dict(native_ring_options, sync_lds_buffers=2))
    assert guarded.metadata.shared == 32768
    assert len(re.findall(r"\bs_barrier\b", _main_loop(guarded.asm["amdgcn"]))) == 2


@pytest.mark.parametrize("arch,depth", [("gfx936", 0), ("gfx936", 3), ("gfx942", 2)])
def test_sync_two_slot_option_validation(arch, depth):
    backend = pytest.importorskip("triton.backends.hcu.compiler_hcu")
    with pytest.raises(ValueError, match="sync_lds_buffers"):
        backend.HIPBackend(GPUTarget("hip", arch, 64)).parse_options(
            {"sync_lds_buffers": depth})

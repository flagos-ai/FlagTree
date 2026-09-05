"""Compile-only integration coverage for mthreads explicit TLE warp specialization."""

import re

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._C import libtriton
from triton._C.libtriton import ir
from triton.backends.compiler import Language
from triton.compiler import ASTSource
from triton.compiler.errors import CompilationError

from test_tle_utils import mthreads_backend, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _ws_consumer(
    a_smem,
    b_smem,
    full_a,
    full_b,
    empty_a,
    empty_b,
    out,
    K_TILES: tl.constexpr,
    STAGES: tl.constexpr,
    BIAS: tl.constexpr,
):
    PERIOD: tl.constexpr = 2 * STAGES
    FULL_PERIODS: tl.constexpr = K_TILES // PERIOD
    TAIL_TILES: tl.constexpr = K_TILES % PERIOD

    for period_idx in tl.range(0, FULL_PERIODS, num_stages=1):
        for u in tl.static_range(0, PERIOD):
            k_iter = period_idx * PERIOD + u
            tle.gpu.barrier_wait(full_a[u % STAGES], phaseIdx=u // STAGES)
            tle.gpu.barrier_wait(full_b[u % STAGES], phaseIdx=u // STAGES)

            a = tl.load(tle.gpu.local_ptr(a_smem.slot(u % STAGES), (0, 0)))
            b = tl.load(tle.gpu.local_ptr(b_smem.slot(u % STAGES), (0, 0)))
            tl.store(out + 2 * k_iter, a + BIAS)
            tl.store(out + 2 * k_iter + 1, b + BIAS)

            tle.gpu.barrier_arrive(empty_a[u % STAGES], phaseIdx=u // STAGES)
            tle.gpu.barrier_arrive(empty_b[u % STAGES], phaseIdx=u // STAGES)

    for u in tl.static_range(0, TAIL_TILES):
        k_iter = FULL_PERIODS * PERIOD + u
        tle.gpu.barrier_wait(full_a[u % STAGES], phaseIdx=u // STAGES)
        tle.gpu.barrier_wait(full_b[u % STAGES], phaseIdx=u // STAGES)

        a = tl.load(tle.gpu.local_ptr(a_smem.slot(u % STAGES), (0, 0)))
        b = tl.load(tle.gpu.local_ptr(b_smem.slot(u % STAGES), (0, 0)))
        tl.store(out + 2 * k_iter, a + BIAS)
        tl.store(out + 2 * k_iter + 1, b + BIAS)

        tle.gpu.barrier_arrive(empty_a[u % STAGES], phaseIdx=u // STAGES)
        tle.gpu.barrier_arrive(empty_b[u % STAGES], phaseIdx=u // STAGES)


@triton.jit
def _ws_producer(
    a_desc,
    b_desc,
    a_smem,
    b_smem,
    full_a,
    full_b,
    empty_a,
    empty_b,
    out,
    duplicate_out,
    dynamic_k,
    K_TILES: tl.constexpr,
    STAGES: tl.constexpr,
    BIAS: tl.constexpr,
):
    # out, duplicate_out and BIAS intentionally exercise capture reconstruction.
    tl.store(duplicate_out, tl.load(out))
    PERIOD: tl.constexpr = 2 * STAGES
    FULL_PERIODS: tl.constexpr = K_TILES // PERIOD
    TAIL_TILES: tl.constexpr = K_TILES % PERIOD

    for period_idx in tl.range(0, FULL_PERIODS, num_stages=1):
        period_base = period_idx * PERIOD
        for u in tl.static_range(0, PERIOD):
            k_offset = dynamic_k + period_base + u
            tle.gpu.barrier_wait(empty_a[u % STAGES], phaseIdx=u // STAGES)
            tle.gpu.barrier_wait(empty_b[u % STAGES], phaseIdx=u // STAGES)
            tle.gpu.copy(
                a_desc,
                a_smem.slot(u % STAGES),
                (256, 64),
                (0, k_offset),
                barrier=full_a[u % STAGES],
            )
            tle.gpu.copy(
                b_desc,
                b_smem.slot(u % STAGES),
                (64, 256),
                (k_offset, 0),
                barrier=full_b[u % STAGES],
            )

    for u in tl.static_range(0, TAIL_TILES):
        k_offset = dynamic_k + FULL_PERIODS * PERIOD + u
        tle.gpu.barrier_wait(empty_a[u % STAGES], phaseIdx=u // STAGES)
        tle.gpu.barrier_wait(empty_b[u % STAGES], phaseIdx=u // STAGES)
        tle.gpu.copy(
            a_desc,
            a_smem.slot(u % STAGES),
            (256, 64),
            (0, k_offset),
            barrier=full_a[u % STAGES],
        )
        tle.gpu.copy(
            b_desc,
            b_smem.slot(u % STAGES),
            (64, 256),
            (k_offset, 0),
            barrier=full_b[u % STAGES],
        )


@triton.jit
def _ws_dot_consumer(
    a_smem,
    b_smem,
    full_a,
    full_b,
    empty_a,
    empty_b,
    out,
    K_TILES: tl.constexpr,
    STAGES: tl.constexpr,
):
    PERIOD: tl.constexpr = 2 * STAGES
    FULL_PERIODS: tl.constexpr = K_TILES // PERIOD
    TAIL_TILES: tl.constexpr = K_TILES % PERIOD

    acc = tl.zeros((256, 256), dtype=tl.float32)
    for _period_idx in tl.range(0, FULL_PERIODS, num_stages=1):
        for u in tl.static_range(0, PERIOD):
            tle.gpu.barrier_wait(full_a[u % STAGES], phaseIdx=u // STAGES)
            tle.gpu.barrier_wait(full_b[u % STAGES], phaseIdx=u // STAGES)

            a = tl.load(tle.gpu.local_ptr(a_smem.slot(u % STAGES)))
            b = tl.load(tle.gpu.local_ptr(b_smem.slot(u % STAGES)))
            acc = tl.dot(a, b, acc=acc)

            tle.gpu.barrier_arrive(empty_a[u % STAGES], phaseIdx=u // STAGES)
            tle.gpu.barrier_arrive(empty_b[u % STAGES], phaseIdx=u // STAGES)

    for u in tl.static_range(0, TAIL_TILES):
        tle.gpu.barrier_wait(full_a[u % STAGES], phaseIdx=u // STAGES)
        tle.gpu.barrier_wait(full_b[u % STAGES], phaseIdx=u // STAGES)

        a = tl.load(tle.gpu.local_ptr(a_smem.slot(u % STAGES)))
        b = tl.load(tle.gpu.local_ptr(b_smem.slot(u % STAGES)))
        acc = tl.dot(a, b, acc=acc)

        tle.gpu.barrier_arrive(empty_a[u % STAGES], phaseIdx=u // STAGES)
        tle.gpu.barrier_arrive(empty_b[u % STAGES], phaseIdx=u // STAGES)

    offsets = tl.arange(0, 256)[:, None] * 256 + tl.arange(0, 256)[None, :]
    tl.store(out + offsets, acc.to(tl.float16))


@triton.jit
def _ws_integration_kernel(
    a_desc,
    b_desc,
    out,
    dynamic_k,
    K_TILES: tl.constexpr,
    STAGES: tl.constexpr,
    BIAS: tl.constexpr,
):
    a_smem = tle.gpu.alloc(
        (STAGES, 256, 64),
        dtype=tl.float16,
        nv_mma_shared_layout=False,
    )
    b_smem = tle.gpu.alloc(
        (STAGES, 64, 256),
        dtype=tl.float16,
        nv_mma_shared_layout=False,
    )
    full_a = tle.gpu.alloc_barriers(
        STAGES,
        arrive_count=1,
        init=tle.gpu.PENDING,
        expect_bytes=32768,
    )
    full_b = tle.gpu.alloc_barriers(
        STAGES,
        arrive_count=1,
        init=tle.gpu.PENDING,
        expect_bytes=32768,
    )
    empty_a = tle.gpu.alloc_barriers(
        STAGES,
        arrive_count=16,
        init=tle.gpu.READY,
    )
    empty_b = tle.gpu.alloc_barriers(
        STAGES,
        arrive_count=16,
        init=tle.gpu.READY,
    )

    tle.gpu.warp_specialize(
        [
            (
                _ws_consumer,
                (a_smem, b_smem, full_a, full_b, empty_a, empty_b, out, K_TILES, STAGES, BIAS),
            ),
            (
                _ws_producer,
                (
                    a_desc,
                    b_desc,
                    a_smem,
                    b_smem,
                    full_a,
                    full_b,
                    empty_a,
                    empty_b,
                    out,
                    out,
                    dynamic_k,
                    K_TILES,
                    STAGES,
                    BIAS,
                ),
            ),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _ws_dot_integration_kernel(
    a_desc,
    b_desc,
    out,
    dynamic_k,
    K_TILES: tl.constexpr,
    STAGES: tl.constexpr,
):
    a_smem = tle.gpu.alloc(
        (STAGES, 256, 64),
        dtype=tl.float16,
        nv_mma_shared_layout=False,
    )
    b_smem = tle.gpu.alloc(
        (STAGES, 64, 256),
        dtype=tl.float16,
        nv_mma_shared_layout=False,
    )
    full_a = tle.gpu.alloc_barriers(
        STAGES,
        arrive_count=1,
        init=tle.gpu.PENDING,
        expect_bytes=32768,
    )
    full_b = tle.gpu.alloc_barriers(
        STAGES,
        arrive_count=1,
        init=tle.gpu.PENDING,
        expect_bytes=32768,
    )
    empty_a = tle.gpu.alloc_barriers(
        STAGES,
        arrive_count=16,
        init=tle.gpu.READY,
    )
    empty_b = tle.gpu.alloc_barriers(
        STAGES,
        arrive_count=16,
        init=tle.gpu.READY,
    )

    tle.gpu.warp_specialize(
        [
            (
                _ws_dot_consumer,
                (a_smem, b_smem, full_a, full_b, empty_a, empty_b, out, K_TILES, STAGES),
            ),
            (
                _ws_producer,
                (
                    a_desc,
                    b_desc,
                    a_smem,
                    b_smem,
                    full_a,
                    full_b,
                    empty_a,
                    empty_b,
                    out,
                    out,
                    dynamic_k,
                    K_TILES,
                    STAGES,
                    1,
                ),
            ),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


def _compile_ws_integration(stages, k_tiles):
    target, backend = mthreads_backend()
    options = backend.parse_options({"num_warps": 16, "num_stages": 1})
    assert options.num_warps == 16
    assert options.num_stages == 1
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    src = ASTSource(
        fn=_ws_integration_kernel,
        signature={
            "a_desc": "tensordesc<fp16[256, 64]>",
            "b_desc": "tensordesc<fp16[64, 256]>",
            "out": "*fp16",
            "dynamic_k": "i32",
            "K_TILES": "constexpr",
            "STAGES": "constexpr",
            "BIAS": "constexpr",
        },
        constexprs={"K_TILES": k_tiles, "STAGES": stages, "BIAS": 1},
        attrs={(0, ): [["musa.tme_tail_divisibility", 4]], (1, ): [["musa.tme_tail_divisibility", 4]]},
    )
    module = src.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )
    compiler_stages = {}
    backend.add_stages(compiler_stages, options, Language.TRITON)
    metadata = {}
    module = compiler_stages["ttir"](module, metadata)
    ttir = module.str_nodebug()
    module = compiler_stages["ttgir"](module, metadata)
    ttgir = module.str_nodebug()

    pm = ir.pass_manager(context)
    libtriton.mthreads.passes.ttgpuir.add_allocate_shared_memory(pm, 31)
    pm.run(module, "allocate_ws_integration_shared_memory")
    return ttir, ttgir, module.str_nodebug()


def _compile_ws_dot_integration(stages):
    target, backend = mthreads_backend()
    options = backend.parse_options({"num_warps": 16, "num_stages": 1})
    assert options.num_warps == 16
    assert options.num_stages == 1
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    src = ASTSource(
        fn=_ws_dot_integration_kernel,
        signature={
            "a_desc": "tensordesc<fp16[256, 64]>",
            "b_desc": "tensordesc<fp16[64, 256]>",
            "out": "*fp16",
            "dynamic_k": "i32",
            "K_TILES": "constexpr",
            "STAGES": "constexpr",
        },
        constexprs={"K_TILES": 16, "STAGES": stages},
        attrs={(0, ): [["musa.tme_tail_divisibility", 4]], (1, ): [["musa.tme_tail_divisibility", 4]]},
    )
    module = src.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )
    compiler_stages = {}
    backend.add_stages(compiler_stages, options, Language.TRITON)
    metadata = {}
    module = compiler_stages["ttir"](module, metadata)
    ttir = module.str_nodebug()
    module = compiler_stages["ttgir"](module, metadata)
    ttgir = module.str_nodebug()

    pm = ir.pass_manager(context)
    libtriton.mthreads.passes.ttgpuir.add_allocate_shared_memory(pm, 31)
    pm.run(module, "allocate_ws_dot_integration_shared_memory")
    allocated = module.str_nodebug()

    pm = ir.pass_manager(context)
    libtriton.passes.convert.add_scf_to_cf(pm)
    libtriton.passes.convert.add_index_to_llvmir(pm)
    libtriton.mthreads.passes.ttgpuir.add_mtgpu_to_llvm(pm, 31)
    libtriton.mthreads.passes.ttgpuir.add_to_llvmir(pm, 31)
    libtriton.mthreads.passes.ttgpuir.add_tle_lower_warp_specialize(pm)
    libtriton.passes.convert.add_scf_to_cf(pm)
    pm.run(module, "lower_ws_dot_integration_to_llvm_cfg")
    return ttir, ttgir, allocated, module.str_nodebug()


def _shared_memory_bytes(ir_text):
    match = re.search(r"ttg\.shared = (\d+) : i32", ir_text)
    assert match, "shared-memory allocation metadata is missing"
    return int(match.group(1))


@pytest.mark.parametrize("stages,k_tiles", [(1, 16), (2, 16), (2, 10)])
def test_mthreads_tle_warp_specialize_integration_contract(stages, k_tiles):
    _, ttgir, allocated = _compile_ws_integration(stages, k_tiles)
    assert "ttg.warp_specialize" in ttgir
    assert "ttmg.async_tme_copy_global_to_local" in ttgir
    # Static dispatch must not add a shared-memory capture mailbox.
    assert _shared_memory_bytes(allocated) == stages * 65536


@pytest.mark.parametrize("stages", [1, 2])
def test_mthreads_tle_warp_specialize_dot_pipeline_resources(stages):
    _, ttgir, allocated, late = _compile_ws_dot_integration(stages)
    assert "ttmg.squad_dot" in ttgir
    assert "ttg.warp_specialize" not in late
    assert "llvm.musa.sqmma." in late
    assert "llvm.musa.barrier0" not in late
    # Bound the fixed probe's SMEM without pinning allocator offsets, scratch
    # buffer counts, or a particular SQMMA decomposition. Improvements are OK.
    assert 0 < _shared_memory_bytes(allocated) <= stages * (65536 + 131072)


def test_mthreads_tle_warp_specialize_stage_three_remains_deferred():
    with pytest.raises(CompilationError, match="Shape element 0 must be a power of 2"):
        _compile_ws_integration(3, 16)

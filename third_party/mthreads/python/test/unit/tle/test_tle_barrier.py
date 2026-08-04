"""Compile-only coverage for mthreads TLE barrier arrays and mbarriers."""

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

from test_tle_utils import compile_to_ttir, mthreads_backend, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _static_barrier_kernel(STAGES: tl.constexpr):
    full = tle.gpu.alloc_barriers(
        STAGES,
        arrive_count=1,
        init=tle.gpu.PENDING,
        expect_bytes=32768,
    )
    empty = tle.gpu.alloc_barriers(
        STAGES,
        arrive_count=16,
        init=tle.gpu.READY,
    )
    for slot in tl.static_range(0, STAGES):
        tle.gpu.barrier_wait(full[slot], phaseIdx=slot)
        tle.gpu.barrier_wait(empty[slot], phaseIdx=slot)
        tle.gpu.barrier_arrive(empty[slot], phaseIdx=slot)


@triton.jit
def _dynamic_barrier_kernel(stage, phase):
    bars = tle.gpu.alloc_barriers(
        2,
        arrive_count=16,
        init=tle.gpu.READY,
    )
    slot = bars[stage]
    tle.gpu.barrier_wait(slot, phaseIdx=phase)
    tle.gpu.barrier_arrive(slot, phaseIdx=phase)


@triton.jit
def _singleton_barrier_kernel():
    bar = tle.gpu.alloc_barrier()
    tle.gpu.barrier_wait(bar, phaseIdx=0)
    tle.gpu.barrier_arrive(bar, phaseIdx=0)


@triton.jit
def _named_barrier_kernel():
    bars = tle.gpu.alloc_barriers(2)
    tle.gpu.barrier_wait(bars[0])


@triton.jit
def _named_barrier_arrive_kernel():
    bars = tle.gpu.alloc_barriers(2)
    tle.gpu.barrier_arrive(bars[0])


@triton.jit
def _invalid_num_barriers_kernel():
    tle.gpu.alloc_barriers(0)


@triton.jit
def _too_many_barriers_kernel():
    tle.gpu.alloc_barriers(64)


@triton.jit
def _invalid_arrive_count_kernel():
    tle.gpu.alloc_barriers(1, arrive_count=0)


@triton.jit
def _invalid_expect_bytes_kernel():
    tle.gpu.alloc_barriers(1, expect_bytes=0)


@triton.jit
def _block_slot_kernel():
    bars = tle.gpu.alloc_barriers(2)
    slot = bars[tl.arange(0, 2)]
    tle.gpu.barrier_wait(slot, phaseIdx=0)


@triton.jit
def _non_integer_slot_kernel():
    bars = tle.gpu.alloc_barriers(2)
    slot = bars[0.0]
    tle.gpu.barrier_wait(slot, phaseIdx=0)


@triton.jit
def _out_of_bounds_slot_kernel(STAGES: tl.constexpr, SLOT: tl.constexpr):
    bars = tle.gpu.alloc_barriers(STAGES)
    tle.gpu.barrier_wait(bars[SLOT], phaseIdx=0)


@triton.jit
def _non_integer_phase_kernel():
    bars = tle.gpu.alloc_barriers(1)
    tle.gpu.barrier_wait(bars[0], phaseIdx=0.0)


def _extract_barrier_ops(ttir):
    allocs = re.findall(r"musa_tle\.barrier\.alloc", ttir)
    indices = re.findall(r"musa_tle\.barrier\.index", ttir)
    waits = re.findall(r"musa_tle\.barrier\.wait", ttir)
    arrives = re.findall(r"musa_tle\.barrier\.arrive", ttir)
    return len(allocs), len(indices), len(waits), len(arrives)


def _compile_barrier_ir(fn, signature, constexprs=None):
    target, backend = mthreads_backend()
    options = backend.parse_options({"num_stages": 1})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs or {})
    module = src.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )
    stages = {}
    backend.add_stages(stages, options, Language.TRITON)
    metadata = {}
    module = stages["ttir"](module, metadata)
    ttir = module.str_nodebug()
    module = stages["ttgir"](module, metadata)
    ttgir = module.str_nodebug()

    pm = ir.pass_manager(context)
    libtriton.mthreads.passes.ttgpuir.add_allocate_shared_memory(pm, 31)
    pm.run(module, "allocate_barrier_shared_memory")
    return ttir, ttgir, module.str_nodebug()


def _barrier_index_constants(ttir):
    constants = {
        name: int(value)
        for name, value in re.findall(
            r"(%[-\w.]+)\s*=\s*(?:arith\.)?constant\s+(-?\d+)\s*:\s*i32",
            ttir,
        )
    }
    values = []
    for index in re.findall(
            r"musa_tle\.barrier\.index\s+%[-\w.]+\[(%[-\w.]+)\]",
            ttir,
    ):
        values.append(constants.get(index))
    return values


def _assert_static_barrier_ir(ttir, stages):
    allocs, indices, waits, arrives = _extract_barrier_ops(ttir)
    assert allocs == 2, ttir
    assert 2 * stages <= indices <= 3 * stages, ttir
    assert waits == 2 * stages, ttir
    assert arrives == stages, ttir
    assert not re.search(r"(?<!musa_)tle\.barrier\.", ttir), ttir
    assert "tle.wgmma_pipeline_mode" not in ttir, ttir

    assert ttir.count("expect_bytes = 32768") == 1, ttir
    assert ttir.count("init_polarity = 0") == 1, ttir
    assert ttir.count("init_polarity = 1") == 1, ttir
    assert ttir.count("num_barriers = %d" % stages) == 2, ttir
    assert "arrive_count = 1" in ttir, ttir
    assert "arrive_count = 16" in ttir, ttir
    assert "ttg.memdesc_index" not in ttir, ttir
    assert "ttg.local_alloc" not in ttir, ttir
    assert "!ttg.memdesc" not in ttir, ttir
    assert "#smem" not in ttir, ttir
    assert re.search(r"%[-\w.]+\s*=\s*musa_tle\.barrier\.alloc", ttir), ttir

    indices = _barrier_index_constants(ttir)
    # TTGIR CSE may share the empty-barrier slot between wait and arrive.
    assert 2 * stages <= len(indices) <= 3 * stages, ttir
    assert set(indices) == set(range(stages)), ttir


@pytest.mark.parametrize("stages", [1, 2])
def test_mthreads_tle_barrier_static_arrays_emit_mbarrier_ir(stages):
    ir_modules = _compile_barrier_ir(
        _static_barrier_kernel,
        {"STAGES": "constexpr"},
        {"STAGES": stages},
    )
    for ir_text in ir_modules:
        _assert_static_barrier_ir(ir_text, stages)
    assert "ttg.shared = 0 : i32" in ir_modules[-1], ir_modules[-1]


def test_mthreads_tle_barrier_preserves_dynamic_slot_and_phase():
    ir_modules = _compile_barrier_ir(
        _dynamic_barrier_kernel,
        {"stage": "i32", "phase": "i32"},
    )
    for ir_text in ir_modules:
        allocs, indices, waits, arrives = _extract_barrier_ops(ir_text)
        assert (allocs, indices, waits, arrives) == (1, 1, 1, 1), ir_text
        assert not re.search(r"(?<!musa_)tle\.barrier\.", ir_text), ir_text
        assert "ttg.memdesc_index" not in ir_text, ir_text
        assert "musa_tle.barrier.index" in ir_text, ir_text
        assert "musa_tle.barrier.wait" in ir_text, ir_text
        assert "musa_tle.barrier.arrive" in ir_text, ir_text
        assert re.search(r"musa_tle\.barrier\.index\s+%[-\w.]+\[%arg0\]", ir_text), ir_text
        assert re.search(r"arith\.andi\s+%arg1,", ir_text), ir_text
        assert "arith.xori" in ir_text, ir_text
        assert "num_barriers = 2" in ir_text, ir_text
    assert "ttg.shared = 0 : i32" in ir_modules[-1], ir_modules[-1]


def test_mthreads_tle_singleton_barrier_uses_indexed_slot():
    for ir_text in _compile_barrier_ir(_singleton_barrier_kernel, {}):
        allocs, indices, waits, arrives = _extract_barrier_ops(ir_text)
        assert (allocs, indices, waits, arrives) == (1, 1, 1, 1), ir_text
        assert _barrier_index_constants(ir_text) == [0], ir_text
        assert "!ttg.memdesc" not in ir_text, ir_text
        assert "ttg.local_alloc" not in ir_text, ir_text


def test_mthreads_tle_barrier_bindings_are_backend_local():
    _, backend = mthreads_backend()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    builder = ir.builder(context)
    for name in (
            "create_barrier_alloc",
            "create_barrier_wait_mbarrier",
            "create_barrier_arrive_mbarrier",
            "create_barrier_wait_named",
            "create_barrier_arrive_named",
    ):
        assert hasattr(builder, name)


@pytest.mark.parametrize("kernel", [_named_barrier_kernel, _named_barrier_arrive_kernel])
def test_mthreads_tle_barrier_named_path_has_stable_diagnostic(kernel):
    with pytest.raises(
            CompilationError,
            match="mthreads TLE named barrier backend is unsupported; phaseIdx is required",
    ):
        compile_to_ttir(kernel, {})


@pytest.mark.parametrize(
    "kernel,diagnostic",
    [
        (_invalid_num_barriers_kernel, "num_barriers must be positive"),
        (_too_many_barriers_kernel, "mthreads TLE barrier allocation exceeds the 63 hardware barrier id limit"),
        (_invalid_arrive_count_kernel, "arrive_count must be positive"),
        (_invalid_expect_bytes_kernel, "expect_bytes must be positive when provided"),
        (_block_slot_kernel, "barrier index must be a scalar integer"),
        (_non_integer_slot_kernel, "barrier index must be integer"),
        (_non_integer_phase_kernel, "barrier phaseIdx must be integer"),
    ],
)
def test_mthreads_tle_barrier_invalid_frontend_inputs_have_stable_diagnostics(kernel, diagnostic):
    with pytest.raises(CompilationError, match=diagnostic):
        compile_to_ttir(kernel, {})


@pytest.mark.parametrize(
    "stages,slot",
    [(1, -1), (1, 1), (2, -1), (2, 2)],
)
def test_mthreads_tle_barrier_rejects_constant_slot_out_of_bounds(stages, slot):
    with pytest.raises(
            CompilationError,
            match=rf"barrier index {slot} out of bounds for {stages} barriers",
    ):
        compile_to_ttir(
            _out_of_bounds_slot_kernel,
            {"STAGES": "constexpr", "SLOT": "constexpr"},
            {"STAGES": stages, "SLOT": slot},
        )

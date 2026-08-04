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
def _four_group_barrier_kernel(STAGES: tl.constexpr):
    full_a = tle.gpu.alloc_barriers(STAGES, arrive_count=1, init=tle.gpu.PENDING, expect_bytes=32768)
    full_b = tle.gpu.alloc_barriers(STAGES, arrive_count=1, init=tle.gpu.PENDING, expect_bytes=32768)
    empty_a = tle.gpu.alloc_barriers(STAGES, arrive_count=16, init=tle.gpu.READY)
    empty_b = tle.gpu.alloc_barriers(STAGES, arrive_count=16, init=tle.gpu.READY)
    for slot in tl.static_range(0, STAGES):
        tle.gpu.barrier_wait(full_a[slot], phaseIdx=slot)
        tle.gpu.barrier_wait(full_b[slot], phaseIdx=slot)
        tle.gpu.barrier_arrive(empty_a[slot], phaseIdx=slot)
        tle.gpu.barrier_arrive(empty_b[slot], phaseIdx=slot)


@triton.jit
def _exhausted_barrier_kernel():
    bars_0 = tle.gpu.alloc_barriers(16)
    bars_1 = tle.gpu.alloc_barriers(16)
    bars_2 = tle.gpu.alloc_barriers(16)
    bars_3 = tle.gpu.alloc_barriers(16)
    tle.gpu.barrier_wait(bars_0[0], phaseIdx=0)
    tle.gpu.barrier_wait(bars_1[0], phaseIdx=0)
    tle.gpu.barrier_wait(bars_2[0], phaseIdx=0)
    tle.gpu.barrier_wait(bars_3[0], phaseIdx=0)


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


@triton.jit
def _completion_wait_missing_phase_kernel():
    bar = tle.gpu.alloc_barrier(expect_bytes=32768)
    tle.gpu.barrier_wait(bar)


@triton.jit
def _completion_arrive_missing_phase_kernel():
    bar = tle.gpu.alloc_barrier(expect_bytes=32768)
    tle.gpu.barrier_arrive(bar)


@triton.jit
def _ready_wait_missing_phase_kernel():
    bar = tle.gpu.alloc_barrier(init=tle.gpu.READY)
    tle.gpu.barrier_wait(bar)


@triton.jit
def _ready_arrive_missing_phase_kernel():
    bar = tle.gpu.alloc_barrier(init=tle.gpu.READY)
    tle.gpu.barrier_arrive(bar)


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


def _constants(ir_text):
    return {
        name: int(value)
        for name, value in re.findall(
            r"(%[-\w.]+)\s*=\s*(?:arith\.)?constant\s+(-?\d+)\s*:\s*i32",
            ir_text,
        )
    }


def _init_arrivals(ir_text):
    constants = _constants(ir_text)
    return [(constants[bar_id], constants[arrive_count], constants[phase])
            for bar_id, arrive_count, phase in re.findall(
                r"ttmg\.init_arrival\s+(%[-\w.]+),\s*(%[-\w.]+),\s*(%[-\w.]+)",
                ir_text,
            )]


def _assert_lowered_static_barrier_ir(ir_text, stages):
    allocs, indices, waits, arrives = _extract_barrier_ops(ir_text)
    assert (allocs, indices, waits, arrives) == (0, 0, 2 * stages, stages), ir_text
    assert _init_arrivals(ir_text) == [
        *[(slot + 1, 1, 0) for slot in range(stages)],
        *[(stages + slot + 1, 16, 1) for slot in range(stages)],
    ], ir_text
    assert f"musa.max_bar_id = {2 * stages}" in ir_text, ir_text
    assert "musa.next_bar_id" not in ir_text, ir_text
    assert "ttmg.bar_record" in ir_text, ir_text
    assert "ttg.memdesc_index" not in ir_text, ir_text
    assert "ttg.local_alloc" not in ir_text, ir_text
    assert "!ttg.memdesc" not in ir_text, ir_text


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
    ttir, ttgir, allocated = _compile_barrier_ir(
        _static_barrier_kernel,
        {"STAGES": "constexpr"},
        {"STAGES": stages},
    )
    _assert_static_barrier_ir(ttir, stages)
    _assert_lowered_static_barrier_ir(ttgir, stages)
    _assert_lowered_static_barrier_ir(allocated, stages)
    assert "ttg.shared = 0 : i32" in allocated, allocated


@pytest.mark.parametrize("stages", [1, 2])
def test_mthreads_tle_four_barrier_groups_receive_contiguous_hardware_ids(stages):
    ttir, ttgir, allocated = _compile_barrier_ir(
        _four_group_barrier_kernel,
        {"STAGES": "constexpr"},
        {"STAGES": stages},
    )
    assert _extract_barrier_ops(ttir)[0] == 4, ttir
    expected = [
        *[(1 + slot, 1, 0) for slot in range(stages)],
        *[(1 + stages + slot, 1, 0) for slot in range(stages)],
        *[(1 + 2 * stages + slot, 16, 1) for slot in range(stages)],
        *[(1 + 3 * stages + slot, 16, 1) for slot in range(stages)],
    ]
    for ir_text in (ttgir, allocated):
        assert _init_arrivals(ir_text) == expected, ir_text
        assert _extract_barrier_ops(ir_text)[0:2] == (0, 0), ir_text
        assert f"musa.max_bar_id = {4 * stages}" in ir_text, ir_text
        assert "musa.next_bar_id" not in ir_text, ir_text
    assert "ttg.shared = 0 : i32" in allocated, allocated


def _lower_backend_barrier_fixture(tmp_path, stages, existing_max=None):
    function_attrs = ""
    if existing_max is not None:
        function_attrs = f" attributes {{musa.max_bar_id = {existing_max} : i32}}"
    allocs = "\n".join(f"    %{index} = musa_tle.barrier.alloc "
                       f"{{arrive_count = {1 if index < 2 else 16} : i32, "
                       f"init_polarity = {0 if index < 2 else 1} : i32, "
                       f"num_barriers = {stages} : i32}}" for index in range(4))
    fixture = f"""module {{
  tt.func public @barrier_fixture(){function_attrs} {{
{allocs}
    tt.return
  }}
}}
"""
    fixture_path = tmp_path / "barrier_fixture.ttgir"
    fixture_path.write_text(fixture)

    _, backend = mthreads_backend()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = ir.parse_mlir_module(str(fixture_path), context)
    pm = ir.pass_manager(context)
    libtriton.mthreads.passes.ttgpuir.add_tle_lower_barrier_allocations(pm)
    libtriton.mthreads.passes.ttgpuir.add_finalize_barriers(pm)
    pm.run(module, "lower_backend_barrier_fixture")
    return module.str_nodebug()


def test_mthreads_tle_stage_three_barriers_use_backend_local_fixture(tmp_path):
    lowered = _lower_backend_barrier_fixture(tmp_path, stages=3)
    assert _init_arrivals(lowered) == [
        *[(1 + slot, 1, 0) for slot in range(3)],
        *[(4 + slot, 1, 0) for slot in range(3)],
        *[(7 + slot, 16, 1) for slot in range(3)],
        *[(10 + slot, 16, 1) for slot in range(3)],
    ], lowered
    assert "musa.max_bar_id = 12" in lowered, lowered
    assert _extract_barrier_ops(lowered)[0:2] == (0, 0), lowered


def test_mthreads_tle_barriers_follow_existing_musa_reservations(tmp_path):
    lowered = _lower_backend_barrier_fixture(tmp_path, stages=1, existing_max=5)
    assert _init_arrivals(lowered) == [
        (6, 1, 0),
        (7, 1, 0),
        (8, 16, 1),
        (9, 16, 1),
    ], lowered
    assert "musa.max_bar_id = 9" in lowered, lowered


def test_mthreads_tle_barrier_id_exhaustion_has_stable_diagnostic(capfd):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _compile_barrier_ir(_exhausted_barrier_kernel, {})
    stderr = capfd.readouterr().err
    assert ("mthreads TLE barrier allocation exhausted hardware barrier ids: "
            "cannot reserve 64 additional ids in [1, 63]") in stderr


def test_mthreads_tle_barrier_preserves_dynamic_slot_and_phase():
    ttir, ttgir, allocated = _compile_barrier_ir(
        _dynamic_barrier_kernel,
        {"stage": "i32", "phase": "i32"},
    )
    assert _extract_barrier_ops(ttir) == (1, 1, 1, 1), ttir
    assert re.search(r"musa_tle\.barrier\.index\s+%[-\w.]+\[%arg0\]", ttir), ttir
    assert "num_barriers = 2" in ttir, ttir
    for ir_text in (ttgir, allocated):
        assert _extract_barrier_ops(ir_text) == (0, 0, 1, 1), ir_text
        assert "arith.addi" in ir_text, ir_text
        assert re.search(r"arith\.andi\s+%arg1,", ir_text), ir_text
        assert "arith.xori" in ir_text, ir_text
        assert _init_arrivals(ir_text) == [(1, 16, 1), (2, 16, 1)], ir_text
    assert "ttg.shared = 0 : i32" in allocated, allocated


def test_mthreads_tle_singleton_barrier_uses_indexed_slot():
    ttir, ttgir, allocated = _compile_barrier_ir(_singleton_barrier_kernel, {})
    assert _extract_barrier_ops(ttir) == (1, 1, 1, 1), ttir
    assert _barrier_index_constants(ttir) == [0], ttir
    for ir_text in (ttgir, allocated):
        assert _extract_barrier_ops(ir_text) == (0, 0, 1, 1), ir_text
        assert _init_arrivals(ir_text) == [(1, 1, 0)], ir_text
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
            "add_tle_lower_barrier_allocations",
    ):
        owner = (libtriton.mthreads.passes.ttgpuir if name.startswith("add_") else builder)
        assert hasattr(owner, name)


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
        (
            _completion_wait_missing_phase_kernel,
            "barrier_wait on a barrier with expect_bytes requires phaseIdx",
        ),
        (
            _completion_arrive_missing_phase_kernel,
            "barrier_arrive on a barrier with expect_bytes requires phaseIdx",
        ),
        (
            _ready_wait_missing_phase_kernel,
            "barrier_wait without phaseIdx selects named barrier, which does not support READY",
        ),
        (
            _ready_arrive_missing_phase_kernel,
            "barrier_arrive without phaseIdx selects named barrier, which does not support READY",
        ),
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

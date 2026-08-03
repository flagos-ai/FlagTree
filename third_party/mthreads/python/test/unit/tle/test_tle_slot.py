"""Compile-only coverage for mthreads TLE buffered_tensor.slot()."""

import re

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.compiler import ASTSource
from triton.compiler.errors import CompilationError

from test_tle_utils import musa_target, mthreads_backend, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _slot_a_kernel(value_ptr, out_ptr, STAGES: tl.constexpr):
    value = tl.load(value_ptr)
    smem = tle.gpu.alloc(
        (STAGES, 256, 64),
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    for slot_index in tl.static_range(0, STAGES):
        slot = smem.slot(slot_index)
        ptr = tle.gpu.local_ptr(slot, (0, 0))
        tl.store(ptr, value)
        tl.store(out_ptr + slot_index, tl.load(ptr))


@triton.jit
def _slot_b_kernel(value_ptr, out_ptr, STAGES: tl.constexpr):
    value = tl.load(value_ptr)
    smem = tle.gpu.alloc(
        (STAGES, 64, 256),
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    for slot_index in tl.static_range(0, STAGES):
        slot = smem.slot(slot_index)
        ptr = tle.gpu.local_ptr(slot, (0, 0))
        tl.store(ptr, value)
        tl.store(out_ptr + slot_index, tl.load(ptr))


@triton.jit
def _slot_dynamic_kernel(value_ptr, out_ptr, stage):
    value = tl.load(value_ptr)
    smem = tle.gpu.alloc(
        (2, 16, 16),
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    slot = smem.slot(stage)
    ptr = tle.gpu.local_ptr(slot, (0, 0))
    tl.store(ptr, value)
    tl.store(out_ptr, tl.load(ptr))


@triton.jit
def _slot_constant_kernel(value_ptr, out_ptr, STAGES: tl.constexpr, SLOT: tl.constexpr):
    value = tl.load(value_ptr)
    smem = tle.gpu.alloc(
        (STAGES, 16, 16),
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    slot = smem.slot(SLOT)
    ptr = tle.gpu.local_ptr(slot, (0, 0))
    tl.store(ptr, value)
    tl.store(out_ptr, tl.load(ptr))


def _compile_slot(fn, stages):
    signature = {
        "value_ptr": "*fp16",
        "out_ptr": "*fp16",
        "STAGES": "constexpr",
    }
    src = ASTSource(fn=fn, signature=signature, constexprs={"STAGES": stages})
    return triton.compile(src, target=musa_target(), options={"num_stages": 1})


def _compile_constant_slot(stages, slot):
    src = ASTSource(
        fn=_slot_constant_kernel,
        signature={
            "value_ptr": "*fp16",
            "out_ptr": "*fp16",
            "STAGES": "constexpr",
            "SLOT": "constexpr",
        },
        constexprs={"STAGES": stages, "SLOT": slot},
    )
    return triton.compile(src, target=musa_target(), options={"num_stages": 1})


def _memdesc_index_ops(ir_text):
    constants = {
        name: int(value)
        for name, value in re.findall(
            r"(%[-\w.]+)\s*=\s*(?:arith\.)?constant\s+(-?\d+)\s*:\s*i32",
            ir_text,
        )
    }
    ops = []
    for line in ir_text.splitlines():
        if "ttg.memdesc_index" not in line:
            continue
        match = re.search(
            r"(?P<result>%[-\w.]+)\s*=\s*ttg\.memdesc_index\s+"
            r"(?P<src>%[-\w.]+)\[(?P<index>%[-\w.]+)\]",
            line,
        )
        assert match, line
        index_name = match.group("index")
        ops.append((match, constants.get(index_name), line))
    return ops


def _assert_slot_ir(ir_text, stages, trailing_shape):
    ops = _memdesc_index_ops(ir_text)
    assert len(ops) == stages, ir_text
    assert {value for _, value, _ in ops} == set(range(stages)), ir_text
    assert "#ttg.nvmma_shared" not in ir_text, ir_text
    layouts = dict(re.findall(
        r"(?m)^\s*(#[-\w.]+)\s*=\s*(#ttg\.swizzled_shared<[^\n]+>)\s*$",
        ir_text,
    ))

    source_shape = f"!ttg.memdesc<{stages}x{trailing_shape}xf16"
    result_shape = f"!ttg.memdesc<{trailing_shape}xf16"
    for match, _, line in ops:
        assert source_shape in line, line
        assert result_shape in line, line
        source_type, result_type = line.split(" : ", 1)[1].split(" -> ", 1)
        source_layout = re.findall(r"#[-\w.]+", source_type)[0]
        result_layout = re.findall(r"#[-\w.]+", result_type)[0]
        assert source_layout in layouts, (line, layouts)
        assert result_layout in layouts, (line, layouts)
        assert "order = [2, 1, 0]" in layouts[source_layout], layouts[source_layout]
        assert "order = [1, 0]" in layouts[result_layout], layouts[result_layout]
        assert source_type.count("#smem") == 1, source_type
        assert result_type.count("#smem") == 1, result_type
        result = match.group("result")
        assert re.search(
            rf"(?:\"musa_tle\.local_pointers\"|musa_tle\.local_pointers)\s*\(\s*"
            rf"{re.escape(result)}\b",
            ir_text,
        ), ir_text


@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize(
    "kernel,trailing_shape",
    [(_slot_a_kernel, "256x64"), (_slot_b_kernel, "64x256")],
)
def test_mthreads_tle_slot_emits_one_index_per_constant_stage(kernel, trailing_shape, stages):
    compiled = _compile_slot(kernel, stages)
    assert compiled.metadata.num_stages == 1
    _assert_slot_ir(compiled.asm["ttir"], stages, trailing_shape)
    _assert_slot_ir(compiled.asm["ttgir"], stages, trailing_shape)


def test_mthreads_tle_slot_binding_is_backend_local():
    _, backend = mthreads_backend()
    from triton._C.libtriton import ir

    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    builder = ir.builder(context)
    assert hasattr(builder, "create_memdesc_index")


def test_mthreads_tle_slot_preserves_dynamic_int32_stage():
    src = ASTSource(
        fn=_slot_dynamic_kernel,
        signature={"value_ptr": "*fp16", "out_ptr": "*fp16", "stage": "i32"},
    )
    compiled = triton.compile(src, target=musa_target(), options={"num_stages": 1})
    assert compiled.metadata.num_stages == 1

    for ir_name in ("ttir", "ttgir"):
        ir_text = compiled.asm[ir_name]
        ops = _memdesc_index_ops(ir_text)
        assert len(ops) == 1, ir_text
        match, constant_value, line = ops[0]
        assert constant_value is None, line
        assert match.group("index") == "%stage", line
        assert "!ttg.memdesc<2x16x16xf16" in line, line
        assert "!ttg.memdesc<16x16xf16" in line, line
        result = match.group("result")
        assert re.search(
            rf"(?:\"musa_tle\.local_pointers\"|musa_tle\.local_pointers)\s*\(\s*"
            rf"{re.escape(result)}\b",
            ir_text,
        ), ir_text


@pytest.mark.parametrize(
    "stages,slot",
    [(1, -1), (1, 1), (2, -1), (2, 2)],
)
def test_mthreads_tle_slot_rejects_constant_out_of_bounds(stages, slot):
    with pytest.raises(
            CompilationError,
            match=rf"mthreads TLE memdesc index {slot} out of bounds for leading dimension {stages}",
    ):
        _compile_constant_slot(stages, slot)

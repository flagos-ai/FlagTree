#!/usr/bin/env python3
import os

import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton.compiler.compiler import ASTSource
from triton.compiler.code_generator import ast_to_ttir
from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.backends.ascend.compiler import NPUOptions, ttir_to_linalg


def compile_kernel(kernel, signature, constants):
    """Helper to compile a kernel function to MLIR in linalg dialect."""
    src = ASTSource(kernel, signature, constants)
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    options = NPUOptions()
    ttir = ast_to_ttir(kernel, src, context, options, {}, {})
    metadata = {**options.__dict__}
    return str(ttir_to_linalg(ttir, metadata, options, named_ops=True))


@al.register_custom_op
class my_custom_macro_op:
    core = al.CORE.VECTOR
    pipe = (al.PIPE.PIPE_MTE2, al.PIPE.PIPE_V)
    mode = al.MODE.SIMD
    symbol = "my_custom_macro_func"
    bitcode = os.path.abspath(__file__)

    def __init__(self, x, out=None):
        self.indexing_map = [al.affine_map.get_identity(1)]


@triton.jit
def my_macro_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i, mask=i < n)
    y = tl.load(y_ptr + i, mask=i < n)
    result = al.custom("my_custom_macro_op", x, out=y)
    tl.store(out_ptr + i, result, mask=i < n)


@al.register_custom_op
class my_custom_macro_sync_op:
    core = al.CORE.VECTOR
    pipe = [al.PIPE.PIPE_MTE2, al.PIPE.PIPE_V]
    mode = al.MODE.SIMD
    symbol = "my_custom_macro_sync_func"
    bitcode = os.path.abspath(__file__)
    sync_event_slots = [
        al.SyncEventSlot(
            set_pipe=al.PIPE.PIPE_MTE2,
            wait_pipe=al.PIPE.PIPE_MTE1,
            sync=al.SYNC_HINT.WAIT,
            event=al.EVENT_ID.EVENT_ID1,
        ),
        (al.PIPE.PIPE_M, al.PIPE.PIPE_MTE2, al.SYNC_HINT.SET),
    ]

    def __init__(self, x, out=None):
        self.indexing_map = [al.affine_map.get_identity(1)]


@triton.jit
def my_macro_sync_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i, mask=i < n)
    y = tl.load(y_ptr + i, mask=i < n)
    result = al.custom(
        "my_custom_macro_sync_op",
        x,
        out=y,
    )
    tl.store(out_ptr + i, result, mask=i < n)


def _custom_macro_lines(mlir: str, op_name: str):
    quoted = f'"{op_name}"'
    return [line for line in mlir.splitlines() if "hivm.hir.custom_macro" in line and quoted in line]


def test_custom_macro_op_lowers_to_custom_macro():
    mlir = compile_kernel(
        my_macro_kernel,
        {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
        {"BLOCK": 256},
    )
    assert mlir and len(mlir) > 0
    assert "func.func @my_macro_kernel(" in mlir
    lines = _custom_macro_lines(mlir, "my_custom_macro_op")
    assert lines, "expected hivm.hir.custom_macro for my_custom_macro_op"
    for line in lines:
        assert "hivm.hir.custom_macro" in line
        assert "hivm.hir.custom " not in line
        assert "hivm.pipe_in = #hivm.pipe<PIPE_MTE2>" in line
        assert "hivm.pipe_out = #hivm.pipe<PIPE_V>" in line
        assert "hivm.tcore_type = #hivm.tcore_type<VECTOR>" in line
        assert "hivm.vf_mode = #hivm.vf_mode<SIMD>" in line
        assert "indexing_map = [" in line
        assert "hivm.pipe = #hivm.pipe" not in line


def test_custom_macro_op_sync_event_slots():
    mlir = compile_kernel(
        my_macro_sync_kernel,
        {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
        {"BLOCK": 256},
    )
    assert mlir and len(mlir) > 0
    lines = _custom_macro_lines(mlir, "my_custom_macro_sync_op")
    assert lines, "expected hivm.hir.custom_macro for my_custom_macro_sync_op"
    line = lines[0]
    assert "hivm.pipe_in = #hivm.pipe<PIPE_MTE2>" in line
    assert "hivm.pipe_out = #hivm.pipe<PIPE_V>" in line
    assert "hivm.tcore_type = #hivm.tcore_type<VECTOR>" in line
    assert "hivm.vf_mode = #hivm.vf_mode<SIMD>" in line
    assert "sync_event_slots = [" in line
    assert "#hivm.sync_event_slot<#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_MTE1>, wait, <EVENT_ID1>>" in line
    assert "#hivm.sync_event_slot<#hivm.pipe<PIPE_M>, #hivm.pipe<PIPE_MTE2>, set>" in line
    assert "ins(%" in line and ": tensor<256xf32>" in line
    assert "outs(%" in line


def test_custom_macro_op_does_not_emit_single_pipe_attr():
    mlir = compile_kernel(
        my_macro_kernel,
        {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
        {"BLOCK": 256},
    )
    assert mlir
    for line in _custom_macro_lines(mlir, "my_custom_macro_op"):
        assert "hivm.pipe = #hivm.pipe" not in line


if __name__ == "__main__":
    test_custom_macro_op_lowers_to_custom_macro()
    test_custom_macro_op_sync_event_slots()
    test_custom_macro_op_does_not_emit_single_pipe_attr()
    mlir = compile_kernel(
        my_macro_sync_kernel,
        {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
        {"BLOCK": 256},
    )
    print(f"Generated MLIR ({len(mlir)} chars):\n")
    print(mlir)

#!/usr/bin/env python3
import subprocess
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
    try:
        options = NPUOptions()
        ttir = ast_to_ttir(kernel, src, context, options, {}, {})
        metadata = {
            **options.__dict__,
        }
        linalg = ttir_to_linalg(ttir, metadata, options, named_ops=True)
        return str(linalg)
    except subprocess.CalledProcessError as ex:
        print(ex.stdout.decode())
        print(ex.stderr.decode())
        print("failed")
        return None


# ============== Kernel definitions ==============


@al.register_custom_op
class my_custom_op:
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT

    def __init__(self, x, ptr1, ptr2, offset: tl.int64, other, out=None):
        pass


@triton.jit
def my_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i, mask=i < n)
    y = tl.load(y_ptr + i, mask=i < n)
    result = al.custom("my_custom_op", x, x_ptr, y_ptr + i, (1, 2, 3), [4.1, 5.2], out=y)
    a = 123
    result = al.custom("my_custom_op", x, x_ptr, y_ptr, (a, n), (1.2, 3.4), out=result)
    tl.store(out_ptr + i, result, mask=i < n)


# ============== Pytest tests ==============


def test_custom_op():
    """Test custom op compile to linalg MLIR."""
    mlir = compile_kernel(my_kernel, {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
                          {"BLOCK": 256})
    assert mlir and len(mlir) > 0
    assert "func.func @my_kernel(" in mlir
    assert "hivm.hir.custom" in mlir
    for line in mlir.splitlines():
        if "hivm.hir.custom" in line:
            # custom op name
            assert '"my_custom_op"' in line
            # All tt.ptr converted to memref.
            assert "tt.ptr" not in line
            # Required attributes are set.
            assert "hivm.pipe = #hivm.pipe" in line
            assert "hivm.tcore_type = #hivm.tcore_type" in line
            assert "hivm.vf_mode = #hivm.vf_mode" in line
            # All offset converted to int64.
            assert 'i64, ' in line
            assert 'i32, ' not in line


# ============== Main for manual testing ==============

if __name__ == "__main__":
    mlir = compile_kernel(my_kernel, {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
                          {"BLOCK": 256})
    print(f"✅ Generated MLIR ({len(mlir)} chars):\n")
    print(mlir)

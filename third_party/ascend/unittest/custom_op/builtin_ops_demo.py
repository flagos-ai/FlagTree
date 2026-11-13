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


@triton.jit
def my_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i, mask=i < n)
    y = tl.load(y_ptr + i, mask=i < n)
    index = tl.full([8], 0, tl.int32)
    x = al.custom("__builtin_index_select", x_ptr, index, dim=0, bound=100, end_offset=(2, 2), start_offset=(0, 0),
                  src_stride=(4, 1), out=x)
    tl.store(out_ptr + i, y, mask=i < n)


if __name__ == "__main__":
    src = ASTSource(my_kernel, {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}, {"BLOCK": 256})
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    options = NPUOptions()
    try:
        ttir = ast_to_ttir(my_kernel, src, context, options, {}, {})
        print("=== TTIR ===")
        print(ttir)
        metadata = {
            **options.__dict__,
        }
        linalg = ttir_to_linalg(ttir, metadata, options, named_ops=True)
        print("=== MLIR (linalg) ===")
        print(linalg)
    except subprocess.CalledProcessError as ex:
        print(ex.stdout.decode())
        print(ex.stderr.decode())
        print("failed")

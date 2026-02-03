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
    value = tl.full([8, 64], 0, tl.float32)
    tmp = tl.full([8], 0, tl.float32)
    x = al.custom("__builtin_index_select", x_ptr, index, dim=0, bound=100, end_offset=(2, 2), start_offset=(0, 0),
                  src_stride=(4, 1), out=x)
    al.custom("__builtin_index_put", x_ptr, index, value, dim=0, bound=12, dst_shape=(1, 2, 3), dst_offset=(4, 5, 6),
              dst_stride=(8, 4, 1))
    tmp = al.custom("__builtin_gather_load", y_ptr, index, bound=100, dim=0, src_stride=(1, ), index_shape=(3, ),
                    offsets=(0, ), out=tmp)
    al.custom("__builtin_scatter_store", out_ptr, value, index, 1, 0, (1, ), (2, ), (1, ))
    y = al.custom("__builtin_indirect_load", x_ptr, index, mask=i < n, other=y, out=y)
    al.custom("__builtin_indirect_store", out_ptr, index, value)
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

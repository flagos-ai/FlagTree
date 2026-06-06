#!/usr/bin/env python3
import os
import subprocess
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton.compiler.compiler import ASTSource
from triton.compiler.code_generator import ast_to_ttir
from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.backends.ascend.compiler import NPUOptions, ttir_to_linalg


def _compose_indexing_maps():
    d0 = ascend_ir.affine_expr.get_dim(0)
    d1 = ascend_ir.affine_expr.get_dim(1)
    c4 = ascend_ir.affine_expr.get_constant(4)

    # Base permutation map.
    perm = ascend_ir.affine_map.get_permutation([1, 0])
    # Tile map (row-major tile decomposition).
    tile = ascend_ir.affine_map.get(2, 0, [d0.floordiv(c4), d1.mod(c4)])
    # Compose tile with permutation to build a different output indexing.
    out = tile.compose(perm)

    in0 = ascend_ir.affine_map.get_identity(2)
    in1 = perm
    return [in0, in1, out]


@al.register_custom_op
class compose_indexing_map_custom_op:
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT
    symbol = "compose_indexing_map_custom"
    bitcode = os.path.abspath(__file__)

    def __init__(self, x, y, out=None):
        assert out is not None
        self.indexing_map = _compose_indexing_maps()


@triton.jit
def my_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i, mask=i < n)
    y = tl.load(y_ptr + i, mask=i < n)
    out = al.custom("compose_indexing_map_custom_op", x, y, out=y)
    tl.store(out_ptr + i, out, mask=i < n)


if __name__ == "__main__":
    src = ASTSource(
        my_kernel,
        {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
        {"BLOCK": 256},
    )
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    options = NPUOptions()
    try:
        ttir = ast_to_ttir(my_kernel, src, context, options, {}, {})
        print("=== TTIR ===")
        print(ttir)
        linalg = ttir_to_linalg(ttir, {**options.__dict__}, options, named_ops=True)
        print("=== MLIR (linalg) ===")
        print(linalg)
    except subprocess.CalledProcessError as ex:
        print(ex.stdout.decode())
        print(ex.stderr.decode())
        print("failed")

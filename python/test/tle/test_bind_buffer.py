import triton
import triton.experimental.tle as tle
import triton.language as tl

from triton.compiler.compiler import ASTSource
from triton.compiler.code_generator import ast_to_ttir
from triton._C.libtriton import ir, tle as tle_ir
from triton._C.libtriton.ascend import ir as ascend_ir


class Options:
    num_warps = 4
    num_stages = 3
    num_ctas = 1
    cluster_dims = (1, 1, 1)
    enable_fp_fusion = True
    debug = False


def compile_kernel(kernel, signature, constants):
    """Helper to compile a kernel to MLIR."""
    src = ASTSource(kernel, signature, constants)
    context = ir.context()
    ir.load_dialects(context)
    tle_ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    module = ast_to_ttir(kernel, src, context, Options(), {}, {})
    return str(module)


@triton.jit
def bind_buffer():
    # tle.dsa.ascend.UB is triton.language.extra.extension.cann.core.ascend_address_space.UB
    buffer1 = tle.dsa.alloc(shape=[32, 32], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    tle.dsa.to_tensor(buffer1, writable=True)


if __name__ == "__main__":
    print("=" * 60)
    mlir = compile_kernel(bind_buffer, {}, {})
    print(mlir)

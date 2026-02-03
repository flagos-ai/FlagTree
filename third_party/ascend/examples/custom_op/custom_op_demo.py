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


@al.register_custom_op
class min_custom_op:
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_MTE2
    mode = al.MODE.SIMD


@al.register_custom_op
class simple_custom_op:
    # name is optional, use class name by default.
    name = 'simple_custom_op'

    # required attributes.
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT

    # __init__ method is optional, but it can be used for better user experience
    # when provided. for example, you can validate arguments here.
    def __init__(self, x, y, dim=0, out=None):
        assert x.shape == y.shape, "x and y should have same shape"
        assert isinstance(dim, int), "dim should be const integer"
        assert out, "out is required"


@al.register_custom_op
class _example_custom_op:
    name = 'example_custom_op'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT

    def __init__(self, src, index, offset: tl.int64, axis, out=None):
        # support validate arguments in __init__ method.
        assert isinstance(src, tl.tensor), "src should be tensor"
        assert index.dtype.is_int(), "index should be integer tensor"
        assert isinstance(offset, int), "offset should be integer"
        assert isinstance(axis, int), "axis should be integer"

        # support multi-output by using tuple or list.
        assert isinstance(out, tuple) and len(out) == 2, "out should be tuple of 2 items"

        # setup the symbol name of the function that will be called at runtime.
        rank = len(index.shape)
        self.symbol = f"{self.name}_{rank}d_{src.dtype.cname}_{index.dtype.cname}"

        # setup source and compile command if it is implemented by user source code.
        self.source = f"workspace/example_custom_op_impl.cce"
        self.compile = "bisheng -O2 -std=c++17 -o $@ -c $<"

        # dynamic set argument type.
        self.arg_type['axis'] = index.dtype


@al.builtin
def example_op(src, index, offset, axis, _builder=None):
    # you can wrap a custom op as a builtin operation,
    # output can be provided here to make it easy to use.
    x = tl.semantic.full(src.shape, 0, tl.float32, _builder)
    y = tl.semantic.full(index.shape, 0, tl.float32, _builder)
    return al.custom_semantic(_example_custom_op.name, src, index, offset, axis, out=(x, y), _builder=_builder)


@triton.jit
def my_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i, mask=i < n)
    y = tl.load(y_ptr + i, mask=i < n)
    y = al.custom("min_custom_op", x, x_ptr, y_ptr + i, al.int64(0), (1, 2, 3), [4.1, 5.2], out=y)
    y = al.custom("simple_custom_op", x, y, dim=1, out=y)
    index = tl.full((2, 3), 0, tl.int64)
    x, y = al.custom("example_custom_op", x, index, offset=1, axis=0, out=(x, y))
    result, _ = example_op(x, index, offset=2, axis=1)
    tl.store(out_ptr + i, result, mask=i < n)


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

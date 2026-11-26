import pytest

import triton
import triton.language as tl
from flagtree import _flagprism
from triton._C.libtriton import buffer_ir, ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, make_backend


@pytest.fixture(scope="module", autouse=True)
def assign_npu():
    """TTIR generation only needs a compiler context."""
    yield


def test_permute_constexpr_order():

    @triton.jit
    def permute_order(out, ORDER: tl.constexpr):
        offsets = tl.arange(0, 8)
        tile = tl.reshape(offsets, (2, 4))
        permuted = tl.permute(tile, ORDER)
        tl.store(out + offsets, tl.reshape(permuted, (8, )))

    target = GPUTarget("npu", "Ascend910B4", 0)
    backend = make_backend(target)
    options = backend.parse_options({})
    source = ASTSource(permute_order, {"out": "*i32", "ORDER": "constexpr"}, {"ORDER": (1, 0)})
    context = ir.context()
    ir.load_dialects(context)
    buffer_ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    backend.load_dialects(context)
    _flagprism.load_dialects(context)
    module = source.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(),
                            context)
    ttir = str(module)
    assert "tt.trans" in ttir
    assert "tensor<4x2xi32>" in ttir

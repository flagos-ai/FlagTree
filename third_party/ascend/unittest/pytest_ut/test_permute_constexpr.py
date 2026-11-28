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


@pytest.mark.parametrize("form", ["permute_tuple", "permute_axes", "trans_tuple"])
def test_permute_constexpr_order(form):

    @triton.jit
    def permute_order(out, ORDER: tl.constexpr, D0: tl.constexpr, D1: tl.constexpr, FORM: tl.constexpr):
        offsets = tl.arange(0, 8)
        tile = tl.reshape(offsets, (2, 4))
        if FORM == "permute_axes":
            permuted = tl.permute(tile, D0, D1)
        elif FORM == "trans_tuple":
            permuted = tl.trans(tile, ORDER)
        else:
            permuted = tl.permute(tile, ORDER)
        tl.store(out + offsets, tl.reshape(permuted, (8, )))

    target = GPUTarget("npu", "Ascend910B4", 0)
    backend = make_backend(target)
    options = backend.parse_options({})
    signature = {"out": "*i32", "ORDER": "constexpr", "D0": "constexpr", "D1": "constexpr", "FORM": "constexpr"}
    source = ASTSource(permute_order, signature, {"ORDER": (1, 0), "D0": 1, "D1": 0, "FORM": form})
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
    assert "order = array<i32: 1, 0>" in ttir
    assert "tensor<4x2xi32>" in ttir

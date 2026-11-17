"""MetaX atomic type checks accept shared-memory pointers."""

import pytest
from triton._C import libtriton
from triton._common_ir import ENABLED
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import make_backend

pytestmark = pytest.mark.skipif(
    not ENABLED or not hasattr(libtriton, "metax"),
    reason="requires a MetaX CommonIR build",
)


@pytest.mark.parametrize("address_space", [1, 3])
@pytest.mark.parametrize("tensor", [False, True])
@pytest.mark.parametrize("operation", ["atomic_rmw", "atomic_cas"])
def test_atomic_pointer_address_space(address_space, tensor, operation, tmp_path):
    pointer = f"!tt.ptr<i32, {address_space}>"
    value, mask = "i32", "i1"
    if tensor:
        pointer, value, mask = f"tensor<16x{pointer}>", "tensor<16xi32>", "tensor<16xi1>"
    if operation == "atomic_rmw":
        operands = "%ptr, %value, %mask"
        types = f"{pointer}, {value}, {mask}"
        properties = "atomic_rmw_op = 4 : i32, scope = 2 : i32, sem = 1 : i32"
    else:
        operands = "%ptr, %value, %value"
        types = f"{pointer}, {value}, {value}"
        properties = "scope = 2 : i32, sem = 1 : i32"
    path = tmp_path / "atomic.mlir"
    path.write_text(f"""
module {{
  tt.func @atomic(%ptr: {pointer}, %value: {value}, %mask: {mask}) -> {value} {{
    %result = "tt.{operation}"({operands}) <{{{properties}}}> : ({types}) -> {value}
    tt.return %result : {value}
  }}
}}
""")
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    make_backend(GPUTarget("maca", 80, 64)).load_dialects(context)
    module = libtriton.ir.parse_mlir_module(str(path), context)
    assert module.verify()

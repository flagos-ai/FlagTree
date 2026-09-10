"""Use the actual packed local ownership and each operand's physical ABI."""

from triton.flagmega.codegen.triton.physical_access import emit_local_scalar_offset
from triton.flagmega.codegen.triton.tensor_transform_renderers import _tensor_type
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import Node
from .op import SigmoidProduct


def sigmoid_product_call(raw):
    from triton.flagmega.codegen.triton.kernel_call_renderers import (
        _buffer,
        _pointer,
        _scalar_local_domain,
        _access_in_result_domain,
        _canonical_writer_active,
    )
    operands = {p.name: _buffer(raw, "inputs", p.name) for p in SigmoidProduct.input_parameters}
    result = _buffer(raw, "outputs", "result")
    inputs = tuple(Node(name, "builtin.var", (), _tensor_type(buffer["abi"])) for name, buffer in operands.items())
    if SigmoidProduct.prepare(inputs, raw["semantic_attrs"]).result_type != _tensor_type(result["abi"]):
        raise CodegenError("SigmoidProduct result ABI disagrees with the operation's local domain.")
    domain = _scalar_local_domain(result["abi"], "_fm_offsets")
    return {
        **domain,
        "tile":
        int(raw["parameters"]["elements_per_program"]),
        "pointers": {name: _pointer(buffer)
                     for name, buffer in operands.items()},
        "offsets": {
            name: _access_in_result_domain(buffer["abi"], result["abi"], domain,
                                           lane_coordinate=domain["lane_coordinate"])
            for name, buffer in operands.items()
        },
        "result":
        _pointer(result),
        "result_offset":
        emit_local_scalar_offset(result["abi"], domain["local_coordinates"], lane_coordinate=domain["lane_coordinate"]),
        "writer_active":
        _canonical_writer_active(result["abi"]),
    }

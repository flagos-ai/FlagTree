"""Use real scalar storage and ordinary typed local-shard offsets."""

from triton.flagmega.codegen.triton.physical_access import emit_local_scalar_offset, emit_scalar_immediate
from triton.flagmega.codegen.triton.tensor_transform_renderers import _tensor_type
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import Node
from .op import ScalarScale


def scalar_scale_call(raw):
    from triton.flagmega.codegen.triton.kernel_call_renderers import (
        _buffer, _pointer, _scalar_local_domain, _access_in_result_domain, _canonical_writer_active,
    )
    value, scale = (_buffer(raw, "inputs", name) for name in ("value", "scale"))
    result = _buffer(raw, "outputs", "result")
    inputs = tuple(Node(name, "builtin.var", (), _tensor_type(buffer["abi"]))
                   for name, buffer in (("value", value), ("scale", scale)))
    if ScalarScale.infer_type(inputs, {}) != _tensor_type(result["abi"]):
        raise CodegenError("ScalarScale physical buffers disagree with the typed operation.")
    domain = _scalar_local_domain(result["abi"], "_fm_offsets")
    scalar = scale["abi"]["storage"] == "scalar"
    immediate = None
    if scalar:
        immediate = (emit_scalar_immediate(scale["abi"], scale["runtime_argument"])
                     if scale["runtime_value_kind"] == "immediate" else scale["runtime_argument"])
    return {
        **domain, "tile": int(raw["parameters"]["elements_per_program"]),
        "value": _pointer(value), "scale": None if scalar else _pointer(scale), "scalar_value": immediate,
        "scale_offset": None if scalar else emit_local_scalar_offset(
            scale["abi"], ("0",) * len(scale["abi"]["local_capacity_shape"])),
        "value_offset": _access_in_result_domain(value["abi"], result["abi"], domain,
                                                 lane_coordinate=domain["lane_coordinate"]),
        "result": _pointer(result), "result_offset": emit_local_scalar_offset(
            result["abi"], domain["local_coordinates"], lane_coordinate=domain["lane_coordinate"]),
        "writer_active": _canonical_writer_active(result["abi"]),
    }

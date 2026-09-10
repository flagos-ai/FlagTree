"""Use the ordinary local-shard ABI and two ordinary owned result buffers."""

from triton.flagmega.codegen.triton.physical_access import emit_local_scalar_offset
from triton.flagmega.codegen.triton.reduction_domain import local_reduction_domain
from triton.flagmega.codegen.triton.tensor_transform_renderers import _tensor_type
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import Node, TupleType
from .op import StagedRouting


def routing_call(raw):
    from triton.flagmega.codegen.triton.kernel_call_renderers import _buffer, _pointer, _canonical_writer_active
    source = _buffer(raw, "inputs", "value")
    values = _buffer(raw, "outputs", "result_0")
    indices = _buffer(raw, "outputs", "result_1")
    attrs = raw["semantic_attrs"]
    source_type = _tensor_type(source["abi"])
    result_type = TupleType((_tensor_type(values["abi"]), _tensor_type(indices["abi"])))
    if StagedRouting.infer_type((Node("source", "builtin.var", (), source_type), ), attrs) != result_type:
        raise CodegenError("Routing tuple buffers disagree with the inferred materialized-axis contract.")
    axis = len(source["abi"]["local_capacity_shape"]) - 1
    domain = local_reduction_domain(source["abi"], (axis, ), int(raw["parameters"]["elements_per_program"]))
    if not 0 < domain["capacity"] <= domain["tile"]:
        raise CodegenError("Staged routing implementation requires one complete local reduction tile.")
    coordinates = list(domain["coordinates"])
    coordinates[axis] = "_fm_selected_offsets"
    return {
        **domain, "k": attrs["k"], "selected_tile": 1 << (attrs["k"] - 1).bit_length(),
        "source": _pointer(source), "source_offset": emit_local_scalar_offset(source["abi"], domain["coordinates"]),
        "values_pointer": _pointer(values), "indices_pointer": _pointer(indices),
        "values_offset": emit_local_scalar_offset(values["abi"], coordinates),
        "indices_offset": emit_local_scalar_offset(indices["abi"], coordinates),
        "values_active": _canonical_writer_active(values["abi"]),
        "indices_active": _canonical_writer_active(indices["abi"]),
    }

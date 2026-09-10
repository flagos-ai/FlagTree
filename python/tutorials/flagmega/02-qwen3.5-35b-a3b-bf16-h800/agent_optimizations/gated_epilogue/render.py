"""All operand and result addressing comes from the verified local call ABI."""

from triton.flagmega.codegen.triton.physical_access import emit_active_extent, emit_local_scalar_offset
from triton.flagmega.codegen.triton.sparse_experts.common import last_axis_offset
from triton.flagmega.codegen.triton.tensor_transform_renderers import _tensor_type
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import Node, TupleType
from .op import GatedResidualNormStats


def gated_epilogue_call(raw):
    from triton.flagmega.codegen.triton.kernel_call_renderers import _buffer, _pointer, _canonical_writer_active, _bounded_vector_tile
    operands = {p.name: _buffer(raw, "inputs", p.name) for p in GatedResidualNormStats.input_parameters}
    value, stats = (_buffer(raw, "outputs", f"result_{i}") for i in (0, 1))
    attrs = raw["semantic_attrs"]
    inputs = tuple(Node(p.name, "builtin.var", (), _tensor_type(operands[p.name]["abi"]))
                   for p in GatedResidualNormStats.input_parameters)
    if GatedResidualNormStats.prepare(inputs, attrs).result_type != TupleType(
            tuple(_tensor_type(v["abi"]) for v in (value, stats))):
        raise CodegenError("Gated epilogue result ABI disagrees with its local arithmetic/statistics contract.")
    output = value["abi"]
    scalar_n = int(output["local_capacity_shape"][1]) * int(output["scalar_lane_count"])
    gate_token = "0" if int(operands["gate_logit"]["abi"]["local_capacity_shape"][0]) == 1 else "_fm_token"
    return {
        "pointers": {name: _pointer(v) for name, v in operands.items()},
        "offsets": {name: last_axis_offset(v["abi"], ("_fm_token",), "_fm_n")
                    for name, v in operands.items() if name != "gate_logit"},
        "gate_offset": emit_local_scalar_offset(operands["gate_logit"]["abi"], (gate_token, "0")),
        "value": _pointer(value), "stats": _pointer(stats),
        "value_offset": last_axis_offset(output, ("_fm_token",), "_fm_n"),
        "stats_offsets": tuple(emit_local_scalar_offset(stats["abi"], (str(i), "_fm_token", "0"))
                               for i in range(2 if attrs["use_mean"] else 1)),
        "value_writer": _canonical_writer_active(output),
        "stats_writer": _canonical_writer_active(stats["abi"]),
        "tokens": int(output["local_capacity_shape"][0]), "scalar_n": scalar_n,
        "token_active": f"(_fm_token < ({emit_active_extent(output, 0)}))",
        "n_active": f"(_fm_n < (({emit_active_extent(output, 1)}) * {output['scalar_lane_count']}))",
        "tile": _bounded_vector_tile(raw["parameters"]["block_size"], scalar_n, name="GatedEpilogue block_size"),
        "compute_warps": raw["parameters"]["compute_num_warps"], "use_mean": attrs["use_mean"],
    }

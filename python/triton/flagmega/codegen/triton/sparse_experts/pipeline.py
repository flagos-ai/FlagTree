# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Owner-local tensor maps for indexed expert weight streams."""

from triton.flagmega.codegen.triton.physical_access import emit_active_extent
from triton.flagmega.codegen.triton.tensor_descriptor_planner import packed_distributed_tensor_map_table_request
from triton.flagmega.errors import CodegenError


def pipeline_context(raw, context, operands, fields):
    if raw.get("variant") != "simt_tma_pipeline":
        return {}
    parameters = raw["parameters"]
    pipeline = raw.get("transfer_pipeline")
    workspaces = raw.get("shared_workspaces", ())
    block = (1, context["block_n"], context["block_k"])
    if (pipeline is None or len(workspaces) != len(fields)
            or any(tuple(workspace["shape"]) != (pipeline["capacity"], *block) for workspace in workspaces)):
        raise CodegenError("Expert TMA pipeline requires one typed stage buffer per weight field.")
    requests, descriptors = [], []
    for field in fields:
        binding = operands[field + "_weight"]
        abi = binding["abi"]
        shape = tuple(int(value) for value in abi["logical_shape"])
        strides = tuple(int(value) for value in abi["scalar_storage_strides"])
        if abi["scalar_lane_count"] != 1 or strides[-1] != 1:
            raise CodegenError("Expert TMA weights require scalar, contiguous K coordinates.")
        offset = int(abi["pool_byte_offset"]) if abi.get("storage") in {"rdata", "workspace"} else 0
        parameter = field + "_descriptor"
        if abi.get("distributed_type") is not None:
            request = packed_distributed_tensor_map_table_request(
                abi, parameter=parameter, source=binding["runtime_argument"], offset_bytes=offset,
                descriptor_shape=shape, descriptor_strides=strides, block_shape=block)
        else:
            request = {"parameter": parameter, "source": binding["runtime_argument"],
                       "kind": "single", "dtype": abi["scalar_dtype"], "offset_bytes": offset,
                       "shape": shape, "strides": strides, "block_shape": block,
                       "source_shape_axes": ((), (), ()), "padding": "zero", "swizzle_mode": 3}
        requests.append(request)
        descriptors.append({"field": field, "parameter": parameter, "table": request["kind"] == "table"})
    weight = operands[fields[0] + "_weight"]["abi"]
    return {
        "host_tensor_descriptor_requests": tuple(requests), "weight_descriptors": tuple(descriptors),
        "pipeline_contract": dict(pipeline), "shared_workspaces": tuple(workspaces),
        "pipeline_channel_field_names": {"weight": fields},
        "num_stages": parameters["num_stages"], "producer_warps": parameters["producer_warps"],
        "producer_registers": parameters["producer_registers"], "consumer_warps": parameters["consumer_warps"],
        "active_n": emit_active_extent(weight, 1), "active_k": emit_active_extent(weight, 2),
    }

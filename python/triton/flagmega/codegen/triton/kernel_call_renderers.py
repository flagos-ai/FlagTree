# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Family/variant call encoders for bufferized Triton PrimFunctions."""

from __future__ import annotations

from collections.abc import Mapping
import keyword
from math import prod
import re

from triton.flagmega.codegen.triton.physical_access import (
    emit_active_extent,
    emit_buffer_pointer,
    emit_global_scalar_offset,
    emit_local_scalar_offset,
    emit_logical_coordinate,
    emit_storage_pointer,
    emit_triton_scalar_type,
)
from triton.flagmega.codegen.triton.dimension_expression import emit_dimension
from triton.flagmega.codegen.triton.descriptor_abi import device_descriptor_request
from triton.flagmega.codegen.triton.tensor_descriptor_planner import (
    packed_distributed_tensor_map_table_request,
    packed_owner_prefix_tensor_map_table_request,
)
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import (
    DistributedType,
    TupleType,
    VectorType,
    local_shard_descriptor,
    type_from_data,
)


def prepare_kernel_calls(
    raw_calls,
    *,
    function_name: str,
) -> list[dict[str, object]]:
    """Lower bufferized PrimFunction calls to structured template operands."""

    calls: list[dict[str, object]] = []
    for ordinal, raw in enumerate(raw_calls):
        if not isinstance(raw, Mapping):
            raise CodegenError("TIR kernel_calls entries must be mappings.")
        family = str(raw.get("family", ""))
        facts = raw.get("facts", {})
        if not isinstance(facts, Mapping):
            raise CodegenError("TIR call facts must be a mapping.")
        internal_grid_barriers = facts.get("internal_grid_barriers", 0)
        if (
            isinstance(internal_grid_barriers, bool)
            or not isinstance(internal_grid_barriers, int)
            or internal_grid_barriers < 0
        ):
            raise CodegenError(
                "TIR internal_grid_barriers fact must be a non-negative integer."
            )
        call = {
            **dict(raw),
            "ordinal": ordinal,
            "call": str(raw.get("call", "")),
            "symbol": _call_symbol(
                function_name, ordinal, str(raw.get("call", ""))
            ),
            "family": family,
            "variant": str(raw.get("variant", "")),
            "execution_kind": str(raw.get("execution_kind", "")),
            "internal_grid_barriers": internal_grid_barriers,
            "arguments": _call_runtime_arguments(raw),
        }
        call["signature"] = ", ".join(call["arguments"])
        encoder = _FAMILY_ENCODERS.get(family)
        if encoder is None:
            raise CodegenError(
                f"TIR call {call['call']!r} has no renderer for "
                f"{family}/{call['variant']}."
            )
        call.update(encoder(raw))
        _bind_host_tensor_descriptor_parameters(call)
        _bind_transfer_pipeline_parameters(call, raw)
        calls.append(call)
    return calls


def _bind_host_tensor_descriptor_parameters(call: dict[str, object]) -> None:
    """Add descriptor formals to one reusable kernel wrapper ABI.

    Family encoders describe descriptor geometry relative to their local
    function binding.  Package expansion later resolves each formal source to
    the root storage of a concrete call instance.  Keeping the two steps
    separate lets one reusable device function receive different descriptors
    at each call site without cloning its generated kernel body.
    """

    raw = call.get("host_tensor_descriptor_requests", ())
    if not isinstance(raw, (tuple, list)):
        raise CodegenError(
            "TIR call host_tensor_descriptor_requests must be a sequence."
        )
    requests = tuple(raw)
    # Packed coordinate renderers access a complete, fixed lane axis. That
    # proof permits backing-resource rebasing without changing zero padding
    # on the (possibly ragged) outer N/K axes. Logical descriptor renderers
    # retain their bounds and only participate in exact-view interning.
    if requests and "descriptor_offsets" in call:
        for request in requests:
            entries = request["entries"] if request["kind"] == "table" else (request,)
            if any(
                int(entry["strides"][-1]) != 1
                or int(entry["shape"][-1]) != int(request["block_shape"][-1])
                for entry in entries
            ):
                raise CodegenError("Packed descriptor rebasing requires a complete stride-one lane axis.")
        requests = tuple({**request, "rebase_axis": -1} for request in requests)
        if len(requests) == 1:
            call["descriptor_offsets"] = (
                *call["descriptor_offsets"][:-1],
                f"({call['descriptor_offsets'][-1]} + {requests[0]['parameter']}__origin_elements)",
            )
    if call.get("transfer_pipeline") is not None:
        requests = tuple(device_descriptor_request(request, call["shared_workspaces"])
                         for request in requests)
    call["descriptor_storage"] = (
        "device" if requests and all(
            request.get("storage") == "device" or request["kind"] == "table"
            for request in requests
        ) else "kernel_parameter"
    )
    parameters: list[str] = []
    for request in requests:
        if not isinstance(request, Mapping):
            raise CodegenError(
                "TIR host tensor descriptor requests must be mappings."
            )
        parameter = str(request.get("parameter", ""))
        source = str(request.get("source", ""))
        if not parameter or not source or parameter in parameters:
            raise CodegenError(
                "TIR host tensor descriptor requests require unique non-empty "
                "parameter names and non-empty sources."
            )
        if parameter in call["arguments"]:
            raise CodegenError(
                f"TIR host tensor descriptor parameter {parameter!r} conflicts "
                "with a pointer argument."
            )
        parameters.append(parameter)
        if request.get("rebase_axis") is not None:
            parameters.append(f"{parameter}__origin_elements")
    call["host_tensor_descriptor_requests"] = requests
    call["descriptor_parameters"] = tuple(parameters)
    call["arguments"] = [*call["arguments"], *parameters]
    call["signature"] = ", ".join(call["arguments"])


def _bind_transfer_pipeline_parameters(
    call: dict[str, object], raw: Mapping[str, object]
) -> None:
    """Describe role-local formals from the typed transfer contract.

    Pipe endpoints and consumer-owned Shared buffers are device-function
    values, not entry ABI arguments.  Keeping their formal names on the
    prepared kernel call lets every algorithm template consume the same
    channel/workspace contract without the package planner knowing its family.
    """

    pipeline = raw.get("transfer_pipeline")
    workspaces = raw.get("shared_workspaces", ())
    if pipeline is None:
        call["pipeline_channels"] = ()
        call["pipeline_consumer_workspaces"] = ()
        call["pipeline_consumer_parameters"] = ()
        call["pipeline_producer_parameters"] = ()
        return
    if not isinstance(pipeline, Mapping) or not isinstance(
        workspaces, (tuple, list)
    ):
        raise CodegenError(
            "A transfer-pipelined TIR call requires typed Shared workspace ABI."
        )
    channels = pipeline.get("channels")
    consumer_indices = pipeline.get("consumer_shared_workspace_indices", ())
    if not isinstance(channels, (tuple, list)) or not channels:
        raise CodegenError("A transfer pipeline must expose at least one channel.")
    if not isinstance(consumer_indices, (tuple, list)):
        raise CodegenError(
            "Transfer-pipeline consumer Shared workspace indices must be a sequence."
        )

    def workspace(index: object, owner: str) -> Mapping[str, object]:
        if isinstance(index, bool) or not isinstance(index, int):
            raise CodegenError(f"{owner} workspace index must be an integer.")
        try:
            value = workspaces[index]
        except IndexError as error:
            raise CodegenError(
                f"{owner} workspace index {index} is outside the Shared ABI."
            ) from error
        if not isinstance(value, Mapping):
            raise CodegenError(f"{owner} Shared workspace must be a mapping.")
        name = str(value.get("name", ""))
        _require_pipeline_identifier(name, f"{owner} Shared workspace")
        return value

    encoded_channels = []
    consumer_parameters = []
    producer_parameters = []
    field_overrides = call.get("pipeline_channel_field_names", {})
    if not isinstance(field_overrides, Mapping):
        raise CodegenError(
            "pipeline_channel_field_names must map channel names to field names."
        )
    for channel in channels:
        if not isinstance(channel, Mapping):
            raise CodegenError("Transfer-pipeline channel must be a mapping.")
        name = str(channel.get("name", ""))
        _require_pipeline_identifier(name, "Transfer-pipeline channel")
        indices = channel.get("shared_workspace_indices")
        if not isinstance(indices, (tuple, list)) or not indices:
            raise CodegenError(
                f"Transfer-pipeline channel {name!r} owns no Shared workspace."
            )
        owned = tuple(workspace(index, f"Channel {name!r}") for index in indices)
        raw_fields = field_overrides.get(name)
        if raw_fields is None:
            field_names = (
                (name,)
                if len(owned) == 1
                else tuple(str(value["name"]) for value in owned)
            )
        elif isinstance(raw_fields, (tuple, list)):
            field_names = tuple(str(value) for value in raw_fields)
        else:
            raise CodegenError(
                f"Transfer-pipeline channel {name!r} field override must be a sequence."
            )
        if len(field_names) != len(owned):
            raise CodegenError(
                f"Transfer-pipeline channel {name!r} has {len(owned)} workspaces "
                f"but {len(field_names)} pipe field names."
            )
        for field_name in field_names:
            _require_pipeline_identifier(
                field_name, f"Transfer-pipeline channel {name!r} field"
            )
        if len(set(field_names)) != len(field_names):
            raise CodegenError(
                f"Transfer-pipeline channel {name!r} has duplicate pipe fields."
            )
        reader = f"pipeline_{name}_reader"
        writer = f"pipeline_{name}_writer"
        encoded_channels.append({
            "name": name,
            "reader_parameter": reader,
            "writer_parameter": writer,
            "workspace_names": tuple(str(value["name"]) for value in owned),
            "fields": tuple({
                "name": field_name,
                "workspace_name": str(value["name"]),
            } for field_name, value in zip(field_names, owned, strict=True)),
        })
        consumer_parameters.append(reader)
        producer_parameters.append(writer)

    encoded_consumer_workspaces = []
    for index in consumer_indices:
        value = workspace(index, "Consumer")
        name = str(value["name"])
        parameter = f"pipeline_consumer_{name}"
        encoded_consumer_workspaces.append({
            "name": name,
            "parameter": parameter,
        })
        consumer_parameters.append(parameter)

    auxiliary = pipeline.get("auxiliary_consumer")
    encoded_auxiliary = None
    if auxiliary is not None:
        if not isinstance(auxiliary, Mapping):
            raise CodegenError(
                "Transfer-pipeline auxiliary consumer must be a mapping."
            )
        channel_indices = auxiliary.get("channel_indices")
        workspace_indices = auxiliary.get(
            "consumer_shared_workspace_indices", ()
        )
        if (
            not isinstance(channel_indices, (tuple, list))
            or not channel_indices
            or not isinstance(workspace_indices, (tuple, list))
        ):
            raise CodegenError(
                "Auxiliary consumer channel/workspace indices must be sequences."
            )
        try:
            selected_channels = tuple(
                encoded_channels[int(index)] for index in channel_indices
            )
        except (IndexError, TypeError, ValueError) as error:
            raise CodegenError(
                "Auxiliary consumer references an invalid transfer channel."
            ) from error
        consumer_by_name = {
            value["name"]: value for value in encoded_consumer_workspaces
        }
        selected_workspaces = []
        for index in workspace_indices:
            value = workspace(index, "Auxiliary consumer")
            try:
                selected_workspaces.append(consumer_by_name[value["name"]])
            except KeyError as error:
                raise CodegenError(
                    "Auxiliary consumer references a Shared workspace not "
                    "owned by the primary consumer."
                ) from error
        primary_parameters = (
            "pipeline_auxiliary_start_writer",
            "pipeline_auxiliary_done_reader",
        )
        auxiliary_parameters = (
            *(value["reader_parameter"] for value in selected_channels),
            "pipeline_auxiliary_start_reader",
            "pipeline_auxiliary_done_writer",
            *(value["parameter"] for value in selected_workspaces),
        )
        encoded_auxiliary = {
            "channel_indices": tuple(int(value) for value in channel_indices),
            "consumer_shared_workspace_indices": tuple(
                int(value) for value in workspace_indices
            ),
            "start_writer_parameter": primary_parameters[0],
            "done_reader_parameter": primary_parameters[1],
            "start_reader_parameter": "pipeline_auxiliary_start_reader",
            "done_writer_parameter": "pipeline_auxiliary_done_writer",
            "primary_parameters": primary_parameters,
            "parameters": auxiliary_parameters,
        }
        consumer_parameters.extend(primary_parameters)

    call["pipeline_contract"] = dict(pipeline)
    call["shared_workspaces"] = tuple(workspaces)
    call["pipeline_channels"] = tuple(encoded_channels)
    call["pipeline_consumer_workspaces"] = tuple(
        encoded_consumer_workspaces
    )
    call["pipeline_consumer_parameters"] = tuple(consumer_parameters)
    call["pipeline_producer_parameters"] = tuple(producer_parameters)
    call["pipeline_auxiliary_consumer"] = encoded_auxiliary


def _require_pipeline_identifier(value: str, owner: str) -> None:
    if not value.isidentifier() or keyword.iskeyword(value):
        raise CodegenError(f"{owner} name {value!r} is not a Python identifier.")


def _call_runtime_arguments(raw) -> list[str]:
    arguments: list[str] = []
    for group in ("inputs", "outputs", "workspaces"):
        parameters = raw.get(group, ())
        if not isinstance(parameters, (tuple, list)):
            raise CodegenError(f"TIR call has no {group!r} parameter list.")
        for parameter in parameters:
            if not isinstance(parameter, Mapping):
                raise CodegenError("TIR call parameter binding must be a mapping.")
            buffers = parameter.get("buffers", ())
            if not isinstance(buffers, (tuple, list)):
                raise CodegenError("TIR call buffer binding must be a sequence.")
            for binding in buffers:
                if not isinstance(binding, Mapping):
                    raise CodegenError("TIR call buffer binding must be a mapping.")
                abi = binding.get("abi")
                if not isinstance(abi, Mapping):
                    raise CodegenError("TIR call buffer has no physical ABI.")
                if binding.get("runtime_value_kind") == "immediate":
                    continue
                argument = binding.get("runtime_argument")
                if not isinstance(argument, str):
                    raise CodegenError("TIR call buffer has no runtime argument.")
                if argument not in arguments:
                    arguments.append(argument)
    return arguments


def _call_symbol(function_name: str, ordinal: int, call_id: str) -> str:
    function_stem = re.sub(
        r"[^a-zA-Z0-9_]+", "_", function_name
    ).strip("_").lower()
    call_stem = re.sub(
        r"[^a-zA-Z0-9_]+", "_", call_id
    ).strip("_").lower()
    return f"_flagmega_{function_stem}_call_{ordinal}_{call_stem}"


def _parameter(raw, group: str, formal: str) -> Mapping[str, object]:
    values = raw.get(group)
    if not isinstance(values, (tuple, list)):
        raise CodegenError(f"TIR call has no {group!r} parameter list.")
    matched = tuple(
        value
        for value in values
        if isinstance(value, Mapping) and str(value.get("formal")) == formal
    )
    if len(matched) != 1:
        raise CodegenError(
            f"TIR call requires exactly one {group} formal {formal!r}."
        )
    return matched[0]


def _buffer(
    raw,
    group: str,
    formal: str,
    index: int = 0,
) -> Mapping[str, object]:
    parameter = _parameter(raw, group, formal)
    buffers = parameter.get("buffers")
    if not isinstance(buffers, (tuple, list)):
        raise CodegenError(f"TIR call formal {formal!r} has no buffer bindings.")
    try:
        binding = buffers[index]
    except IndexError as error:
        raise CodegenError(
            f"TIR call formal {formal!r} has no buffer field {index}."
        ) from error
    if not isinstance(binding, Mapping) or not isinstance(binding.get("abi"), Mapping):
        raise CodegenError(f"TIR call formal {formal!r} has an invalid buffer ABI.")
    if not isinstance(binding.get("runtime_argument"), str):
        raise CodegenError(f"TIR call formal {formal!r} has no runtime argument.")
    return binding


def _buffer_from_formals(
    raw,
    group: str,
    formals: tuple[str, ...],
    index: int = 0,
) -> Mapping[str, object]:
    """Resolve the one family ABI formal present at a call site.

    A kernel family may implement multiple semantic operations whose
    ``ParameterInfo`` names differ (for example, MatMul uses ``rhs`` while
    PackedDenseMatMul uses ``weight``).  The selected call ABI is authoritative;
    branching on the semantic op duplicates that schema and breaks as soon as
    another operation lowers to the same family.
    """

    values = raw.get(group)
    if not isinstance(values, (tuple, list)):
        raise CodegenError(f"TIR call has no {group!r} parameter list.")
    available = {
        str(value.get("formal"))
        for value in values
        if isinstance(value, Mapping)
    }
    matched = tuple(formal for formal in formals if formal in available)
    if len(matched) != 1:
        raise CodegenError(
            f"TIR call requires exactly one {group} formal from {formals!r}; "
            f"found {matched!r}."
        )
    return _buffer(raw, group, matched[0], index)


def _buffers(raw, group: str, formal: str) -> tuple[Mapping[str, object], ...]:
    parameter = _parameter(raw, group, formal)
    values = parameter.get("buffers")
    if not isinstance(values, (tuple, list)) or not values:
        raise CodegenError(f"TIR call formal {formal!r} has no buffer bindings.")
    result = tuple(values)
    if any(
        not isinstance(binding, Mapping)
        or not isinstance(binding.get("abi"), Mapping)
        or not isinstance(binding.get("runtime_argument"), str)
        for binding in result
    ):
        raise CodegenError(f"TIR call formal {formal!r} has an invalid buffer ABI.")
    return result


def _flattened_buffers(raw, group: str) -> tuple[Mapping[str, object], ...]:
    parameters = raw.get(group)
    if not isinstance(parameters, (tuple, list)) or not parameters:
        raise CodegenError(f"TIR call has no {group!r} buffer bindings.")
    result: list[Mapping[str, object]] = []
    for parameter in parameters:
        if not isinstance(parameter, Mapping):
            raise CodegenError("TIR call parameter binding must be a mapping.")
        values = parameter.get("buffers")
        if not isinstance(values, (tuple, list)) or not values:
            raise CodegenError(
                f"TIR call {group} formal {parameter.get('formal')!r} has no "
                "buffer bindings."
            )
        for binding in values:
            if (
                not isinstance(binding, Mapping)
                or not isinstance(binding.get("abi"), Mapping)
                or not isinstance(binding.get("runtime_argument"), str)
            ):
                raise CodegenError(
                    f"TIR call {group} formal {parameter.get('formal')!r} has "
                    "an invalid buffer ABI."
                )
            result.append(binding)
    return tuple(result)


def _pointer(binding: Mapping[str, object]) -> str:
    return emit_buffer_pointer(
        binding["abi"],
        str(binding["runtime_argument"]),
    )


def _static_shape(abi: Mapping[str, object], key: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value) for value in abi[key])
    except (KeyError, TypeError, ValueError) as error:
        raise CodegenError(f"TIR renderer requires a static {key}.") from error
    return values


def _embedding_call(raw) -> dict[str, object]:
    indices = _buffer(raw, "inputs", "indices")
    weight = _buffer(raw, "inputs", "weight")
    result = _buffer(raw, "outputs", "result")
    result_abi = result["abi"]
    local_shape = _static_shape(result_abi, "local_capacity_shape")
    if len(local_shape) != 2 or local_shape[0] != 1:
        raise CodegenError(
            "TIR decode embedding requires one local token row; batching is "
            "lowered by a separate schedule."
        )
    local_feature = "local_offsets"
    global_feature = emit_logical_coordinate(
        result_abi, 1, ("0", local_feature)
    )
    weight_abi = weight["abi"]
    if str(weight_abi["coordinate_space"]) != "canonical_global":
        raise CodegenError(
            "Embedding lookup requires a canonical vocabulary table or an "
            "explicit lookup-table Boxing implementation."
        )
    padding_idx = raw.get("semantic_attrs", {}).get("padding_idx")
    return {
        "indices": _pointer(indices),
        "weight": _pointer(weight),
        "result": _pointer(result),
        "local_capacity": local_shape[-1],
        "active": emit_active_extent(result_abi, 1),
        "weight_offset": emit_global_scalar_offset(
            weight_abi, ("token_id", global_feature)
        ),
        "result_offset": emit_local_scalar_offset(
            result_abi, ("0", local_feature)
        ),
        "tile": int(raw["parameters"]["elements_per_program"]),
        "vocab_size": _static_shape(weight_abi, "logical_shape")[0],
        "padding_idx": -1 if padding_idx is None else int(padding_idx),
    }


def _rms_norm_call(raw) -> dict[str, object]:
    value = _buffer(raw, "inputs", "value")
    weight = _buffer(raw, "inputs", "weight")
    result = _buffer(raw, "outputs", "result")
    value_abi = value["abi"]
    weight_abi = weight["abi"]
    result_abi = result["abi"]
    logical_shape = _static_shape(result_abi, "logical_shape")
    value_logical_shape = _static_shape(value_abi, "logical_shape")
    weight_logical_shape = _static_shape(weight_abi, "logical_shape")
    if not logical_shape or value_logical_shape != logical_shape:
        raise CodegenError(
            "TIR RMSNorm value and result must have one identical, non-empty "
            "logical shape."
        )
    if weight_logical_shape != (logical_shape[-1],):
        raise CodegenError(
            "TIR RMSNorm weight must exactly match the normalized last axis: "
            f"expected {(logical_shape[-1],)}, got {weight_logical_shape}."
        )

    local_shape = _static_shape(result_abi, "local_capacity_shape")
    value_local_shape = _static_shape(value_abi, "local_capacity_shape")
    if value_local_shape != local_shape:
        raise CodegenError(
            "TIR RMSNorm value and result must expose the same local shard "
            "capacity; insert explicit Boxing before code generation."
        )
    lane_count = int(result_abi.get("scalar_lane_count", 1))
    lane_shape = tuple(result_abi.get("scalar_lane_shape", ()))
    if lane_count <= 0:
        raise CodegenError("TIR RMSNorm requires a positive scalar lane count.")
    for name, abi in (("value", value_abi), ("weight", weight_abi)):
        if (
            int(abi.get("scalar_lane_count", 1)) != lane_count
            or tuple(abi.get("scalar_lane_shape", ())) != lane_shape
        ):
            raise CodegenError(
                "TIR RMSNorm value/weight/result vector lanes must match; "
                f"{name} differs from the result."
            )

    outer_shape = local_shape[:-1]
    outer_coordinates = _unflattened_coordinates(
        outer_shape, "rms_outer_index"
    )
    physical_inner = (
        "rms_reduction_offsets"
        if lane_count == 1
        else f"((rms_reduction_offsets) // {lane_count})"
    )
    lane_coordinate = (
        None
        if lane_count == 1
        else f"((rms_reduction_offsets) % {lane_count})"
    )
    local_coordinates = (*outer_coordinates, physical_inner)
    logical_coordinates = tuple(
        emit_logical_coordinate(result_abi, axis, local_coordinates)
        for axis in range(len(local_shape))
    )
    source_active = " & ".join(
        f"(({coordinate}) < ({emit_active_extent(value_abi, axis)}))"
        for axis, coordinate in enumerate(local_coordinates)
    ) or "True"
    result_active = " & ".join(
        f"(({coordinate}) < ({emit_active_extent(result_abi, axis)}))"
        for axis, coordinate in enumerate(local_coordinates)
    ) or "True"
    domain = {
        "local_coordinates": local_coordinates,
        "logical_coordinates": logical_coordinates,
    }
    reduction_capacity = local_shape[-1] * lane_count
    return {
        "value": _pointer(value),
        "weight": _pointer(weight),
        "result": _pointer(result),
        "partial_stats": None,
        "lane_coordinate": lane_coordinate,
        "outer_capacity": prod(outer_shape, start=1),
        "reduction_capacity": reduction_capacity,
        "normalization_size": logical_shape[-1] * lane_count,
        "active": f"({source_active}) & ({result_active})",
        "writer_active": _canonical_writer_active(result_abi),
        "value_offset": _access_in_result_domain(
            value_abi,
            result_abi,
            domain,
            lane_coordinate=lane_coordinate,
        ),
        "weight_offset": _norm_apply_parameter_offset(
            weight_abi,
            result_abi,
            len(logical_shape) - 1,
            local_coordinates,
            logical_coordinates,
            lane_coordinate,
            "weight",
        ),
        "result_offset": emit_local_scalar_offset(
            result_abi,
            local_coordinates,
            lane_coordinate=lane_coordinate,
        ),
        "epsilon": repr(float(raw.get("semantic_attrs", {}).get("epsilon", 0.0))),
        "weight_bias": repr(float(raw.get("semantic_attrs", {}).get("weight_bias", 0.0))),
        "tile": _bounded_vector_tile(
            raw["parameters"]["block_size"],
            reduction_capacity,
            name="RMSNorm block_size",
        ),
        "output_type": _triton_dtype(str(result_abi["scalar_dtype"])),
    }


def _block_fp8_call(raw) -> dict[str, object]:
    value = _buffer(raw, "inputs", "value")
    weight = _buffer(raw, "inputs", "weight")
    scale = _buffer(raw, "inputs", "weight_scale")
    result = _buffer(raw, "outputs", "result")
    value_abi = value["abi"]
    scale_abi = scale["abi"]
    result_abi = result["abi"]
    value_shape = _static_shape(value_abi, "logical_shape")
    result_local = _static_shape(result_abi, "local_capacity_shape")
    if len(value_shape) != 2 or len(result_local) != 2:
        raise CodegenError("TIR block-FP8 renderer requires rank-two GEMV buffers.")
    local_n = "local_n_offsets"
    global_n = emit_logical_coordinate(result_abi, 1, ("0", local_n))
    global_k_size = _scalar_logical_axis_extent(value_abi, -1)
    global_n_size = _scalar_logical_axis_extent(result_abi, -1)
    block_n, block_k = _semantic_weight_block_shape(
        raw,
        scales=(scale_abi,),
        global_n=global_n_size,
        global_k=global_k_size,
        owner="TIR block-FP8",
    )
    # The K-major packed outer row starts at the same scalar address; K is
    # represented by its outer vector coordinate plus flattened lane.
    weight_row = f"({global_n}) * {global_k_size}"
    scale_row = emit_global_scalar_offset(
        scale_abi,
        (f"({global_n}) // {block_n}", "0"),
    )
    return {
        "value": _pointer(value),
        "weight": _pointer(weight),
        "scale": _pointer(scale),
        "result": _pointer(result),
        "local_n_capacity": result_local[-1],
        "active_n": emit_active_extent(result_abi, 1),
        "global_n": global_n,
        "weight_row_offset": weight_row,
        "scale_row_offset": scale_row,
        "destination_offset": emit_local_scalar_offset(
            result_abi, ("0", local_n)
        ),
        "k": global_k_size,
        "block_k": block_k,
        "tile_n": int(raw["parameters"]["tile_n"]),
    }


def _matmul_glu_call(raw) -> dict[str, object]:
    value = _buffer(raw, "inputs", "value")
    gate_weight = _buffer(raw, "inputs", "gate_weight")
    up_weight = _buffer(raw, "inputs", "up_weight")
    gate_scale = _buffer(raw, "inputs", "gate_scale")
    up_scale = _buffer(raw, "inputs", "up_scale")
    result = _buffer(raw, "outputs", "result")
    value_shape = _static_shape(value["abi"], "logical_shape")
    result_abi = result["abi"]
    result_local = _static_shape(result_abi, "local_capacity_shape")
    if len(value_shape) != 2 or len(result_local) != 2:
        raise CodegenError("TIR MatMulGlu renderer requires rank-two GEMV buffers.")
    local_n = "local_n_offsets"
    global_n = emit_logical_coordinate(result_abi, 1, ("0", local_n))
    global_k_size = _scalar_logical_axis_extent(value["abi"], -1)
    global_n_size = _scalar_logical_axis_extent(result_abi, -1)
    block_n, block_k = _semantic_weight_block_shape(
        raw,
        scales=(gate_scale["abi"], up_scale["abi"]),
        global_n=global_n_size,
        global_k=global_k_size,
        owner="TIR MatMulGlu",
    )
    return {
        "value": _pointer(value),
        "gate_weight": _pointer(gate_weight),
        "up_weight": _pointer(up_weight),
        "gate_scale": _pointer(gate_scale),
        "up_scale": _pointer(up_scale),
        "result": _pointer(result),
        "local_n_capacity": result_local[-1],
        "active_n": emit_active_extent(result_abi, 1),
        "global_n": global_n,
        "weight_row_offset": f"({global_n}) * {global_k_size}",
        "scale_row_offset": emit_global_scalar_offset(
            gate_scale["abi"],
            (f"({global_n}) // {block_n}", "0"),
        ),
        "destination_offset": emit_local_scalar_offset(
            result_abi, ("0", local_n)
        ),
        "k": global_k_size,
        "block_k": block_k,
        "tile_n": int(raw["parameters"]["tile_n"]),
    }


def _boxing_call(raw) -> dict[str, object]:
    sources = _flattened_buffers(raw, "inputs")
    results = _flattened_buffers(raw, "outputs")
    transitions = tuple(
        str(value) for value in raw["parameters"].get("leaf_transitions", ())
    )
    if len(sources) != len(results) or len(sources) != len(transitions):
        raise CodegenError(
            "TIR Boxing leaf transitions must match its flattened input/output "
            f"buffers: {len(transitions)} != {len(sources)} != {len(results)}."
        )
    leaves = tuple(
        _boxing_leaf(source, result, transition, raw)
        for source, result, transition in zip(
            sources, results, transitions, strict=True
        )
    )
    # Preserve the convenient scalar-leaf descriptor fields used by existing
    # diagnostics while making tuple leaves first-class for source emission.
    return {**(leaves[0] if len(leaves) == 1 else {}), "leaves": leaves}


def _boxing_leaf(source, result, transition: str, raw) -> dict[str, object]:
    source_abi = source["abi"]
    result_abi = result["abi"]
    if transition not in {
        "identity", "tensor_load", "tensor_store", "gather_reduce_scatter",
    }:
        raise CodegenError(f"TIR Boxing has unknown leaf transition {transition!r}.")
    if _has_partial(result_abi):
        raise CodegenError(
            "TIR Boxing cannot create a partial value from a materialized source."
        )
    source_lanes = int(source_abi["scalar_lane_count"])
    result_lanes = int(result_abi["scalar_lane_count"])
    if (
        source_lanes != result_lanes
        or tuple(source_abi.get("scalar_lane_shape", ()))
        != tuple(result_abi.get("scalar_lane_shape", ()))
    ):
        raise CodegenError("TIR Boxing source/result vector lanes must match.")
    source_is_scalar = str(source_abi.get("storage")) == "scalar"
    result_is_scalar = str(result_abi.get("storage")) == "scalar"
    if source_is_scalar or result_is_scalar:
        if (
            not source_is_scalar
            or result_is_scalar
            or transition != "tensor_load"
            or source_lanes != 1
            or _static_shape(source_abi, "logical_shape")
            or _static_shape(result_abi, "logical_shape")
        ):
            raise CodegenError(
                "Scalar Boxing only supports tensor_load from one rank-zero "
                "runtime/compile-time value into a materialized rank-zero tensor."
            )
        result_offset = (
            emit_global_scalar_offset(result_abi, ())
            if str(result_abi.get("coordinate_space")) == "canonical_global"
            else emit_local_scalar_offset(result_abi, ())
        )
        return {
            "transition": transition,
            "reduction": False,
            "mode": "scalar_load",
            "source_is_scalar": True,
            "source": _scalar_expression(source),
            "result": _pointer(result),
            "capacity": 1,
            "source_offset": "0",
            "result_offset": result_offset,
            "active": "True",
            "writer_active": _canonical_writer_active(result_abi),
            "tile": 1,
        }
    if _has_partial(source_abi):
        return _partial_boxing_leaf(
            source, result, transition, raw, lane_count=source_lanes
        )
    source_space = str(source_abi["coordinate_space"])
    result_space = str(result_abi["coordinate_space"])
    if source_space == "local" and result_space == "canonical_global":
        mode = "scatter"
        iteration_abi = source_abi
        logical_abi = source_abi
    elif source_space == "canonical_global" and result_space == "local":
        mode = "gather"
        iteration_abi = result_abi
        logical_abi = result_abi
    elif source_space == "canonical_global" and result_space == "canonical_global":
        mode = "canonical_copy"
        iteration_abi = result_abi
        logical_abi = result_abi
    else:
        source_mapping = tuple(source_abi["logical_coordinate_expressions"])
        result_mapping = tuple(result_abi["logical_coordinate_expressions"])
        if (
            source_mapping != result_mapping
            or tuple(source_abi["local_capacity_shape"])
            != tuple(result_abi["local_capacity_shape"])
        ):
            raise CodegenError(
                "Compact-to-compact Boxing with different owner maps requires "
                "an explicit routed transfer lowering."
            )
        mode = "local_copy"
        iteration_abi = source_abi
        logical_abi = source_abi
    local_shape = _static_shape(iteration_abi, "local_capacity_shape")
    scalar_capacity = prod(local_shape, start=1) * source_lanes
    physical_flat = (
        "boxing_offsets"
        if source_lanes == 1
        else f"((boxing_offsets) // {source_lanes})"
    )
    lane_coordinate = (
        None
        if source_lanes == 1
        else f"((boxing_offsets) % {source_lanes})"
    )
    local_coordinates = _unflattened_coordinates(local_shape, physical_flat)
    active = " & ".join(
        f"(({coordinate}) < ({emit_active_extent(iteration_abi, axis)}))"
        for axis, coordinate in enumerate(local_coordinates)
    ) or "True"
    logical_coordinates = tuple(
        emit_logical_coordinate(logical_abi, axis, local_coordinates)
        for axis in range(len(local_shape))
    )
    if mode in {"scatter", "local_copy"}:
        source_offset = emit_local_scalar_offset(
            source_abi, local_coordinates, lane_coordinate=lane_coordinate
        )
    else:
        source_offset = emit_global_scalar_offset(
            source_abi, logical_coordinates, lane_coordinate=lane_coordinate
        )
    if mode in {"gather", "local_copy"}:
        result_offset = emit_local_scalar_offset(
            result_abi, local_coordinates, lane_coordinate=lane_coordinate
        )
    else:
        result_offset = emit_global_scalar_offset(
            result_abi, logical_coordinates, lane_coordinate=lane_coordinate
        )
    return {
        "transition": transition,
        "reduction": False,
        "mode": mode,
        "source": _pointer(source),
        "result": _pointer(result),
        "capacity": scalar_capacity,
        "source_offset": source_offset,
        "result_offset": result_offset,
        "active": active,
        "writer_active": (
            _distributed_unique_writer_active(iteration_abi)
            if result_space == "canonical_global"
            else "True"
        ),
        "tile": _bounded_vector_tile(
            raw["parameters"]["tile"], scalar_capacity, name="Boxing tile"
        ),
    }


def _partial_boxing_leaf(
    source,
    result,
    transition: str,
    raw,
    *,
    lane_count: int,
) -> dict[str, object]:
    from triton.flagmega.codegen.triton.kernels.distributed_boxing.reduction import (
        partial_reduction_context,
    )

    source_abi = source["abi"]
    result_abi = result["abi"]
    if transition != "gather_reduce_scatter":
        raise CodegenError(
            "A partial Boxing source requires gather_reduce_scatter."
        )
    if str(source_abi.get("storage_kind")) != "compact_per_owner":
        raise CodegenError(
            "Partial Boxing requires one compact source component per owner."
        )
    if str(result_abi.get("coordinate_space")) != "canonical_global":
        raise CodegenError(
            "Partial Boxing currently materializes canonical-global storage; "
            "a later local view may expose any compatible target sharding."
        )
    owner_stride = int(source_abi.get("component_stride_scalar_elements", 0))
    if owner_stride <= 0:
        raise CodegenError("Partial Boxing source has no owner component stride.")
    local_shape = _static_shape(source_abi, "local_capacity_shape")
    scalar_capacity = prod(local_shape, start=1) * lane_count
    physical_flat = (
        "boxing_offsets"
        if lane_count == 1
        else f"((boxing_offsets) // {lane_count})"
    )
    lane_coordinate = (
        None
        if lane_count == 1
        else f"((boxing_offsets) % {lane_count})"
    )
    local_coordinates = _unflattened_coordinates(local_shape, physical_flat)
    logical_coordinates = tuple(
        emit_logical_coordinate(source_abi, axis, local_coordinates)
        for axis in range(len(local_shape))
    )
    active = " & ".join(
        f"(({coordinate}) < ({emit_active_extent(source_abi, axis)}))"
        for axis, coordinate in enumerate(local_coordinates)
    ) or "True"
    distributed = source_abi.get("distributed_type")
    assert isinstance(distributed, Mapping)
    partial = distributed.get("partial")
    assert isinstance(partial, Mapping)
    axes = tuple(int(value) for value in partial.get("axes", ()))
    placement = distributed.get("placement")
    if not isinstance(placement, Mapping):
        raise CodegenError("Partial Boxing source has no placement ABI.")
    hierarchy = tuple(int(value) for value in placement.get("hierarchy", ()))
    owner_count = prod((hierarchy[axis] for axis in axes), start=1)
    placement_owner_count = prod(hierarchy, start=1)
    value_tile = _bounded_vector_tile(
        raw["parameters"]["tile"], scalar_capacity, name="Partial Boxing tile"
    )
    # Share the implementation's scalar tile budget across value and owner
    # dimensions. Small local tensors use otherwise idle lanes to reduce
    # owners in parallel; wide tensors do not grow an unbounded 2-D tile.
    owner_tile = _bounded_vector_tile(
        int(raw["parameters"]["tile"]) // value_tile,
        owner_count,
        name="Partial Boxing owner tile",
    )
    return {
        "transition": transition,
        "reduction": True,
        **partial_reduction_context(
            str(partial.get("reduce_op")), str(source_abi["scalar_dtype"]),
        ),
        "source": emit_storage_pointer(
            source_abi, str(source["runtime_argument"])
        ),
        "result": _pointer(result),
        "capacity": scalar_capacity,
        "source_offset": emit_local_scalar_offset(
            source_abi,
            local_coordinates,
            lane_coordinate=lane_coordinate,
        ),
        "result_offset": emit_global_scalar_offset(
            result_abi,
            logical_coordinates,
            lane_coordinate=lane_coordinate,
        ),
        "active": active,
        "writer_active": _distributed_unique_writer_active(source_abi),
        "partial_owner_count": owner_count,
        "partial_owner_tile": owner_tile,
        "placement_owner_count": placement_owner_count,
        "partial_owner": _partial_group_owner_expression_for_axes(
            source_abi, axes, "boxing_partial_member"
        ),
        "owner_stride": owner_stride,
        "tile": value_tile,
        "output_type": _triton_dtype(str(result_abi["scalar_dtype"])),
    }


def _has_partial(abi: Mapping[str, object]) -> bool:
    distributed = abi.get("distributed_type")
    return isinstance(distributed, Mapping) and distributed.get("partial") is not None


def _unflattened_coordinates(shape: tuple[int, ...], flat: str) -> tuple[str, ...]:
    result = []
    for axis, extent in enumerate(shape):
        inner = prod(shape[axis + 1 :], start=1)
        coordinate = flat if inner == 1 else f"(({flat}) // {inner})"
        if axis:
            coordinate = f"(({coordinate}) % {extent})"
        result.append(coordinate)
    return tuple(result)


def _gdn_convolution_call(raw) -> dict[str, object]:
    qkv = _buffer(raw, "inputs", "qkv")
    state = _buffer(raw, "inputs", "state", 0)
    weight = _buffer(raw, "inputs", "conv_weight")
    result = _buffer(raw, "outputs", "result_0")
    result_abi = result["abi"]
    local_shape = _static_shape(result_abi, "local_capacity_shape")
    if len(local_shape) != 2:
        raise CodegenError("TIR GDN convolution requires a rank-two local output.")
    local_channel = "local_channels"
    global_channel = emit_logical_coordinate(
        result_abi, 1, ("0", local_channel)
    )
    attrs = raw.get("semantic_attrs", {})
    kernel = int(attrs["conv_kernel_size"])
    return {
        "qkv": _pointer(qkv),
        "state": _pointer(state),
        "weight": _pointer(weight),
        "result": _pointer(result),
        "local_capacity": local_shape[-1],
        "active_channels": emit_active_extent(result_abi, 1),
        "global_channel": global_channel,
        "history": kernel - 1,
        "kernel": kernel,
        "tile": int(raw["parameters"]["tile_channels"]),
        # Mutable state is shared across broadcast replicas.  Elect one
        # representative for every distinct result shard so replicas do not
        # apply the recurrence repeatedly to the same state coordinates.
        "writer_active": _distributed_unique_writer_active(result_abi),
    }


def _gdn_recurrent_call(raw) -> dict[str, object]:
    state = _buffer(raw, "inputs", "state", 1)
    qkv = _buffer(raw, "inputs", "qkv")
    z = _buffer(raw, "inputs", "z")
    projection_input = _buffer(raw, "inputs", "projection_input")
    b_weight = _buffer(raw, "inputs", "b_weight")
    a_weight = _buffer(raw, "inputs", "a_weight")
    a_log = _buffer(raw, "inputs", "a_log")
    dt_bias = _buffer(raw, "inputs", "dt_bias")
    norm_weight = _buffer(raw, "inputs", "norm_weight")
    result = _buffer(raw, "outputs", "result_0")
    scratch = _buffer(raw, "workspaces", "core_scratch")
    result_abi = result["abi"]
    local_shape = _static_shape(result_abi, "local_capacity_shape")
    local_value = "local_values"
    global_value = emit_logical_coordinate(result_abi, 1, ("0", local_value))
    attrs = raw.get("semantic_attrs", {})
    return {
        "state": _pointer(state),
        "qkv": _pointer(qkv),
        "z": _pointer(z),
        "projection_input": _pointer(projection_input),
        "b_weight": _pointer(b_weight),
        "a_weight": _pointer(a_weight),
        "a_log": _pointer(a_log),
        "dt_bias": _pointer(dt_bias),
        "norm_weight": _pointer(norm_weight),
        "result": _pointer(result),
        "core_scratch": _pointer(scratch),
        "local_capacity": local_shape[-1],
        "active_values": emit_active_extent(result_abi, 1),
        "global_value": global_value,
        "value_heads": int(attrs["num_value_heads"]),
        "key_heads": int(attrs["num_key_heads"]),
        "key_dim": int(attrs["key_head_dim"]),
        "value_dim": int(attrs["value_head_dim"]),
        "projection_dim": _static_shape(
            projection_input["abi"], "logical_shape"
        )[-1],
        "repeats": int(attrs["num_value_heads"]) // int(attrs["num_key_heads"]),
        "epsilon": repr(float(attrs["epsilon"])),
        "value_tile": int(raw["parameters"]["tile_state"][1]),
        "head_block": int(raw["parameters"]["tile_state"][0]),
        "projection_tile": int(raw["parameters"]["projection_tile"]),
        "query_scale": repr(1.0 / (int(attrs["key_head_dim"]) ** 0.5)),
        "writer_active": _distributed_unique_writer_active(result_abi),
    }


def _add_norm_stats_call(raw) -> dict[str, object]:
    source = _buffer(raw, "inputs", "input")
    residual = _buffer(raw, "inputs", "addend")
    result = _buffer(raw, "outputs", "result_0")
    stats = _buffer(raw, "outputs", "result_1")
    workspaces = raw.get("workspaces", ())
    partials = (
        _buffer(raw, "workspaces", "norm_stats_partials")
        if workspaces
        else None
    )
    source_abi = source["abi"]
    residual_abi = residual["abi"]
    result_abi = result["abi"]
    stats_abi = stats["abi"]
    shape = _static_shape(result_abi, "logical_shape")
    lane_count = int(result_abi.get("scalar_lane_count", 1))
    lane_shape = tuple(result_abi.get("scalar_lane_shape", ()))
    if lane_count <= 0:
        raise CodegenError("AddNormStats requires a positive scalar lane count.")
    for name, abi in (("input", source_abi), ("addend", residual_abi)):
        if (
            int(abi.get("scalar_lane_count", 1)) != lane_count
            or tuple(abi.get("scalar_lane_shape", ())) != lane_shape
        ):
            raise CodegenError(
                "AddNormStats input/addend/result vector lanes must match; "
                f"{name} differs from the result."
            )
    if int(stats_abi.get("scalar_lane_count", 1)) != 1:
        raise CodegenError("AddNormStats statistics must use scalar elements.")
    variant = str(raw["parameters"].get("variant", ""))
    if variant == "local_partial_rms":
        attrs = raw.get("semantic_attrs", {})
        axis = int(attrs.get("axis", -1))
        axis = axis + len(shape) if axis < 0 else axis
        if axis != len(shape) - 1 or bool(attrs.get("use_mean", False)):
            raise CodegenError(
                "AddNormStats local_partial_rms requires an RMS reduction over "
                "the last logical axis."
            )
        if int(stats_abi.get("scalar_lane_count", 1)) != 1:
            raise CodegenError(
                "AddNormStats local partial statistics must use scalar elements."
            )
        stats_distributed = stats_abi.get("distributed_type")
        stats_partial = (
            stats_distributed.get("partial")
            if isinstance(stats_distributed, Mapping)
            else None
        )
        if (
            not isinstance(stats_partial, Mapping)
            or stats_partial.get("reduce_op") != "sum"
        ):
            raise CodegenError(
                "AddNormStats local_partial_rms requires an explicit Sum-partial "
                "statistics result."
            )
        domain = _scalar_local_domain(result_abi, "add_stats_local_offsets")
        local_capacity = int(domain["capacity"])
        for name, abi in (("input", source_abi), ("addend", residual_abi)):
            if (
                int(abi.get("scalar_lane_count", 1)) != lane_count
                or tuple(abi.get("scalar_lane_shape", ()))
                != tuple(result_abi.get("scalar_lane_shape", ()))
            ):
                raise CodegenError(
                    f"AddNormStats local partial {name} vector lanes differ "
                    "from its result."
                )
        lane_coordinate = domain["lane_coordinate"]
        stats_shape = _static_shape(stats_abi, "local_capacity_shape")
        if prod(stats_shape, start=1) != 1:
            raise CodegenError(
                "AddNormStats local_partial_rms currently requires one local "
                "statistics element."
            )
        return {
            "source": _pointer(source),
            "residual": _pointer(residual),
            "result": _pointer(result),
            "stats": _pointer(stats),
            "partials": None,
            "mode": "local_partial_rms",
            "local_capacity": local_capacity,
            "active": domain["active"],
            "source_offset": _access_in_result_domain(
                source_abi, result_abi, domain,
                lane_coordinate=lane_coordinate,
            ),
            "residual_offset": _access_in_result_domain(
                residual_abi, result_abi, domain,
                lane_coordinate=lane_coordinate,
            ),
            "result_offset": emit_local_scalar_offset(
                result_abi,
                domain["local_coordinates"],
                lane_coordinate=lane_coordinate,
            ),
            "stats_offset": emit_local_scalar_offset(
                stats_abi, ("0",) * len(stats_shape)
            ),
            "stats_writer_active": _canonical_writer_active(stats_abi),
            "tile": _bounded_vector_tile(
                raw["parameters"]["tile"],
                local_capacity,
                name="AddNormStats local partial tile",
            ),
            "output_type": _triton_dtype(str(result_abi["scalar_dtype"])),
            "elements": prod(shape),
            "owner_count": int(raw["parameters"].get("owner_count", 1)),
        }
    facts = raw.get("facts", {})
    if not isinstance(facts, Mapping):
        raise CodegenError("AddNormStats implementation facts must be a mapping.")
    participant_scope = str(facts.get("participant_scope", "all_programs"))
    if participant_scope not in {"all_programs", "single_program"}:
        raise CodegenError(
            "AddNormStats participant_scope must be all_programs or single_program."
        )
    return {
        "source": _pointer(source),
        "residual": _pointer(residual),
        "result": _pointer(result),
        "stats": _pointer(stats),
        "partials": None if partials is None else _pointer(partials),
        "elements": prod(shape) * lane_count,
        "tile": int(raw["parameters"]["tile"]),
        "owner_count": int(raw["parameters"].get("owner_count", 1)),
        "mode": "global_rms",
        "participant_active": (
            "(shard_index == 0)"
            if participant_scope == "single_program"
            else "True"
        ),
    }


def _gather_reduce_add_norm_stats_call(raw) -> dict[str, object]:
    partial = _buffer(raw, "inputs", "input")
    residual = _buffer(raw, "inputs", "addend")
    result = _buffer(raw, "outputs", "result_0")
    stats = _buffer(raw, "outputs", "result_1")
    collective = _buffer(raw, "workspaces", "collective")
    stats_partials = _buffer(raw, "workspaces", "norm_stats_partials")
    partial_abi = partial["abi"]
    residual_abi = residual["abi"]
    result_abi = result["abi"]
    stats_abi = stats["abi"]
    logical_shape = _static_shape(partial_abi, "logical_shape")
    if prod(logical_shape[:-1], start=1) != 1:
        raise CodegenError(
            "GatherReduceAddNormStats sum_rms currently requires one logical "
            "outer row; select a row-indexed implementation for batched input."
        )
    if bool(raw.get("semantic_attrs", {}).get("use_mean", False)):
        raise CodegenError("GatherReduceAddNormStats sum_rms does not implement mean.")
    if str(partial_abi.get("storage_kind")) != "compact_per_owner":
        raise CodegenError(
            "GatherReduceAddNormStats requires its partial input to retain one "
            "compact component per placement owner."
        )
    owner_stride = int(partial_abi.get("component_stride_scalar_elements", 0))
    if owner_stride <= 0:
        raise CodegenError(
            "GatherReduceAddNormStats partial input has no owner component stride."
        )
    lane_count = int(partial_abi.get("scalar_lane_count", 1))
    lane_shape = tuple(partial_abi.get("scalar_lane_shape", ()))
    for name, abi in (("addend", residual_abi), ("result", result_abi)):
        if (
            int(abi.get("scalar_lane_count", 1)) != lane_count
            or tuple(abi.get("scalar_lane_shape", ())) != lane_shape
        ):
            raise CodegenError(
                "GatherReduceAddNormStats partial/addend/result vector lanes "
                f"must match; {name} differs from the partial input."
            )
    if int(stats_abi.get("scalar_lane_count", 1)) != 1:
        raise CodegenError(
            "GatherReduceAddNormStats statistics result must be scalar-valued."
        )
    local_shape = _static_shape(partial_abi, "local_capacity_shape")
    scalar_capacity = prod(local_shape, start=1) * lane_count
    flat = "gather_local_offsets"
    physical_flat = flat if lane_count == 1 else f"(({flat}) // {lane_count})"
    lane_coordinate = None if lane_count == 1 else f"(({flat}) % {lane_count})"
    local_coordinates = _unflattened_coordinates(local_shape, physical_flat)
    logical_coordinates = tuple(
        emit_logical_coordinate(partial_abi, axis, local_coordinates)
        for axis in range(len(local_shape))
    )
    active = " & ".join(
        f"(({coordinate}) < ({emit_active_extent(partial_abi, axis)}))"
        for axis, coordinate in enumerate(local_coordinates)
    ) or "True"
    domain = {
        "local_coordinates": local_coordinates,
        "logical_coordinates": logical_coordinates,
    }
    partial_axes = tuple(int(value) for value in raw["parameters"]["partial_axes"])
    partial_owner_count = int(raw["parameters"]["partial_owner_count"])
    placement_owner_count = int(raw["parameters"]["owner_count"])
    work_partition_index, work_partition_count = (
        _mesh_group_member_expression_for_axes(partial_abi, partial_axes)
    )
    if work_partition_count != partial_owner_count:
        raise CodegenError(
            "GatherReduceAddNormStats partial owner count disagrees with its "
            "placement axes."
        )
    stats_distributed = stats_abi.get("distributed_type")
    if not isinstance(stats_distributed, Mapping):
        raise CodegenError(
            "GatherReduceAddNormStats statistics result has no DistributedType ABI."
        )
    stats_placement = stats_distributed.get("placement")
    partial_distributed = partial_abi.get("distributed_type")
    partial_placement = (
        None
        if not isinstance(partial_distributed, Mapping)
        else partial_distributed.get("placement")
    )
    if not isinstance(stats_placement, Mapping) or stats_placement != partial_placement:
        raise CodegenError(
            "GatherReduceAddNormStats value and statistics placements disagree."
        )
    hierarchy = tuple(int(value) for value in stats_placement.get("hierarchy", ()))
    stats_partial = stats_distributed.get("partial")
    stats_preserved_axes = (
        ()
        if not isinstance(stats_partial, Mapping)
        else tuple(int(value) for value in stats_partial.get("axes", ()))
    )
    stats_reduction_axes = tuple(
        axis for axis in range(len(hierarchy)) if axis not in stats_preserved_axes
    )
    stats_reduction_count = prod(
        (hierarchy[axis] for axis in stats_reduction_axes), start=1
    )
    work_tile = _bounded_vector_tile(
        raw["parameters"]["tile"],
        (scalar_capacity + work_partition_count - 1) // work_partition_count,
        name="GatherReduceAddNormStats tile",
    )
    return {
        "partial": emit_storage_pointer(
            partial_abi, str(partial["runtime_argument"])
        ),
        "residual": _pointer(residual),
        "result": _pointer(result),
        "collective": _pointer(collective),
        "stats_partials": _pointer(stats_partials),
        "stats": _pointer(stats),
        "owner_stride": owner_stride,
        "owner_count": placement_owner_count,
        "partial_owner_count": partial_owner_count,
        "placement_owner_count": placement_owner_count,
        "partial_owner": _partial_group_owner_expression_for_axes(
            partial_abi, partial_axes, "gather_partial_lane"
        ),
        "stats_source_owner": _partial_group_owner_expression_for_axes(
            partial_abi, partial_axes, "0"
        ),
        "stats_partial_owner": _group_owner_expression_for_axes(
            stats_abi, stats_reduction_axes, "gather_stats_lane"
        ),
        "stats_reduction_count": stats_reduction_count,
        "stats_reduction_width": _bounded_vector_tile(
            raw["parameters"].get("partial_reduction_width"),
            stats_reduction_count,
            name="GatherReduceAddNormStats statistics reduction_width",
        ),
        "stats_store_active": (
            "True"
            if str(stats_abi.get("storage_kind")) == "compact_per_owner"
            else _distributed_unique_writer_active(stats_abi)
        ),
        "participant_active": _distributed_unique_writer_active(
            partial_abi, allow_redundant_axes=partial_axes
        ),
        "work_partition_index": work_partition_index,
        "work_partition_count": work_partition_count,
        "local_capacity": scalar_capacity,
        "active": active,
        "partial_offset": emit_local_scalar_offset(
            partial_abi,
            local_coordinates,
            lane_coordinate=lane_coordinate,
        ),
        "residual_offset": _access_in_result_domain(
            residual_abi,
            partial_abi,
            domain,
            lane_coordinate=lane_coordinate,
        ),
        "result_offset": _access_in_result_domain(
            result_abi,
            partial_abi,
            domain,
            lane_coordinate=lane_coordinate,
        ),
        "collective_offset": emit_global_scalar_offset(
            collective["abi"],
            logical_coordinates,
            lane_coordinate=lane_coordinate,
        ),
        "stats_offset": emit_global_scalar_offset(
            stats_abi, ("0",) * len(_static_shape(stats_abi, "logical_shape"))
        ),
        "tile": work_tile,
        "partial_reduction_width": _bounded_vector_tile(
            raw["parameters"].get("partial_reduction_width"),
            int(raw["parameters"]["partial_owner_count"]),
            name="GatherReduceAddNormStats partial_reduction_width",
        ),
        "output_type": _triton_dtype(str(result_abi["scalar_dtype"])),
    }


def _gather_reduce_add_norm_apply_call(raw) -> dict[str, object]:
    partial = _buffer(raw, "inputs", "input")
    addend = _buffer(raw, "inputs", "addend")
    scale = _buffer(raw, "inputs", "scale")
    bias = _buffer(raw, "inputs", "bias")
    value_result = _buffer(raw, "outputs", "result_0")
    norm_result = _buffer(raw, "outputs", "result_1")
    stats_partials = _buffer(raw, "workspaces", "norm_stats_partials")
    partial_abi = partial["abi"]
    addend_abi = addend["abi"]
    scale_abi = scale["abi"]
    bias_abi = bias["abi"]
    value_abi = value_result["abi"]
    norm_abi = norm_result["abi"]
    stats_abi = stats_partials["abi"]
    if any(
        str(abi.get("coordinate_space")) != "canonical_global"
        for abi in (value_abi, norm_abi)
    ):
        raise CodegenError(
            "GatherReduceAddNormApply publishes canonical-global results; "
            "private output copies violate its CHIP_WRITE effect contract."
        )
    attrs = raw.get("semantic_attrs", {})
    parameters = raw.get("parameters", {})
    logical_shape = _static_shape(value_abi, "logical_shape")
    axis = int(attrs.get("axis", 0))
    axis = axis + len(logical_shape) if axis < 0 else axis
    if axis < 0 or axis >= len(logical_shape):
        raise CodegenError(
            f"GatherReduceAddNormApply axis {attrs.get('axis')!r} is outside "
            f"rank {len(logical_shape)}."
        )
    if prod(logical_shape[:axis], start=1) != 1:
        raise CodegenError(
            "GatherReduceAddNormApply sum requires one logical outer row; "
            "select a row-indexed implementation for batched input."
        )
    if str(partial_abi.get("storage_kind")) != "compact_per_owner":
        raise CodegenError(
            "GatherReduceAddNormApply requires compact per-owner partial input."
        )
    owner_stride = int(partial_abi.get("component_stride_scalar_elements", 0))
    if owner_stride <= 0:
        raise CodegenError(
            "GatherReduceAddNormApply partial input has no owner component stride."
        )
    distributed = partial_abi.get("distributed_type")
    if not isinstance(distributed, Mapping):
        raise CodegenError(
            "GatherReduceAddNormApply partial input has no distributed ABI."
        )
    partial_description = distributed.get("partial")
    placement = distributed.get("placement")
    if not isinstance(partial_description, Mapping) or not isinstance(placement, Mapping):
        raise CodegenError(
            "GatherReduceAddNormApply requires a partial distribution and placement."
        )
    if str(partial_description.get("reduce_op")) != "sum":
        raise CodegenError("GatherReduceAddNormApply only supports additive partials.")
    partial_axes = tuple(int(value) for value in parameters.get("partial_axes", ()))
    hierarchy = tuple(int(value) for value in placement.get("hierarchy", ()))
    if (
        partial_axes != tuple(int(value) for value in partial_description.get("axes", ()))
        or not partial_axes
        or tuple(sorted(set(partial_axes))) != partial_axes
        or any(index < 0 or index >= len(hierarchy) for index in partial_axes)
    ):
        raise CodegenError(
            "GatherReduceAddNormApply has inconsistent partial placement axes."
        )
    owner_count = int(parameters.get("owner_count", 0))
    partial_owner_count = int(parameters.get("partial_owner_count", 0))
    if owner_count != prod(hierarchy, start=1) or partial_owner_count != prod(
        (hierarchy[index] for index in partial_axes), start=1
    ):
        raise CodegenError(
            "GatherReduceAddNormApply owner counts disagree with its placement."
        )
    use_mean = bool(attrs.get("use_mean", False))
    stats_components = 2 if use_mean else 1
    if (
        _static_shape(stats_abi, "logical_shape") != (stats_components, owner_count)
        or str(stats_abi.get("scalar_dtype")) != "float32"
        or int(stats_abi.get("scalar_lane_count", 1)) != 1
    ):
        raise CodegenError(
            "GatherReduceAddNormApply statistics workspace must be float32 "
            f"[{stats_components}, {owner_count}]."
        )
    lane_count = int(partial_abi.get("scalar_lane_count", 1))
    lane_shape = tuple(partial_abi.get("scalar_lane_shape", ()))
    for name, abi in (
        ("addend", addend_abi),
        ("value result", value_abi),
        ("norm result", norm_abi),
    ):
        if (
            int(abi.get("scalar_lane_count", 1)) != lane_count
            or tuple(abi.get("scalar_lane_shape", ())) != lane_shape
        ):
            raise CodegenError(
                "GatherReduceAddNormApply partial/addend/results vector lanes "
                f"must match; {name} differs from the partial input."
            )
    for name, abi in (("scale", scale_abi), ("bias", bias_abi)):
        if (
            int(abi.get("scalar_lane_count", 1)) != lane_count
            or tuple(abi.get("scalar_lane_shape", ())) != lane_shape
        ):
            raise CodegenError(
                "GatherReduceAddNormApply scale/bias vector lanes must match "
                f"the result; {name} differs."
            )

    partial_local_shape = _static_shape(partial_abi, "local_capacity_shape")
    value_local_shape = _static_shape(value_abi, "local_capacity_shape")
    partial_scalar_capacity = prod(partial_local_shape, start=1) * lane_count
    value_scalar_capacity = prod(value_local_shape, start=1) * lane_count
    # A materialization may either gather a non-partial source split into a
    # broader result (attention projection), or scatter a fully partial source
    # into a sharded result (MLP down projection).  Assign work using the finer
    # of the two reviewed local domains; addressing below proves that every
    # other buffer can be represented in that domain.
    domain_abi = (
        partial_abi
        if partial_scalar_capacity <= value_scalar_capacity
        else value_abi
    )
    local_shape = _static_shape(domain_abi, "local_capacity_shape")
    scalar_capacity = prod(local_shape, start=1) * lane_count
    flat = "gather_local_offsets"
    physical_flat = flat if lane_count == 1 else f"(({flat}) // {lane_count})"
    lane_coordinate = None if lane_count == 1 else f"(({flat}) % {lane_count})"
    local_coordinates = _unflattened_coordinates(local_shape, physical_flat)
    logical_coordinates = tuple(
        emit_logical_coordinate(domain_abi, index, local_coordinates)
        for index in range(len(local_shape))
    )
    active = " & ".join(
        f"(({coordinate}) < ({emit_active_extent(domain_abi, index)}))"
        for index, coordinate in enumerate(local_coordinates)
    ) or "True"
    domain = {
        "local_coordinates": local_coordinates,
        "logical_coordinates": logical_coordinates,
    }
    # These checks also prove that local output coordinates can be addressed
    # from the partial input iteration domain.
    addend_offset = _access_in_result_domain(
        addend_abi, domain_abi, domain, lane_coordinate=lane_coordinate
    )
    value_offset = _access_in_result_domain(
        value_abi, domain_abi, domain, lane_coordinate=lane_coordinate
    )
    norm_offset = _access_in_result_domain(
        norm_abi, domain_abi, domain, lane_coordinate=lane_coordinate
    )
    partial_offset = (
        emit_local_scalar_offset(
            partial_abi,
            local_coordinates,
            lane_coordinate=lane_coordinate,
        )
        if domain_abi is partial_abi
        else _contiguous_local_offset_in_domain(
            partial_abi,
            domain,
            lane_coordinate=lane_coordinate,
        )
    )
    domain_distributed = domain_abi["distributed_type"]
    domain_split_axes = {
        int(axis)
        for policy in domain_distributed["axis_policies"]
        if policy.get("kind") == "split"
        for stage in policy["stages"]
        for axis in stage["hierarchy_axes"]
    }
    # Divide distinct canonical element writes among otherwise redundant
    # partial owners. Axes already splitting the domain are excluded.
    partition_axes = tuple(
        axis for axis in partial_axes if axis not in domain_split_axes
    )
    work_partition_index, work_partition_count = (
        _mesh_group_member_expression_for_axes(domain_abi, partition_axes)
        if partition_axes else ("0", 1)
    )
    stats_writer_index, stats_owner_count = _distributed_unique_writer_index(
        domain_abi, allow_redundant_axes=partition_axes,
    )
    return {
        "partial": emit_storage_pointer(
            partial_abi, str(partial["runtime_argument"])
        ),
        "addend": _pointer(addend),
        "scale": _pointer(scale),
        "bias": _pointer(bias),
        "value_result": _pointer(value_result),
        "norm_result": _pointer(norm_result),
        "stats_partials": _pointer(stats_partials),
        "owner_stride": owner_stride,
        "owner_count": owner_count,
        "partial_owner_count": partial_owner_count,
        "placement_owner_count": owner_count,
        "partial_owner": _partial_group_owner_expression_for_axes(
            partial_abi, partial_axes, "gather_partial_lane"
        ),
        "participant_active": _distributed_unique_writer_active(
            domain_abi, allow_redundant_axes=partition_axes,
        ),
        "work_partition_index": work_partition_index,
        "work_partition_count": work_partition_count,
        "stats_writer_index": stats_writer_index,
        "stats_owner_count": stats_owner_count,
        "local_capacity": scalar_capacity,
        "active": active,
        "partial_offset": partial_offset,
        "addend_offset": addend_offset,
        "value_offset": value_offset,
        "norm_offset": norm_offset,
        "scale_offset": _norm_apply_parameter_offset(
            scale_abi,
            value_abi,
            axis,
            local_coordinates,
            logical_coordinates,
            lane_coordinate,
            "scale",
        ),
        "bias_offset": _norm_apply_parameter_offset(
            bias_abi,
            value_abi,
            axis,
            local_coordinates,
            logical_coordinates,
            lane_coordinate,
            "bias",
        ),
        "normalization_size": prod(logical_shape[axis:], start=1) * lane_count,
        "tile": _bounded_vector_tile(
            parameters.get("tile"),
            (scalar_capacity + work_partition_count - 1) // work_partition_count,
            name="GatherReduceAddNormApply tile",
        ),
        "reduction_width": _bounded_vector_tile(
            parameters.get("reduction_width"),
            stats_owner_count,
            name="GatherReduceAddNormApply reduction_width",
        ),
        "partial_reduction_width": _bounded_vector_tile(
            parameters.get("partial_reduction_width"),
            partial_owner_count,
            name="GatherReduceAddNormApply partial_reduction_width",
        ),
        "epsilon": repr(float(attrs.get("epsilon", 0.0))),
        "use_mean": use_mean,
        "has_bias": bool(attrs.get("has_bias", True)),
        "stats_square_base": owner_count if use_mean else 0,
        "round_before_scale": bool(attrs.get("round_before_scale", False)),
        "output_type": _triton_dtype(str(value_abi["scalar_dtype"])),
        "norm_output_type": _triton_dtype(str(norm_abi["scalar_dtype"])),
    }


def _norm_apply_call(raw) -> dict[str, object]:
    return _norm_apply_context(raw, stats_formal="stats")


def _norm_apply_context(
    raw, *, stats_formal: str
) -> dict[str, object]:
    source = _buffer(raw, "inputs", "input")
    stats = _buffer(raw, "inputs", stats_formal)
    scale = _buffer(raw, "inputs", "scale")
    bias = _buffer(raw, "inputs", "bias")
    result = _buffer(raw, "outputs", "result")
    source_abi = source["abi"]
    stats_abi = stats["abi"]
    scale_abi = scale["abi"]
    bias_abi = bias["abi"]
    result_abi = result["abi"]
    logical_shape = _static_shape(result_abi, "logical_shape")
    attrs = raw.get("semantic_attrs", {})
    axis = int(attrs.get("axis", 0))
    axis = axis + len(logical_shape) if axis < 0 else axis
    if axis < 0 or axis >= len(logical_shape):
        raise CodegenError(
            f"TIR NormApply axis {attrs.get('axis')!r} is outside rank "
            f"{len(logical_shape)}."
        )
    use_mean = bool(attrs.get("use_mean", False))
    lane_count = int(result_abi.get("scalar_lane_count", 1))
    lane_shape = tuple(result_abi.get("scalar_lane_shape", ()))
    for name, abi in (
        ("input", source_abi), ("scale", scale_abi), ("bias", bias_abi),
    ):
        if (
            int(abi.get("scalar_lane_count", 1)) != lane_count
            or tuple(abi.get("scalar_lane_shape", ())) != lane_shape
        ):
            raise CodegenError(
                "TIR NormApply input/scale/bias/result vector lanes must "
                f"match; {name} differs from the result."
            )
    if int(stats_abi.get("scalar_lane_count", 1)) != 1:
        raise CodegenError("TIR NormApply statistics must use scalar elements.")

    local_shape = _static_shape(result_abi, "local_capacity_shape")
    scalar_capacity = prod(local_shape, start=1) * lane_count
    flat = "norm_apply_local_offsets"
    physical_flat = flat if lane_count == 1 else f"(({flat}) // {lane_count})"
    lane_coordinate = None if lane_count == 1 else f"(({flat}) % {lane_count})"
    local_coordinates = _unflattened_coordinates(local_shape, physical_flat)
    logical_coordinates = tuple(
        emit_logical_coordinate(result_abi, index, local_coordinates)
        for index in range(len(local_shape))
    )
    active = " & ".join(
        f"(({coordinate}) < ({emit_active_extent(result_abi, index)}))"
        for index, coordinate in enumerate(local_coordinates)
    ) or "True"
    domain = {
        "local_coordinates": local_coordinates,
        "logical_coordinates": logical_coordinates,
    }
    stats_offsets = _norm_apply_stats_offsets(
        stats_abi,
        result_abi,
        axis,
        local_coordinates,
        logical_coordinates,
        use_mean,
    )
    return {
        "source": _pointer(source),
        "stats": _pointer(stats),
        "scale": _pointer(scale),
        "bias": _pointer(bias),
        "result": _pointer(result),
        "axis": axis,
        "use_mean": use_mean,
        "lane_coordinate": lane_coordinate,
        "round_before_scale": bool(attrs.get("round_before_scale", False)),
        "local_capacity": scalar_capacity,
        "active": active,
        "writer_active": _canonical_writer_active(result_abi),
        "source_offset": _access_in_result_domain(
            source_abi, result_abi, domain,
            lane_coordinate=lane_coordinate,
        ),
        "stats_sum_offset": stats_offsets[0] if use_mean else None,
        "stats_square_sum_offset": stats_offsets[-1],
        "scale_offset": _norm_apply_parameter_offset(
            scale_abi,
            result_abi,
            axis,
            local_coordinates,
            logical_coordinates,
            lane_coordinate,
            "scale",
        ),
        "bias_offset": _norm_apply_parameter_offset(
            bias_abi,
            result_abi,
            axis,
            local_coordinates,
            logical_coordinates,
            lane_coordinate,
            "bias",
        ),
        "result_offset": emit_local_scalar_offset(
            result_abi,
            local_coordinates,
            lane_coordinate=lane_coordinate,
        ),
        "normalization_size": prod(logical_shape[axis:], start=1) * lane_count,
        "tile": _bounded_vector_tile(
            raw["parameters"]["block_size"],
            scalar_capacity,
            name="NormApply block_size",
        ),
        "epsilon": repr(float(attrs.get("epsilon", 0.0))),
        "output_type": _triton_dtype(str(result_abi["scalar_dtype"])),
    }


def _gather_reduce_norm_apply_call(raw) -> dict[str, object]:
    partial = _buffer(raw, "inputs", "partial_stats")
    source = _buffer(raw, "inputs", "input")
    scale = _buffer(raw, "inputs", "scale")
    bias = _buffer(raw, "inputs", "bias")
    result = _buffer(raw, "outputs", "result")
    partial_abi = partial["abi"]
    source_abi = source["abi"]
    scale_abi = scale["abi"]
    bias_abi = bias["abi"]
    result_abi = result["abi"]
    if str(partial_abi.get("storage_kind")) != "compact_per_owner":
        raise CodegenError(
            "GatherReduceNormApply requires compact per-owner partial statistics."
        )
    if int(partial_abi.get("scalar_lane_count", 1)) != 1:
        raise CodegenError(
            "GatherReduceNormApply partial statistics must use scalar elements."
        )
    owner_stride = int(
        partial_abi.get("component_stride_scalar_elements", 0)
    )
    if owner_stride <= 0:
        raise CodegenError(
            "GatherReduceNormApply partial statistics require distinct owner storage."
        )
    distributed = partial_abi.get("distributed_type")
    if not isinstance(distributed, Mapping):
        raise CodegenError(
            "GatherReduceNormApply partial statistics have no distributed ABI."
        )
    partial_description = distributed.get("partial")
    placement = distributed.get("placement")
    if not isinstance(partial_description, Mapping) or not isinstance(
        placement, Mapping
    ):
        raise CodegenError(
            "GatherReduceNormApply requires a partial distribution and placement."
        )
    if str(partial_description.get("reduce_op")) != "sum":
        raise CodegenError(
            "GatherReduceNormApply only supports additive partial statistics."
        )
    partial_axes = tuple(
        int(value) for value in partial_description.get("axes", ())
    )
    hierarchy = tuple(int(value) for value in placement.get("hierarchy", ()))
    if (
        not partial_axes
        or len(set(partial_axes)) != len(partial_axes)
        or any(axis < 0 or axis >= len(hierarchy) for axis in partial_axes)
    ):
        raise CodegenError(
            "GatherReduceNormApply has invalid partial placement axes."
        )
    parameters = raw.get("parameters", {})
    reduction_width = int(parameters.get("reduction_width", 0))
    if reduction_width <= 0:
        raise CodegenError(
            "GatherReduceNormApply reduction_width must be positive."
        )
    attrs = raw.get("semantic_attrs", {})
    logical_shape = _static_shape(result_abi, "logical_shape")
    axis = int(attrs.get("axis", 0))
    axis = axis + len(logical_shape) if axis < 0 else axis
    if axis < 0 or axis >= len(logical_shape):
        raise CodegenError(
            f"GatherReduceNormApply axis {attrs.get('axis')!r} is outside "
            f"rank {len(logical_shape)}."
        )
    use_mean = bool(attrs.get("use_mean", False))
    lane_count = int(result_abi.get("scalar_lane_count", 1))
    lane_shape = tuple(result_abi.get("scalar_lane_shape", ()))
    for name, abi in (
        ("input", source_abi), ("scale", scale_abi), ("bias", bias_abi),
    ):
        if (
            int(abi.get("scalar_lane_count", 1)) != lane_count
            or tuple(abi.get("scalar_lane_shape", ())) != lane_shape
        ):
            raise CodegenError(
                "GatherReduceNormApply input/scale/bias/result vector lanes "
                f"must match; {name} differs from the result."
            )

    local_shape = _static_shape(result_abi, "local_capacity_shape")
    outer_shape = local_shape[:axis]
    inner_shape = local_shape[axis:]
    outer_coordinates = _unflattened_coordinates(
        outer_shape, "norm_outer_index"
    )
    physical_inner = (
        "norm_apply_inner_offsets"
        if lane_count == 1
        else f"((norm_apply_inner_offsets) // {lane_count})"
    )
    lane_coordinate = (
        None
        if lane_count == 1
        else f"((norm_apply_inner_offsets) % {lane_count})"
    )
    inner_coordinates = _unflattened_coordinates(inner_shape, physical_inner)
    local_coordinates = (*outer_coordinates, *inner_coordinates)
    logical_coordinates = tuple(
        emit_logical_coordinate(result_abi, index, local_coordinates)
        for index in range(len(local_shape))
    )
    outer_active = " & ".join(
        f"(({coordinate}) < ({emit_active_extent(result_abi, index)}))"
        for index, coordinate in enumerate(outer_coordinates)
    ) or "True"
    inner_active = " & ".join(
        f"(({coordinate}) < ({emit_active_extent(result_abi, axis + index)}))"
        for index, coordinate in enumerate(inner_coordinates)
    ) or "True"
    domain = {
        "local_coordinates": local_coordinates,
        "logical_coordinates": logical_coordinates,
    }
    stats_local_coordinates = (
        *outer_coordinates,
        *("0" for _ in inner_shape),
    )
    stats_logical_coordinates = tuple(
        emit_logical_coordinate(
            result_abi, index, stats_local_coordinates
        )
        for index in range(len(local_shape))
    )
    stats_offsets = _norm_apply_stats_offsets(
        partial_abi,
        result_abi,
        axis,
        stats_local_coordinates,
        stats_logical_coordinates,
        use_mean,
    )
    scalar_inner_capacity = prod(inner_shape, start=1) * lane_count
    call = {
        "source": _pointer(source),
        "scale": _pointer(scale),
        "bias": _pointer(bias),
        "result": _pointer(result),
        "partial_stats": emit_storage_pointer(
            partial_abi, str(partial["runtime_argument"])
        ),
        "partial_owner": _partial_group_owner_expression_for_axes(
            partial_abi, partial_axes, "norm_partial_lane"
        ),
        "partial_owner_count": prod(
            (hierarchy[axis] for axis in partial_axes), start=1
        ),
        "placement_owner_count": prod(hierarchy, start=1),
        "owner_stride": owner_stride,
        "reduction_width": reduction_width,
        "has_bias": bool(attrs.get("has_bias", True)),
        "axis": axis,
        "use_mean": use_mean,
        "local_outer_capacity": prod(outer_shape, start=1),
        "round_before_scale": bool(attrs.get("round_before_scale", False)),
        "local_inner_capacity": scalar_inner_capacity,
        "outer_active": outer_active,
        "active": inner_active,
        "writer_active": _canonical_writer_active(result_abi),
        "partial_stats_sum_offset": stats_offsets[0] if use_mean else None,
        "partial_stats_square_sum_offset": stats_offsets[-1],
        "source_offset": _access_in_result_domain(
            source_abi,
            result_abi,
            domain,
            lane_coordinate=lane_coordinate,
        ),
        "scale_offset": _norm_apply_parameter_offset(
            scale_abi,
            result_abi,
            axis,
            local_coordinates,
            logical_coordinates,
            lane_coordinate,
            "scale",
        ),
        "bias_offset": _norm_apply_parameter_offset(
            bias_abi,
            result_abi,
            axis,
            local_coordinates,
            logical_coordinates,
            lane_coordinate,
            "bias",
        ),
        "result_offset": emit_local_scalar_offset(
            result_abi,
            local_coordinates,
            lane_coordinate=lane_coordinate,
        ),
        "normalization_size": (
            prod(logical_shape[axis:], start=1) * lane_count
        ),
        "tile": _bounded_vector_tile(
            parameters["block_size"],
            scalar_inner_capacity,
            name="GatherReduceNormApply block_size",
        ),
        "epsilon": repr(float(attrs.get("epsilon", 0.0))),
        "output_type": _triton_dtype(str(result_abi["scalar_dtype"])),
    }
    return call


def _norm_apply_stats_offsets(
    stats_abi,
    result_abi,
    axis,
    local_coordinates,
    logical_coordinates,
    use_mean,
) -> tuple[str, ...]:
    result_shape = _static_shape(result_abi, "logical_shape")
    stats_shape = _static_shape(stats_abi, "logical_shape")
    count = 2 if use_mean else 1
    expected = (count, *result_shape[:axis], *(1 for _ in result_shape[axis:]))
    if stats_shape != expected:
        raise CodegenError(
            "TIR NormApply statistics do not match its normalized suffix: "
            f"expected {expected}, got {stats_shape}."
        )
    suffix_zeros = ("0",) * (len(result_shape) - axis)
    if str(stats_abi["coordinate_space"]) == "canonical_global":
        return tuple(
            emit_global_scalar_offset(
                stats_abi,
                (str(component), *logical_coordinates[:axis], *suffix_zeros),
            )
            for component in range(count)
        )
    stats_local = _static_shape(stats_abi, "local_capacity_shape")
    expected_local = (
        count,
        *_static_shape(result_abi, "local_capacity_shape")[:axis],
        *(1 for _ in result_shape[axis:]),
    )
    if stats_local != expected_local or not _same_axis_mapping(
        stats_abi, result_abi, range(1, axis + 1), range(axis)
    ):
        raise CodegenError(
            "NormApply local statistics are not aligned with the result "
            "outer axes; insert an explicit Boxing before code generation."
        )
    return tuple(
        emit_local_scalar_offset(
            stats_abi,
            (str(component), *local_coordinates[:axis], *suffix_zeros),
        )
        for component in range(count)
    )


def _norm_apply_parameter_offset(
    parameter_abi,
    result_abi,
    axis,
    local_coordinates,
    logical_coordinates,
    lane_coordinate,
    name,
) -> str:
    result_shape = _static_shape(result_abi, "logical_shape")
    parameter_shape = _static_shape(parameter_abi, "logical_shape")
    suffix_shape = result_shape[axis:]
    if len(parameter_shape) != len(suffix_shape) or any(
        actual not in {1, expected}
        for actual, expected in zip(parameter_shape, suffix_shape, strict=True)
    ):
        raise CodegenError(
            f"TIR NormApply {name} is not broadcastable to normalized suffix "
            f"{suffix_shape}: got {parameter_shape}."
        )
    global_parameter_coordinates = tuple(
        "0" if extent == 1 else logical_coordinates[axis + index]
        for index, extent in enumerate(parameter_shape)
    )
    if str(parameter_abi["coordinate_space"]) == "canonical_global":
        return emit_global_scalar_offset(
            parameter_abi,
            global_parameter_coordinates,
            lane_coordinate=lane_coordinate,
        )
    parameter_local = _static_shape(parameter_abi, "local_capacity_shape")
    result_local = _static_shape(result_abi, "local_capacity_shape")
    local_parameter_coordinates = tuple(
        "0" if extent == 1 else local_coordinates[axis + index]
        for index, extent in enumerate(parameter_shape)
    )
    for index, extent in enumerate(parameter_shape):
        if extent == 1:
            if parameter_local[index] != 1:
                raise CodegenError(
                    f"TIR NormApply broadcast {name} axis {index} has a "
                    "non-unit local extent."
                )
            continue
        result_axis = axis + index
        if (
            parameter_local[index] != result_local[result_axis]
            or not _same_axis_mapping(
                parameter_abi, result_abi, (index,), (result_axis,)
            )
        ):
            raise CodegenError(
                f"NormApply {name} axis {index} is not locally aligned with "
                "the result; insert an explicit Boxing before code generation."
            )
    return emit_local_scalar_offset(
        parameter_abi,
        local_parameter_coordinates,
        lane_coordinate=lane_coordinate,
    )


def _elementwise_call(raw) -> dict[str, object]:
    variant = str(raw.get("variant", raw.get("parameters", {}).get("variant", "")))
    binary = variant in {"add", "mul"}
    lhs = _buffer(raw, "inputs", "lhs" if binary else "value")
    rhs = _buffer(raw, "inputs", "rhs") if binary else None
    result = _buffer(raw, "outputs", "result")
    result_abi = result["abi"]
    schedule = raw["parameters"]["vector_schedule"]
    physical = schedule.get("physical", {}) if isinstance(schedule, Mapping) else {}
    domain = _elementwise_domain(result_abi, schedule, "local_offsets")
    lhs_offset, lhs_active = _elementwise_operand_access(
        lhs["abi"], result_abi, domain, raw
    )
    if rhs is None:
        rhs_offset = None
        rhs_active = "True"
    else:
        rhs_offset, rhs_active = _elementwise_operand_access(
            rhs["abi"], result_abi, domain, raw
        )
    active = " & ".join(
        f"({value})" for value in (domain["active"], lhs_active, rhs_active)
        if value != "True"
    ) or "True"
    return {
        "lhs": _pointer(lhs),
        "rhs": None if rhs is None else _pointer(rhs),
        "result": _pointer(result),
        "local_capacity": domain["capacity"],
        "logical_capacity": domain["logical_capacity"],
        "logical_shape": domain["logical_shape"],
        "padded_shape": domain["padded_shape"],
        "vector_axes": domain["vector_axes"],
        "active": active,
        "lhs_offset": lhs_offset,
        "rhs_offset": rhs_offset,
        "result_offset": emit_local_scalar_offset(
            result_abi,
            domain["local_coordinates"],
            lane_coordinate=domain.get("lane_coordinate"),
        ),
        "tile": int(physical["elements_per_program"]),
        "output_type": _triton_dtype(str(result_abi["scalar_dtype"])),
    }


def _elementwise_domain(
    abi: Mapping[str, object],
    schedule: object,
    flat_coordinate: str,
) -> dict[str, object]:
    shape = _static_shape(abi, "local_capacity_shape")
    contract = schedule.get("contract", {}) if isinstance(schedule, Mapping) else {}
    if not isinstance(contract, Mapping):
        raise CodegenError("Elementwise vector schedule has no contract mapping.")
    kind = str(contract.get("kind", "scalar"))
    axes = tuple(int(value) for value in contract.get("axes", ()))
    lanes = tuple(int(value) for value in contract.get("lanes", ()))
    if kind == "scalar":
        if axes or lanes:
            raise CodegenError("Scalar elementwise schedule cannot carry vector axes.")
    elif kind == "axes":
        if len(axes) != len(lanes) or len(set(axes)) != len(axes):
            raise CodegenError("Elementwise vector axes and lanes are inconsistent.")
        if any(axis < 0 or axis >= len(shape) for axis in axes):
            raise CodegenError("Elementwise vector axis is outside the local rank.")
        if any(lane <= 1 for lane in lanes):
            raise CodegenError("Elementwise vector lanes must be greater than one.")
    else:
        raise CodegenError(
            f"Elementwise call cannot consume vector schedule kind {kind!r}."
        )
    abi_lanes = tuple(int(value) for value in abi.get("scalar_lane_shape", ()))
    abi_lane_count = int(abi.get("scalar_lane_count", 1))
    if abi_lanes:
        if kind != "axes" or lanes != abi_lanes:
            raise CodegenError(
                "A typed-vector elementwise result must agree with its selected "
                "vector axes/lanes contract."
            )
        scalar = _scalar_local_domain(abi, flat_coordinate)
        return {
            **scalar,
            "logical_capacity": prod(shape) * abi_lane_count,
            "logical_shape": shape,
            "padded_shape": shape,
            "vector_axes": axes,
        }
    padded = list(shape)
    for axis, lane in zip(axes, lanes, strict=True):
        padded[axis] = ((padded[axis] + lane - 1) // lane) * lane
    padded_shape = tuple(padded)
    local_coordinates = _unflattened_coordinates(
        padded_shape, flat_coordinate
    )
    active = " & ".join(
        f"(({coordinate}) < ({emit_active_extent(abi, axis)}))"
        for axis, coordinate in enumerate(local_coordinates)
    ) or "True"
    logical_coordinates = tuple(
        emit_logical_coordinate(abi, axis, local_coordinates)
        for axis in range(len(shape))
    )
    return {
        "capacity": prod(padded_shape),
        "logical_capacity": prod(shape),
        "logical_shape": shape,
        "padded_shape": padded_shape,
        "vector_axes": axes,
        "local_coordinates": local_coordinates,
        "logical_coordinates": logical_coordinates,
        "active": active,
    }


def _elementwise_operand_access(
    operand_abi: Mapping[str, object],
    result_abi: Mapping[str, object],
    domain: Mapping[str, object],
    raw: Mapping[str, object],
) -> tuple[str, str]:
    """Map one scalar element of the result domain to an operand layout."""

    result_lanes = tuple(int(value) for value in result_abi.get("scalar_lane_shape", ()))
    operand_lanes = tuple(int(value) for value in operand_abi.get("scalar_lane_shape", ()))
    if result_lanes == operand_lanes:
        return (
            _access_in_result_domain(
                operand_abi,
                result_abi,
                domain,
                lane_coordinate=domain.get("lane_coordinate"),
            ),
            "True",
        )
    variant = str(raw.get("variant", raw.get("parameters", {}).get("variant", "")))
    attrs = raw.get("semantic_attrs", {})
    if variant != "cast" or not isinstance(attrs, Mapping):
        raise CodegenError(
            "Elementwise operands with different VectorType lanes require an "
            "explicit vectorized cast contract."
        )
    axes = tuple(int(value) for value in attrs.get("vectorize_axes", ()))
    return _vectorized_cast_operand_access(
        operand_abi,
        result_abi,
        domain,
        axes,
    )


def _vectorized_cast_operand_access(
    operand_abi: Mapping[str, object],
    result_abi: Mapping[str, object],
    domain: Mapping[str, object],
    axes: tuple[int, ...],
) -> tuple[str, str]:
    """Repack a scalar coordinate between two VectorType element layouts."""

    operand_shape = _static_shape(operand_abi, "local_capacity_shape")
    result_shape = _static_shape(result_abi, "local_capacity_shape")
    operand_lanes = tuple(int(value) for value in operand_abi.get("scalar_lane_shape", ()))
    result_lanes = tuple(int(value) for value in result_abi.get("scalar_lane_shape", ()))
    if (
        len(operand_shape) != len(result_shape)
        or len(axes) != len(operand_lanes)
        or len(axes) != len(result_lanes)
        or len(set(axes)) != len(axes)
        or any(axis < 0 or axis >= len(result_shape) for axis in axes)
    ):
        raise CodegenError("Vectorized cast ABI has inconsistent axes, rank, or lanes.")
    if (
        prod(operand_shape) * prod(operand_lanes)
        != prod(result_shape) * prod(result_lanes)
    ):
        raise CodegenError(
            "Vectorized cast input and result do not cover the same local scalar extent."
        )
    result_lane = domain.get("lane_coordinate")
    if not isinstance(result_lane, str):
        raise CodegenError("Vectorized cast result has no scalar lane coordinate.")
    result_components = _lane_components(result_lane, result_lanes)
    input_coordinates, input_components = _repack_vector_coordinates(
        tuple(domain["local_coordinates"]),
        axes,
        operand_lanes,
        result_lanes,
        result_components,
    )
    input_lane = _flatten_lane_components(input_components, operand_lanes)
    active = " & ".join(
        f"(({coordinate}) < ({emit_active_extent(operand_abi, axis)}))"
        for axis, coordinate in enumerate(input_coordinates)
    ) or "True"
    if str(operand_abi.get("coordinate_space")) == "canonical_global":
        global_coordinates, _ = _repack_vector_coordinates(
            tuple(domain["logical_coordinates"]),
            axes,
            operand_lanes,
            result_lanes,
            result_components,
        )
        offset = emit_global_scalar_offset(
            operand_abi,
            global_coordinates,
            lane_coordinate=input_lane,
        )
    else:
        offset = emit_local_scalar_offset(
            operand_abi,
            input_coordinates,
            lane_coordinate=input_lane,
        )
    return offset, active


def _repack_vector_coordinates(
    base_coordinates: tuple[str, ...],
    axes: tuple[int, ...],
    input_lanes: tuple[int, ...],
    output_lanes: tuple[int, ...],
    output_components: tuple[str, ...],
) -> tuple[tuple[str, ...], list[str]]:
    coordinates = list(base_coordinates)
    components: list[str] = []
    for axis, input_lane, output_lane, output_component in zip(
        axes, input_lanes, output_lanes, output_components, strict=True
    ):
        scalar_coordinate = (
            f"(({base_coordinates[axis]}) * {output_lane} + "
            f"({output_component}))"
        )
        coordinates[axis] = f"(({scalar_coordinate}) // {input_lane})"
        components.append(f"(({scalar_coordinate}) % {input_lane})")
    return tuple(coordinates), components


def _lane_components(flat: str, lanes: tuple[int, ...]) -> tuple[str, ...]:
    values: list[str] = []
    for index, lane in enumerate(lanes):
        suffix = prod(lanes[index + 1 :], start=1)
        values.append(
            f"(({flat}) % {lane})"
            if suffix == 1
            else f"((({flat}) // {suffix}) % {lane})"
        )
    return tuple(values)


def _flatten_lane_components(
    components: list[str], lanes: tuple[int, ...]
) -> str:
    terms = []
    for index, component in enumerate(components):
        stride = prod(lanes[index + 1 :], start=1)
        terms.append(component if stride == 1 else f"({component}) * {stride}")
    return " + ".join(terms) if terms else "0"


def _norm_stats_call(raw) -> dict[str, object]:
    source = _buffer(raw, "inputs", "input")
    result = _buffer(raw, "outputs", "result")
    source_abi = source["abi"]
    result_abi = result["abi"]
    source_shape = _static_shape(source_abi, "local_capacity_shape")
    result_shape = _static_shape(result_abi, "local_capacity_shape")
    attrs = raw.get("semantic_attrs", {})
    axis = int(attrs.get("axis", 0))
    axis = axis + len(source_shape) if axis < 0 else axis
    if axis < 0 or axis >= len(source_shape):
        raise CodegenError(
            f"TIR NormStats axis {attrs.get('axis')!r} is outside rank "
            f"{len(source_shape)}."
        )
    use_mean = bool(attrs.get("use_mean", False))
    stats_count = 2 if use_mean else 1
    expected_result_shape = (
        stats_count, *source_shape[:axis],
        *(1 for _ in source_shape[axis:]),
    )
    if result_shape != expected_result_shape:
        raise CodegenError(
            "TIR NormStats result local shape does not preserve the input "
            f"prefix: expected {expected_result_shape}, got {result_shape}."
        )
    lane_count = int(source_abi.get("scalar_lane_count", 1))
    if lane_count <= 0:
        raise CodegenError("TIR NormStats requires a positive scalar lane count.")
    if int(result_abi.get("scalar_lane_count", 1)) != 1:
        raise CodegenError("TIR NormStats result must have a scalar element type.")

    outer_shape = source_shape[:axis]
    reduction_shape = source_shape[axis:]
    outer_coordinates = _unflattened_coordinates(
        outer_shape, "norm_outer_index"
    )
    physical_reduction = (
        "norm_reduction_offsets"
        if lane_count == 1
        else f"((norm_reduction_offsets) // {lane_count})"
    )
    reduction_coordinates = _unflattened_coordinates(
        reduction_shape, physical_reduction
    )
    source_coordinates = (*outer_coordinates, *reduction_coordinates)
    source_active = " & ".join(
        f"(({coordinate}) < ({emit_active_extent(source_abi, index)}))"
        for index, coordinate in enumerate(source_coordinates)
    ) or "True"
    outer_active = " & ".join(
        f"(({coordinate}) < ({emit_active_extent(source_abi, index)}))"
        for index, coordinate in enumerate(outer_coordinates)
    ) or "True"
    lane_coordinate = (
        None
        if lane_count == 1
        else f"((norm_reduction_offsets) % {lane_count})"
    )
    suffix_zeros = ("0",) * (len(source_shape) - axis)
    result_offsets = tuple(
        emit_local_scalar_offset(
            result_abi,
            (str(component), *outer_coordinates, *suffix_zeros),
        )
        for component in range(stats_count)
    )
    reduction_capacity = prod(reduction_shape, start=1) * lane_count
    return {
        "source": _pointer(source),
        "result": _pointer(result),
        "partial_stats": None,
        "axis": axis,
        "use_mean": use_mean,
        "lane_count": lane_count,
        "outer_capacity": prod(outer_shape, start=1),
        "reduction_capacity": reduction_capacity,
        "source_active": source_active,
        "outer_active": outer_active,
        "writer_active": _canonical_writer_active(result_abi),
        "source_offset": emit_local_scalar_offset(
            source_abi,
            source_coordinates,
            lane_coordinate=lane_coordinate,
        ),
        "result_offsets": result_offsets,
        "tile": _bounded_vector_tile(
            raw["parameters"]["block_size"],
            reduction_capacity,
            name="NormStats block_size",
        ),
    }


def _dense_matmul_call(raw) -> dict[str, object]:
    if str(raw.get("semantic_op", "")) == "ntt.matmul_norm_stats":
        return _dense_matmul_norm_stats_call(raw)
    source = _buffer(raw, "inputs", "lhs")
    weight = _buffer_from_formals(raw, "inputs", ("rhs", "weight"))
    result = _buffer(raw, "outputs", "result")
    source_abi = source["abi"]
    weight_abi = weight["abi"]
    result_abi = result["abi"]
    source_domain = _scalar_last_axis_domain(
        source_abi, "dense_local_k_offsets", owner="DenseMatMul lhs"
    )
    result_domain = _scalar_last_axis_domain(
        result_abi, "dense_local_n_offsets", owner="DenseMatMul result"
    )
    global_k = _scalar_logical_axis_extent(source_abi, -1)
    global_n = _scalar_logical_axis_extent(result_abi, -1)
    packed_layout = raw["parameters"].get("packed_layout")
    if packed_layout is not None:
        if str(packed_layout) != "k_major_n8_k16":
            raise CodegenError(
                f"TIR DenseMatMul has unsupported packed layout {packed_layout!r}."
            )
        _verify_k_major_n8_k16_weight(weight_abi, global_n, global_k)
    else:
        weight_shape = _static_shape(weight_abi, "logical_shape")
        if len(weight_shape) != 2:
            raise CodegenError("TIR DenseMatMul logical RHS must be rank two.")
        attrs = raw.get("semantic_attrs", {})
        transpose_b = bool(attrs.get("transpose_b", False))
        expected = (global_n, global_k) if transpose_b else (global_k, global_n)
        if weight_shape != expected:
            raise CodegenError(
                "TIR DenseMatMul RHS shape disagrees with its logical MxK @ KxN "
                f"contract: {weight_shape} != {expected}."
            )
    weight_pointer, weight_offset = _dense_weight_access(
        weight,
        packed_layout=None if packed_layout is None else str(packed_layout),
        transpose_b=bool(raw.get("semantic_attrs", {}).get("transpose_b", False)),
        local_n="dense_local_n_offsets[:, None]",
        local_k="dense_local_k_offsets[None, :]",
        global_n="dense_global_n[:, None]",
        global_k="dense_global_k[None, :]",
    )
    helper_by_variant = {
        "gemv": "_flagmega_dense_matmul_gemv_accumulate",
        "tensor_descriptor_gemv": (
            "_flagmega_dense_matmul_tensor_descriptor_gemv_accumulate"
        ),
        "tensor_descriptor_smem_pipeline_gemv": (
            "_flagmega_dense_matmul_tensor_descriptor_gemv_accumulate"
        ),
        "tensor_descriptor_smem_pipeline_aux_gemv": (
            "_flagmega_dense_matmul_tensor_descriptor_gemv_accumulate"
        ),
        "packed_k_major_gemv": (
            "_flagmega_dense_matmul_packed_k_major_gemv_accumulate"
        ),
        "packed_tensor_descriptor_smem_pipeline_gemv": (
            "_flagmega_dense_matmul_packed_tensor_descriptor_smem_pipeline_accumulate"
        ),
        "split_k_gemv": "_flagmega_dense_matmul_split_k_gemv_accumulate",
        "split_k_packed_k_major_gemv": (
            "_flagmega_dense_matmul_split_k_packed_k_major_gemv_accumulate"
        ),
        "split_k_n_packed_k_major_gemv": (
            "_flagmega_dense_matmul_split_k_n_packed_k_major_gemv_accumulate"
        ),
    }
    variant = str(raw["variant"])
    reduction_tile = raw["parameters"][
        "split_k_block_k" if variant.startswith("split_k") else "block_k"
    ]
    try:
        accumulate_helper = helper_by_variant[variant]
    except KeyError as error:
        raise CodegenError(
            f"TIR DenseMatMul variant {variant!r} has no local-shard template ABI."
        ) from error
    block_k = _bounded_vector_tile(
        reduction_tile,
        source_domain["capacity"],
        name="DenseMatMul reduction tile",
    )
    tile_n = _bounded_vector_tile(
        raw["parameters"]["tile_n"],
        result_domain["capacity"],
        name="DenseMatMul output tile",
    )
    descriptor_requests = ()
    descriptor_variants = {
        "tensor_descriptor_gemv",
        "tensor_descriptor_smem_pipeline_gemv",
        "tensor_descriptor_smem_pipeline_aux_gemv",
        "packed_tensor_descriptor_smem_pipeline_gemv",
    }
    if variant in descriptor_variants:
        if str(weight_abi.get("coordinate_space")) != "canonical_global":
            raise CodegenError(
                "Tensor-descriptor DenseMatMul requires canonical-global RHS storage."
            )
        weight_shape = _static_shape(weight_abi, "logical_shape")
        weight_strides = _static_shape(weight_abi, "scalar_storage_strides")
        transpose_b = bool(raw.get("semantic_attrs", {}).get("transpose_b", False))
        packed_descriptor = (
            variant == "packed_tensor_descriptor_smem_pipeline_gemv"
        )
        if packed_descriptor:
            if packed_layout != "k_major_n8_k16":
                raise CodegenError(
                    "Packed tensor-descriptor DenseMatMul requires "
                    "k_major_n8_k16 RHS storage."
                )
            if tuple(weight_abi.get("scalar_lane_shape", ())) != (8, 2, 8):
                raise CodegenError(
                    "Packed tensor-descriptor DenseMatMul requires typed "
                    "VectorType(N8,KPack2,KVector8) storage."
                )
            expected_strides = (weight_shape[1] * 128, 128)
            if len(weight_shape) != 2 or weight_strides != expected_strides:
                raise CodegenError(
                    "Packed tensor-descriptor DenseMatMul requires a contiguous "
                    "[K/16,N/8] vector backing."
                )
            if tile_n % 8 or block_k % 16:
                raise CodegenError(
                    "Packed tensor-descriptor DenseMatMul tile does not preserve "
                    "its N8/K16 atoms."
                )
            # Preserve the vector backing byte order while exposing a
            # 128-byte contiguous TMA axis, matching [N, KPack, KVector]
            # flattened as [KPack, N*KVector].
            descriptor_shape = (*weight_shape, 2, 64)
            descriptor_strides = (*weight_strides, 64, 1)
            descriptor_block_shape = (block_k // 16, tile_n // 8, 2, 64)
        else:
            if packed_layout is not None:
                raise CodegenError(
                    "Logical tensor-descriptor DenseMatMul requires an unpacked RHS."
                )
            if int(weight_abi.get("scalar_lane_count", 1)) != 1:
                raise CodegenError(
                    "Logical tensor-descriptor DenseMatMul requires a scalar RHS dtype."
                )
            if weight_strides[-1] != 1:
                raise CodegenError(
                    "Logical tensor-descriptor DenseMatMul requires a contiguous RHS inner axis."
                )
            if (
                variant in {
                    "tensor_descriptor_smem_pipeline_gemv",
                    "tensor_descriptor_smem_pipeline_aux_gemv",
                }
                and not transpose_b
            ):
                raise CodegenError(
                    "Tensor-descriptor Shared pipeline DenseMatMul requires "
                    "transpose_b=True so each TMA tile is contiguous N x K."
                )
            descriptor_shape = weight_shape
            descriptor_strides = weight_strides
            descriptor_block_shape = (
                (tile_n, block_k) if transpose_b else (block_k, tile_n)
            )
        descriptor_offset_bytes = (
            int(weight_abi["pool_byte_offset"])
            if str(weight_abi.get("storage")) in {"rdata", "workspace"}
            else 0
        )
        descriptor_kind = str(
            raw["parameters"].get("descriptor_kind", "single")
        )
        if descriptor_kind == "table":
            if not packed_descriptor:
                raise CodegenError(
                    "DenseMatMul tensor-map tables currently require packed RHS storage."
                )
            descriptor_requests = (
                packed_distributed_tensor_map_table_request(
                    weight_abi,
                    parameter="weight_descriptor",
                    source=str(weight["runtime_argument"]),
                    offset_bytes=descriptor_offset_bytes,
                    descriptor_shape=descriptor_shape,
                    descriptor_strides=descriptor_strides,
                    block_shape=descriptor_block_shape,
                ),
            )
        elif descriptor_kind == "single":
            descriptor_requests = ({
                "parameter": "weight_descriptor",
                "source": str(weight["runtime_argument"]),
                "kind": "single",
                "offset_bytes": descriptor_offset_bytes,
                "dtype": str(weight_abi["scalar_dtype"]),
                "shape": descriptor_shape,
                "strides": descriptor_strides,
                "block_shape": descriptor_block_shape,
                "source_shape_axes": tuple(() for _ in descriptor_shape),
                "padding": "zero",
            },)
        else:
            raise CodegenError(
                f"DenseMatMul has unsupported descriptor kind {descriptor_kind!r}."
            )
    result = {
        "source": _pointer(source),
        "weight": weight_pointer,
        "result": _pointer(result),
        "local_k_capacity": source_domain["capacity"],
        "local_n_capacity": result_domain["capacity"],
        "source_active": source_domain["active"],
        "source_owner_active": source_domain["owner_active"],
        "source_active_extent": source_domain["physical_active_extent"],
        "source_lane_count": source_domain["lane_count"],
        "source_owner_offset": source_domain["owner_base_offset"],
        "source_local_offset": source_domain["local_offset"],
        "result_active": result_domain["active"],
        "source_offset": source_domain["offset"],
        "result_offset": result_domain["offset"],
        "global_k": source_domain["global_scalar"],
        "global_n": result_domain["global_scalar"],
        "global_k_size": global_k,
        "global_n_size": global_n,
        "weight_offset": weight_offset,
        "packed_layout": None if packed_layout is None else str(packed_layout),
        "accumulate_helper": accumulate_helper,
        "block_k": block_k,
        "tile_n": tile_n,
        "output_type": _triton_dtype(str(result_abi["scalar_dtype"])),
        "host_tensor_descriptor_requests": descriptor_requests,
    }
    if descriptor_requests:
        descriptor_n_offset = _scalar_last_axis_domain(
            result_abi,
            "dense_local_n_start",
            owner="DenseMatMul descriptor result",
        )["global_scalar"]
        descriptor_k_offset = _scalar_last_axis_domain(
            source_abi,
            "dense_local_k_start",
            owner="DenseMatMul descriptor lhs",
        )["global_scalar"]
        result.update({
            "weight_descriptor": "weight_descriptor",
            "descriptor_kind": descriptor_kind,
            "descriptor_n_offset": (
                "tl.full((), (dense_local_n_start), tl.int32)"
                if descriptor_kind == "table"
                else f"tl.full((), ({descriptor_n_offset}), tl.int32)"
            ),
            "descriptor_k_offset": (
                "tl.full((), (dense_local_k_start), tl.int32)"
                if descriptor_kind == "table"
                else f"tl.full((), ({descriptor_k_offset}), tl.int32)"
            ),
            "transpose_b": bool(
                raw.get("semantic_attrs", {}).get("transpose_b", False)
            ),
        })
        if packed_descriptor:
            packed_k_offset = (
                "dense_local_k_start"
                if descriptor_kind == "table"
                else descriptor_k_offset
            )
            packed_n_offset = (
                "dense_local_n_start"
                if descriptor_kind == "table"
                else descriptor_n_offset
            )
            result.update({
                "descriptor_offsets": (
                    "tl.full((), "
                    f"(({packed_k_offset}) // 16), tl.int32)",
                    "tl.full((), "
                    f"(({packed_n_offset}) // 8), tl.int32)",
                    "tl.full((), 0, tl.int32)",
                    "tl.full((), 0, tl.int32)",
                ),
                "descriptor_block_shape": descriptor_block_shape,
            })
    if variant in {
        "tensor_descriptor_smem_pipeline_gemv",
        "tensor_descriptor_smem_pipeline_aux_gemv",
        "packed_tensor_descriptor_smem_pipeline_gemv",
    }:
        pipeline = raw.get("transfer_pipeline")
        workspaces = raw.get("shared_workspaces")
        if not isinstance(pipeline, Mapping) or not isinstance(
            workspaces, (tuple, list)
        ):
            raise CodegenError(
                "Shared-pipeline DenseMatMul requires typed transfer-pipeline "
                "and Shared workspace ABI metadata."
            )
        result["pipeline_contract"] = dict(pipeline)
        result["shared_workspaces"] = tuple(workspaces)
        result["num_stages"] = int(raw["parameters"]["num_stages"])
        result["producer_warps"] = int(raw["parameters"]["producer_warps"])
        result["producer_registers"] = int(
            raw["parameters"]["producer_registers"]
        )
        if variant == "packed_tensor_descriptor_smem_pipeline_gemv":
            parameters = raw["parameters"]
            reduction_group = int(parameters["reduction_group"])
            consumer_warps = int(parameters["consumer_warps"])
            worker_width = int(parameters["worker_width"])
            if (
                reduction_group <= 0
                or reduction_group & (reduction_group - 1)
                or block_k % reduction_group
                or consumer_warps <= 0
                or tile_n % consumer_warps
            ):
                raise CodegenError(
                    "Packed descriptor DenseMatMul has an invalid reduction/"
                    "consumer partition."
                )
            threads_per_warp_n = tile_n // consumer_warps
            if (
                threads_per_warp_n <= 0
                or worker_width % threads_per_warp_n
            ):
                raise CodegenError(
                    "Packed descriptor DenseMatMul cannot partition a warp "
                    "across its output tile."
                )
            threads_per_warp_k = worker_width // threads_per_warp_n
            if reduction_group % threads_per_warp_k:
                raise CodegenError(
                    "Packed descriptor DenseMatMul reduction group cannot be "
                    "partitioned exactly across a warp."
                )
            reduction_groups = block_k // reduction_group
            reduction_unroll = parameters.get("reduction_unroll", reduction_groups)
            if (not isinstance(reduction_unroll, int) or isinstance(reduction_unroll, bool)
                    or reduction_unroll <= 0 or reduction_groups % reduction_unroll):
                raise CodegenError("Packed descriptor DenseMatMul reduction unroll must divide its stage group count.")
            result.update({
                "reduction_group": reduction_group,
                "reduction_groups_per_stage": reduction_groups,
                "reduction_unroll": reduction_unroll,
                "consumer_size_per_thread": (
                    1,
                    reduction_group // threads_per_warp_k,
                ),
                "consumer_threads_per_warp": (
                    threads_per_warp_n,
                    threads_per_warp_k,
                ),
                "consumer_warps_per_cta": (consumer_warps, 1),
                "consumer_warps": consumer_warps,
                "worker_width": worker_width,
            })
    return result


def _dense_matmul_norm_stats_call(raw) -> dict[str, object]:
    """Encode the packed projection epilogue as an explicit two-result ABI."""

    variant = str(raw.get("variant", ""))
    base_variants = {
        "packed_k_major_gemv_norm_stats": "packed_k_major_gemv",
        "packed_tensor_descriptor_smem_pipeline_gemv_norm_stats": (
            "packed_tensor_descriptor_smem_pipeline_gemv"
        ),
    }
    try:
        base_variant = base_variants[variant]
    except KeyError as error:
        raise CodegenError(
            f"TIR MatMulNormStats has no dense_matmul renderer for {variant!r}."
        ) from error
    attrs = raw.get("semantic_attrs", {})
    if (
        not isinstance(attrs, Mapping)
        or attrs.get("rhs_layout") != "k_major"
        or bool(attrs.get("transpose_a", False))
        or bool(attrs.get("transpose_b", False))
        or bool(attrs.get("use_mean", False))
    ):
        raise CodegenError(
            "Packed MatMulNormStats requires K-major RHS, fixed transpose "
            "semantics, and RMS statistics."
        )
    result_parameter = _parameter(raw, "outputs", "result_0")
    base_raw = {
        **dict(raw),
        "semantic_op": "ntt.packed_matmul",
        "variant": base_variant,
        "outputs": ({**dict(result_parameter), "formal": "result"},),
        "parameters": {
            **dict(raw["parameters"]),
            "variant": base_variant,
        },
    }
    result = _dense_matmul_call(base_raw)
    lhs_stage_extent = raw["parameters"].get("lhs_stage_extent")
    lhs_copy_kind = raw["parameters"].get("lhs_copy_kind", "sync")
    if lhs_copy_kind not in ("sync", "async"):
        raise CodegenError("LHS copy kind must be explicitly 'sync' or 'async'.")
    if lhs_copy_kind == "async" and lhs_stage_extent is None:
        raise CodegenError("LHS async copy requires a typed staging workspace.")
    if lhs_stage_extent is not None:
        copy_tile = raw["parameters"].get("lhs_copy_tile")
        if (
            variant != "packed_tensor_descriptor_smem_pipeline_gemv_norm_stats"
            or isinstance(lhs_stage_extent, bool)
            or not isinstance(lhs_stage_extent, int)
            or lhs_stage_extent < result["local_k_capacity"]
            or lhs_stage_extent <= 0
            or lhs_stage_extent & (lhs_stage_extent - 1)
            or isinstance(copy_tile, bool)
            or not isinstance(copy_tile, int)
            or copy_tile <= 0
            or copy_tile & (copy_tile - 1)
            or lhs_stage_extent % copy_tile
        ):
            raise CodegenError(
                "Packed MatMulNormStats LHS staging requires a power-of-two "
                "capacity covering the local input and a dividing copy tile."
            )
        workspaces = result["shared_workspaces"]
        pipeline = result["pipeline_contract"]
        if (
            len(workspaces) != 2
            or tuple(pipeline.get("consumer_shared_workspace_indices", ())) != (1,)
            or workspaces[1].get("name") != "lhs_stage"
            or tuple(workspaces[1].get("shape", ())) != (1, lhs_stage_extent)
            or workspaces[1].get("dtype") != "bfloat16"
            or workspaces[1].get("alignment_bytes", 0) < 128
            or not workspaces[1].get("matrix_compatible", False)
        ):
            raise CodegenError(
                "Packed MatMulNormStats LHS staging requires one typed, "
                "128-byte-aligned consumer workspace matching its capacity."
            )
        copy_denominator = result["consumer_warps"] * result["worker_width"] * 2
        if copy_tile % copy_denominator:
            raise CodegenError(
                "Packed MatMulNormStats LHS copy cannot partition its consumer warps."
            )
        result.update({
            "lhs_copy_kind": lhs_copy_kind,
            "lhs_stage_extent": lhs_stage_extent,
            "lhs_copy_tile": copy_tile,
            "input_copy_size_per_thread": copy_tile // copy_denominator,
        })
        if lhs_copy_kind == "async":
            from triton.flagmega.codegen.triton.row_transfer import validate_async_row_copy

            validate_async_row_copy(_buffer(raw, "inputs", "lhs")["abi"], copy_tile)
    residual = _buffer(raw, "inputs", "addend")
    value = _buffer(raw, "outputs", "result_0")
    stats = _buffer(raw, "outputs", "result_1")
    residual_abi = residual["abi"]
    value_abi = value["abi"]
    stats_abi = stats["abi"]
    if (
        _static_shape(residual_abi, "logical_shape")
        != _static_shape(value_abi, "logical_shape")
        or tuple(residual_abi.get("scalar_lane_shape", ()))
        != tuple(value_abi.get("scalar_lane_shape", ()))
        or str(residual_abi.get("scalar_dtype")) != "bfloat16"
        or str(value_abi.get("scalar_dtype")) != "bfloat16"
    ):
        raise CodegenError(
            "Packed MatMulNormStats residual and value must have one identical "
            "BF16 vector ABI."
        )
    value_shape = _static_shape(value_abi, "logical_shape")
    value_lane_count = int(value_abi.get("scalar_lane_count", 1))
    value_scalar_strides = _static_shape(
        value_abi, "scalar_storage_strides"
    )
    if (
        prod(value_shape[:-1], start=1) != 1
        or not value_scalar_strides
        or value_scalar_strides[-1] != value_lane_count
    ):
        raise CodegenError(
            "Packed MatMulNormStats statistics reduction requires one row in "
            "a contiguous result backing."
        )
    if (
        int(stats_abi.get("scalar_lane_count", 1)) != 1
        or prod(_static_shape(stats_abi, "local_capacity_shape"), start=1) != 1
        or str(stats_abi.get("scalar_dtype")) != "float32"
    ):
        raise CodegenError(
            "Packed MatMulNormStats requires one scalar F32 statistics result."
        )
    residual_domain = _scalar_last_axis_domain(
        residual_abi,
        "dense_local_n_offsets",
        owner="MatMulNormStats residual",
    )
    stats_shape = _static_shape(stats_abi, "local_capacity_shape")
    result.update({
        "residual": _pointer(residual),
        "residual_offset": residual_domain["offset"],
        "stats": _pointer(stats),
        "stats_offset": emit_local_scalar_offset(
            stats_abi, ("0",) * len(stats_shape)
        ),
        "result_writer_active": _canonical_writer_active(value_abi),
        "stats_writer_active": _canonical_writer_active(stats_abi),
    })
    if variant == "packed_k_major_gemv_norm_stats":
        # This fused variant owns a variant-local helper in its template.
        # Do not retain the helper symbol inherited from the unfused base
        # encoder: the package renderer is free to omit that base template.
        result["accumulate_helper"] = (
            "_flagmega_dense_matmul_packed_k_major_gemv_norm_stats_accumulate"
        )
        return result

    if str(value_abi.get("coordinate_space")) not in {
        "canonical_global",
        "local",
        "parent_shard_local",
    }:
        raise CodegenError(
            "Packed descriptor MatMulNormStats has an unsupported result "
            "coordinate space."
        )
    stats_distributed = stats_abi.get("distributed_type")
    stats_partial = (
        stats_distributed.get("partial")
        if isinstance(stats_distributed, Mapping)
        else None
    )
    placement = (
        stats_distributed.get("placement")
        if isinstance(stats_distributed, Mapping)
        else None
    )
    hierarchy = (
        tuple(int(value) for value in placement.get("hierarchy", ()))
        if isinstance(placement, Mapping)
        else ()
    )
    if (
        str(stats_abi.get("storage_kind")) != "compact_per_owner"
        or str(stats_abi.get("coordinate_space")) != "local"
        or not isinstance(stats_partial, Mapping)
        or str(stats_partial.get("reduce_op")) != "sum"
        or tuple(int(value) for value in stats_partial.get("axes", ()))
        != tuple(range(len(hierarchy)))
    ):
        raise CodegenError(
            "Packed MatMulNormStats requires one compact Sum-partial "
            "statistics component per placement owner."
        )
    owner_count = int(raw["parameters"].get("owner_count", 0))
    tile_n = int(result["tile_n"])
    global_n = int(result["global_n_size"])
    if (
        owner_count <= 0
        or global_n % (owner_count * tile_n)
        or owner_count != int(value_abi.get("owner_count", 0))
    ):
        raise CodegenError(
            "Packed MatMulNormStats output cannot be partitioned into equal "
            "full N tiles across its owners."
        )
    result.update({
        "owner_count": owner_count,
        "tiles_per_owner": global_n // (owner_count * tile_n),
    })
    return result


def _dense_matmul_glu_call(raw) -> dict[str, object]:
    source = _buffer(raw, "inputs", "value")
    gate_weight = _buffer(raw, "inputs", "gate_weight")
    up_weight = _buffer(raw, "inputs", "up_weight")
    result = _buffer(raw, "outputs", "result")
    source_abi = source["abi"]
    result_abi = result["abi"]
    source_domain = _scalar_last_axis_domain(
        source_abi, "dense_glu_local_k_offsets", owner="DenseMatMulGlu input"
    )
    result_domain = _scalar_last_axis_domain(
        result_abi, "dense_glu_local_n_offsets", owner="DenseMatMulGlu result"
    )
    global_k = _scalar_logical_axis_extent(source_abi, -1)
    global_n = _scalar_logical_axis_extent(result_abi, -1)
    packed_layout = raw["parameters"].get("packed_layout")
    if packed_layout is not None:
        if str(packed_layout) != "k_major_n8_k16":
            raise CodegenError(
                f"TIR DenseMatMulGlu has unsupported packed layout {packed_layout!r}."
            )
        for binding in (gate_weight, up_weight):
            _verify_k_major_n8_k16_weight(binding["abi"], global_n, global_k)
    else:
        expected = (global_n, global_k)
        for binding in (gate_weight, up_weight):
            if _static_shape(binding["abi"], "logical_shape") != expected:
                raise CodegenError(
                    "TIR DenseMatMulGlu logical weights must be [N, K]."
                )
    gate_weight_pointer, gate_weight_offset = _dense_weight_access(
        gate_weight,
        packed_layout=None if packed_layout is None else str(packed_layout),
        transpose_b=True,
        local_n="dense_glu_local_n_offsets[:, None]",
        local_k="dense_glu_local_k_offsets[None, :]",
        global_n="dense_glu_global_n[:, None]",
        global_k="dense_glu_global_k[None, :]",
    )
    up_weight_pointer, up_weight_offset = _dense_weight_access(
        up_weight,
        packed_layout=None if packed_layout is None else str(packed_layout),
        transpose_b=True,
        local_n="dense_glu_local_n_offsets[:, None]",
        local_k="dense_glu_local_k_offsets[None, :]",
        global_n="dense_glu_global_n[:, None]",
        global_k="dense_glu_global_k[None, :]",
    )
    helper_by_variant = {
        "gemv": "_flagmega_dense_matmul_glu_gemv_accumulate",
        "packed_k_major_gemv": (
            "_flagmega_dense_matmul_glu_packed_k_major_gemv_accumulate"
        ),
        "packed_tensor_descriptor_smem_pipeline_gemv": (
            "_flagmega_dense_matmul_glu_packed_tensor_descriptor_"
            "smem_pipeline_accumulate"
        ),
    }
    variant = str(raw["variant"])
    try:
        accumulate_helper = helper_by_variant[variant]
    except KeyError as error:
        raise CodegenError(
            f"TIR DenseMatMulGlu variant {variant!r} has no local-shard template ABI."
        ) from error
    block_k = _bounded_vector_tile(
        raw["parameters"]["block_k"],
        source_domain["capacity"],
        name="DenseMatMulGlu reduction tile",
    )
    tile_n = _bounded_vector_tile(
        raw["parameters"]["tile_n"],
        result_domain["capacity"],
        name="DenseMatMulGlu output tile",
    )
    if variant == "packed_tensor_descriptor_smem_pipeline_gemv":
        # A TMA tile is a selected physical Shared/descriptor ABI, not just a
        # register-vector bound. Tail owners mask outputs; they must not shrink
        # the tile and silently change the declared Shared allocation shape.
        tile_n = int(raw["parameters"]["tile_n"])
    result = {
        "source": _pointer(source),
        "gate_weight": gate_weight_pointer,
        "up_weight": up_weight_pointer,
        "result": _pointer(result),
        "local_k_capacity": source_domain["capacity"],
        "local_n_capacity": result_domain["capacity"],
        "source_active": source_domain["active"],
        "source_owner_active": source_domain["owner_active"],
        "source_active_extent": source_domain["physical_active_extent"],
        "source_lane_count": source_domain["lane_count"],
        "result_active": result_domain["active"],
        "source_offset": source_domain["offset"],
        "result_offset": result_domain["offset"],
        "global_k": source_domain["global_scalar"],
        "global_n": result_domain["global_scalar"],
        "global_n_size": global_n,
        "gate_weight_offset": gate_weight_offset,
        "up_weight_offset": up_weight_offset,
        "packed_layout": None if packed_layout is None else str(packed_layout),
        "accumulate_helper": accumulate_helper,
        "block_k": block_k,
        "tile_n": tile_n,
        "output_type": _triton_dtype(str(result_abi["scalar_dtype"])),
    }
    if variant == "packed_tensor_descriptor_smem_pipeline_gemv":
        if packed_layout != "k_major_n8_k16":
            raise CodegenError(
                "Packed descriptor DenseMatMulGlu requires k_major_n8_k16 weights."
            )
        descriptor_shape = None
        descriptor_strides = None
        descriptor_requests = []
        descriptor_block_shape = (block_k // 16, tile_n // 8, 2, 64)
        descriptor_kind = str(
            raw["parameters"].get("descriptor_kind", "single")
        )
        if descriptor_kind not in {"single", "table"}:
            raise CodegenError(
                "Packed descriptor DenseMatMulGlu has unsupported descriptor "
                f"kind {descriptor_kind!r}."
            )
        for parameter, binding in (
            ("gate_weight_descriptor", gate_weight),
            ("up_weight_descriptor", up_weight),
        ):
            abi = binding["abi"]
            if str(abi.get("coordinate_space")) != "canonical_global":
                raise CodegenError(
                    "Packed descriptor DenseMatMulGlu requires canonical-global "
                    "weight storage."
                )
            weight_shape = _static_shape(abi, "logical_shape")
            weight_strides = _static_shape(abi, "scalar_storage_strides")
            lane_shape = tuple(abi.get("scalar_lane_shape", ()))
            if lane_shape == (8, 2, 8) and len(weight_shape) == 2:
                expected_strides = (weight_shape[1] * 128, 128)
                current_shape = (*weight_shape, 2, 64)
                current_strides = (*weight_strides, 64, 1)
            elif (
                not lane_shape
                and len(weight_shape) == 4
                and weight_shape[-2:] == (2, 64)
            ):
                expected_strides = (weight_shape[1] * 128, 128, 64, 1)
                current_shape = weight_shape
                current_strides = weight_strides
            else:
                raise CodegenError(
                    "Packed descriptor DenseMatMulGlu requires either typed "
                    "VectorType(N8,KPack2,KVector8) weights or canonical "
                    "physical [K/16,N/8,2,64] weights."
                )
            if weight_strides != expected_strides:
                raise CodegenError(
                    "Packed descriptor DenseMatMulGlu requires contiguous "
                    "[K/16,N/8,2,64] scalar backing."
                )
            if descriptor_shape is not None and (
                current_shape != descriptor_shape
                or current_strides != descriptor_strides
            ):
                raise CodegenError(
                    "Packed descriptor DenseMatMulGlu gate/up weights must have "
                    "identical physical geometry."
                )
            descriptor_shape = current_shape
            descriptor_strides = current_strides
            offset_bytes = (
                int(abi["pool_byte_offset"])
                if str(abi.get("storage")) in {"rdata", "workspace"}
                else 0
            )
            if descriptor_kind == "table":
                descriptor_requests.append(
                    packed_distributed_tensor_map_table_request(
                        abi,
                        parameter=parameter,
                        source=str(binding["runtime_argument"]),
                        offset_bytes=offset_bytes,
                        descriptor_shape=current_shape,
                        descriptor_strides=current_strides,
                        block_shape=descriptor_block_shape,
                    )
                )
            else:
                descriptor_requests.append({
                    "parameter": parameter,
                    "source": str(binding["runtime_argument"]),
                    "kind": "single",
                    "offset_bytes": offset_bytes,
                    "dtype": str(abi["scalar_dtype"]),
                    "shape": current_shape,
                    "strides": current_strides,
                    "block_shape": descriptor_block_shape,
                    "source_shape_axes": tuple(() for _ in current_shape),
                    "padding": "zero",
                })
        if tile_n % 8 or block_k % 16:
            raise CodegenError(
                "Packed descriptor DenseMatMulGlu tile does not preserve N8/K16 atoms."
            )
        if descriptor_kind == "table":
            descriptor_n_offset = "dense_glu_local_n_start"
            descriptor_k_offset = "dense_glu_local_k_start"
        else:
            descriptor_n_offset = _scalar_last_axis_domain(
                result_abi,
                "dense_glu_local_n_start",
                owner="DenseMatMulGlu descriptor result",
            )["global_scalar"]
            descriptor_k_offset = _scalar_last_axis_domain(
                source_abi,
                "dense_glu_local_k_start",
                owner="DenseMatMulGlu descriptor input",
            )["global_scalar"]
        pipeline = raw.get("transfer_pipeline")
        workspaces = raw.get("shared_workspaces")
        if not isinstance(pipeline, Mapping) or not isinstance(
            workspaces, (tuple, list)
        ):
            raise CodegenError(
                "Packed descriptor DenseMatMulGlu requires typed transfer-pipeline "
                "and Shared workspace ABI metadata."
            )
        parameters = raw["parameters"]
        reduction_group = int(parameters["reduction_group"])
        consumer_warps = int(parameters["consumer_warps"])
        worker_width = int(parameters["worker_width"])
        if (
            reduction_group <= 0
            or reduction_group & (reduction_group - 1)
            or block_k % reduction_group
            or consumer_warps <= 0
            or tile_n % consumer_warps
        ):
            raise CodegenError(
                "Packed descriptor DenseMatMulGlu has an invalid reduction/"
                "consumer partition."
            )
        threads_per_warp_n = tile_n // consumer_warps
        if threads_per_warp_n <= 0 or worker_width % threads_per_warp_n:
            raise CodegenError(
                "Packed descriptor DenseMatMulGlu cannot partition its output tile."
            )
        threads_per_warp_k = worker_width // threads_per_warp_n
        if reduction_group % threads_per_warp_k:
            raise CodegenError(
                "Packed descriptor DenseMatMulGlu cannot partition its reduction."
            )
        reduction_groups = block_k // reduction_group
        reduction_unroll = parameters.get("reduction_unroll", reduction_groups)
        if (
            not isinstance(reduction_unroll, int)
            or isinstance(reduction_unroll, bool)
            or reduction_unroll <= 0
            or reduction_groups % reduction_unroll
        ):
            raise CodegenError(
                "Packed descriptor DenseMatMulGlu reduction unroll must be a "
                "positive integer divisor of its reduction groups."
            )
        result.update({
            "host_tensor_descriptor_requests": tuple(descriptor_requests),
            "gate_weight_descriptor": "gate_weight_descriptor",
            "up_weight_descriptor": "up_weight_descriptor",
            "descriptor_offsets": (
                "tl.full((), "
                f"(({descriptor_k_offset}) // 16), tl.int32)",
                "tl.full((), "
                f"(({descriptor_n_offset}) // 8), tl.int32)",
                "tl.full((), 0, tl.int32)",
                "tl.full((), 0, tl.int32)",
            ),
            "descriptor_block_shape": descriptor_block_shape,
            "descriptor_kind": descriptor_kind,
            "pipeline_contract": dict(pipeline),
            "shared_workspaces": tuple(workspaces),
            "num_stages": int(parameters["num_stages"]),
            "producer_warps": int(parameters["producer_warps"]),
            "producer_registers": int(parameters["producer_registers"]),
            "reduction_group": reduction_group,
            "reduction_groups_per_stage": reduction_groups,
            "reduction_unroll": reduction_unroll,
            "consumer_size_per_thread": (
                1,
                reduction_group // threads_per_warp_k,
            ),
            "consumer_threads_per_warp": (
                threads_per_warp_n,
                threads_per_warp_k,
            ),
            "consumer_warps_per_cta": (consumer_warps, 1),
            "consumer_warps": consumer_warps,
            "worker_width": worker_width,
            "paired_weight_fields": bool(
                parameters.get("paired_weight_fields", False)
            ),
        })
        lhs_stage_extent = parameters.get("lhs_stage_extent")
        if result["paired_weight_fields"]:
            capacity = pipeline.get("capacity")
            if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
                raise CodegenError("Packed descriptor DenseMatMulGlu requires a positive integer pipe capacity.")
            expected_stage_shape = (capacity, *descriptor_block_shape)
            channels = pipeline.get("channels", ())
            consumer_indices = tuple(
                pipeline.get("consumer_shared_workspace_indices", ())
            )
            expected_consumer_indices = (
                (2,) if lhs_stage_extent is not None else ()
            )
            expected_workspace_count = 2 + int(lhs_stage_extent is not None)
            if (
                len(workspaces) != expected_workspace_count
                or len(channels) != 1
                or tuple(channels[0].get("shared_workspace_indices", ()))
                != (0, 1)
                or consumer_indices != expected_consumer_indices
            ):
                raise CodegenError(
                    "Packed descriptor DenseMatMulGlu paired transfer requires "
                    "one two-field channel with the selected capacity and any complete "
                    "LHS workspace owned by the consumer."
                )
            if tuple(
                (value.get("name"), tuple(value.get("shape", ())))
                for value in workspaces[:2]
            ) != (
                ("gate_stage", expected_stage_shape),
                ("up_stage", expected_stage_shape),
            ):
                raise CodegenError(
                    "Packed descriptor DenseMatMulGlu paired transfer Shared "
                    "workspaces do not match its two descriptor tiles."
                )
            result["pipeline_channel_field_names"] = {
                "weight": ("gate", "up")
            }
        if lhs_stage_extent is not None:
            if (
                isinstance(lhs_stage_extent, bool)
                or not isinstance(lhs_stage_extent, int)
                or lhs_stage_extent != source_domain["capacity"]
                or lhs_stage_extent & (lhs_stage_extent - 1)
            ):
                raise CodegenError(
                    "Packed descriptor DenseMatMulGlu complete LHS staging "
                    "requires a power-of-two extent equal to the local input "
                    "capacity."
                )
            consumer_indices = tuple(
                pipeline.get("consumer_shared_workspace_indices", ())
            )
            lhs_workspace_index = (
                2 if result["paired_weight_fields"] else 1
            )
            if (
                consumer_indices != (lhs_workspace_index,)
                or len(workspaces) != lhs_workspace_index + 1
            ):
                raise CodegenError(
                    "Packed descriptor DenseMatMulGlu complete LHS staging "
                    "requires one consumer-owned workspace after the weight stage."
                )
            lhs_workspace = workspaces[lhs_workspace_index]
            if (
                not isinstance(lhs_workspace, Mapping)
                or lhs_workspace.get("name") != "lhs_stage"
                or tuple(lhs_workspace.get("shape", ()))
                != (1, lhs_stage_extent)
            ):
                raise CodegenError(
                    "Packed descriptor DenseMatMulGlu LHS workspace does not "
                    "match its selected staging extent."
                )
            copy_denominator = consumer_warps * worker_width * 2
            if lhs_stage_extent % copy_denominator:
                raise CodegenError(
                    "Packed descriptor DenseMatMulGlu LHS copy cannot be evenly "
                    "partitioned across its consumer warps."
                )
            result.update({
                "lhs_stage_extent": lhs_stage_extent,
                "input_copy_size_per_thread": (
                    lhs_stage_extent // copy_denominator
                ),
            })
    return result


def _qkv_parallel_linear_call(raw) -> dict[str, object]:
    source = _buffer(raw, "inputs", "input")
    weight = _buffer(raw, "inputs", "q_weight_qkv_fused")
    outputs = tuple(
        _buffer(raw, "outputs", f"result_{index}") for index in range(3)
    )
    source_abi = source["abi"]
    weight_abi = weight["abi"]
    source_domain = _scalar_last_axis_domain(
        source_abi, "qkv_local_k_offsets", owner="PackedQKV input"
    )
    output_domains = tuple(
        _scalar_last_axis_domain(
            output["abi"], "qkv_local_n_offsets", owner=f"PackedQKV output {index}"
        )
        for index, output in enumerate(outputs)
    )
    lane_shape = tuple(int(value) for value in weight_abi["scalar_lane_shape"])
    if len(lane_shape) != 3:
        raise CodegenError(
            "Canonical fused PackedQKV weight requires VectorType(N,KPack,KVector)."
        )
    n_lane, k_pack, k_vector = lane_shape
    k_lane = k_pack * k_vector
    weight_shape = _static_shape(weight_abi, "logical_shape")
    weight_strides = tuple(int(value) for value in weight_abi["scalar_storage_strides"])
    if len(weight_shape) != 3 or len(weight_strides) != 3:
        raise CodegenError("Canonical fused PackedQKV weight must be rank three.")
    if source_domain["capacity"] != weight_shape[1] * k_lane:
        raise CodegenError(
            "Canonical fused PackedQKV owner-local K capacity disagrees with its input."
        )
    group_capacities = tuple(
        domain["capacity"] // n_lane for domain in output_domains
    )
    if (
        any(domain["capacity"] % n_lane for domain in output_domains)
        or sum(group_capacities) != weight_shape[2]
    ):
        raise CodegenError(
            "Canonical fused PackedQKV output capacities disagree with its fused RHS."
        )
    projection_capacities = tuple(
        int(value)
        for value in raw.get("semantic_attrs", {}).get(
            "projection_n_capacities", ()
        )
    )
    if projection_capacities and projection_capacities != tuple(
        _scalar_logical_axis_extent(output["abi"], -1) for output in outputs
    ):
        raise CodegenError(
            "Canonical fused PackedQKV projection capacities disagree with its outputs."
        )
    group_start = 0
    output_calls = []
    for output, domain, group_capacity in zip(
        outputs, output_domains, group_capacities, strict=True
    ):
        output_calls.append({
            "result": _pointer(output),
            "local_n_capacity": domain["capacity"],
            "active": domain["active"],
            "result_offset": domain["offset"],
            "weight_group_start": group_start,
            "projection_start": group_start * n_lane,
        })
        group_start += group_capacity
    result = {
        "source": _pointer(source),
        "source_offset": source_domain["offset"],
        "source_active": source_domain["active"],
        "local_k_capacity": source_domain["capacity"],
        "weight": emit_storage_pointer(
            weight_abi, str(weight["runtime_argument"])
        ),
        "weight_owner_stride": weight_strides[0],
        "weight_k_group_stride": weight_strides[1],
        "weight_n_group_stride": weight_strides[2],
        "n_lane": n_lane,
        "k_lane": k_lane,
        "k_vector": k_vector,
        "outputs": tuple(output_calls),
        "output_type": _triton_dtype(str(outputs[0]["abi"]["scalar_dtype"])),
    }
    variant = str(raw.get("variant", ""))
    if variant == "packed_fused_gemv":
        result.update({
            "block_k": _bounded_vector_tile(
                raw["parameters"]["block_k"],
                source_domain["capacity"],
                name="PackedQKV reduction tile",
            ),
            "tile_n": _bounded_vector_tile(
                raw["parameters"]["tile_n"],
                max(domain["capacity"] for domain in output_domains),
                name="PackedQKV output tile",
            ),
        })
        return result
    if variant not in {"packed_mma_smem_pipeline", "packed_gemv_smem_pipeline"}:
        raise CodegenError(
            f"PackedQKV variant {variant!r} has no local-shard template ABI."
        )

    parameters = raw.get("parameters", {})
    pipeline = raw.get("transfer_pipeline")
    workspaces = raw.get("shared_workspaces")
    if (
        not isinstance(parameters, Mapping)
        or not isinstance(pipeline, Mapping)
        or not isinstance(workspaces, (tuple, list))
        or len(workspaces) != 2
    ):
        raise CodegenError(
            "PackedQKV MMA requires typed transfer-pipeline and two Shared "
            "workspace descriptors."
        )
    block_n = int(parameters["block_n"])
    block_k = int(parameters["block_k"])
    num_stages = int(parameters["num_stages"])
    consumer_warps = int(parameters["consumer_warps"])
    worker_width = int(parameters["worker_width"])
    local_n = sum(domain["capacity"] for domain in output_domains)
    if (
        block_n != local_n
        or source_domain["capacity"] % block_k
        or block_n % 8
        or block_k % 16
        or num_stages <= 0
        or consumer_warps != 8
        or worker_width != 32
    ):
        raise CodegenError(
            "PackedQKV MMA requires a full owner-local N tile, complete K16 "
            "tiles, eight consumer warps, and 32 threads per warp."
        )
    if lane_shape != (8, 2, 8):
        raise CodegenError(
            "PackedQKV MMA requires VectorType(N8,KPack2,KVector8) weights."
        )
    expected_weight_shape = (
        int(outputs[0]["abi"].get("owner_count", 0)),
        source_domain["capacity"] // 16,
        block_n // 8,
    )
    expected_weight_strides = (
        expected_weight_shape[1] * expected_weight_shape[2] * 128,
        expected_weight_shape[2] * 128,
        128,
    )
    if (
        weight_shape != expected_weight_shape
        or weight_strides != expected_weight_strides
        or str(weight_abi.get("storage_kind")) != "compact_local"
    ):
        raise CodegenError(
            "PackedQKV MMA requires one contiguous owner-major fused RHS "
            "[owner,K/16,N/8]<8,2,8>."
        )
    if any(
        int(output["abi"].get("owner_count", 0)) != weight_shape[0]
        or str(output["abi"].get("scalar_dtype")) != "bfloat16"
        or tuple(output["abi"].get("scalar_lane_shape", ())) != (8,)
        for output in outputs
    ):
        raise CodegenError(
            "PackedQKV MMA output owner counts or BF16 vector lanes differ."
        )
    # The immutable packed backing is [owner,K/16,N/8]<N8,KPack2,K8>.
    # Present a strided [owner,N/8,K/16,2,64] tensor map so the Shared tile is
    # N-major and can be reshaped directly to the MMA [N,K] operand.
    descriptor_shape = (
        weight_shape[0], weight_shape[2], weight_shape[1], 2, 64
    )
    descriptor_strides = (
        weight_strides[0], weight_strides[2], weight_strides[1], 64, 1
    )
    descriptor_block_shape = (1, block_n // 8, block_k // 16, 2, 64)
    descriptor_kind = str(parameters.get("descriptor_kind", "single"))
    if descriptor_kind == "table":
        rendered_block_shape = descriptor_block_shape[1:]
    elif descriptor_kind == "single":
        rendered_block_shape = descriptor_block_shape
    else:
        raise CodegenError(
            f"PackedQKV MMA has unsupported descriptor kind {descriptor_kind!r}."
        )
    expected_shared_shapes = (
        (num_stages, *rendered_block_shape),
        (1, source_domain["capacity"]),
    )
    if tuple(tuple(value["shape"]) for value in workspaces) != expected_shared_shapes:
        raise CodegenError(
            "PackedQKV MMA Shared workspace shapes disagree with its selected tiles."
        )
    descriptor_offset_bytes = (
        int(weight_abi["pool_byte_offset"])
        if str(weight_abi.get("storage")) in {"rdata", "workspace"}
        else 0
    )
    if descriptor_kind == "table":
        descriptor_request = packed_owner_prefix_tensor_map_table_request(
            weight_abi,
            parameter="weight_descriptor",
            source=str(weight["runtime_argument"]),
            offset_bytes=descriptor_offset_bytes,
            owner_count=weight_shape[0],
            descriptor_shape=descriptor_shape,
            descriptor_strides=descriptor_strides,
            block_shape=descriptor_block_shape,
        )
        descriptor_offsets = (
            "tl.full((), 0, tl.int32)",
            f"tl.full((), qkv_k_tile * {block_k // 16}, tl.int32)",
            "tl.full((), 0, tl.int32)",
            "tl.full((), 0, tl.int32)",
        )
    elif descriptor_kind == "single":
        descriptor_request = {
            "parameter": "weight_descriptor",
            "source": str(weight["runtime_argument"]),
            "kind": "single",
            "offset_bytes": descriptor_offset_bytes,
            "dtype": str(weight_abi["scalar_dtype"]),
            "shape": descriptor_shape,
            "strides": descriptor_strides,
            "block_shape": descriptor_block_shape,
            "source_shape_axes": tuple(() for _ in descriptor_shape),
            "padding": "zero",
        }
        descriptor_offsets = (
            "tl.full((), shard_index, tl.int32)",
            "tl.full((), 0, tl.int32)",
            f"tl.full((), qkv_k_tile * {block_k // 16}, tl.int32)",
            "tl.full((), 0, tl.int32)",
            "tl.full((), 0, tl.int32)",
        )
    result.update({
        "host_tensor_descriptor_requests": (descriptor_request,),
        "weight_descriptor": "weight_descriptor",
        "descriptor_kind": descriptor_kind,
        "descriptor_block_shape": rendered_block_shape,
        "descriptor_offsets": descriptor_offsets,
        "pipeline_contract": dict(pipeline),
        "shared_workspaces": tuple(workspaces),
        "pipeline_channel_field_names": {"weight": ("weight",)},
        "block_n": block_n,
        "block_k": block_k,
        "num_stages": num_stages,
        "num_k_tiles": source_domain["capacity"] // block_k,
        "consumer_warps": consumer_warps,
        "worker_width": worker_width,
        "producer_warps": int(parameters["producer_warps"]),
        "producer_registers": int(parameters["producer_registers"]),
    })
    if variant == "packed_gemv_smem_pipeline":
        reduction_group = parameters.get("reduction_group")
        threads_n = block_n // consumer_warps
        if (
            not isinstance(reduction_group, int) or isinstance(reduction_group, bool)
            or reduction_group <= 0 or reduction_group & (reduction_group - 1)
            or block_k % reduction_group or block_n % consumer_warps
            or threads_n <= 0 or worker_width % threads_n
            or reduction_group % (worker_width // threads_n)
        ):
            raise CodegenError("PackedQKV SIMT has an invalid reduction/worker partition.")
        result.update({
            "reduction_group": reduction_group,
            "reduction_groups_per_stage": block_k // reduction_group,
            "consumer_size_per_thread": (1, reduction_group // (worker_width // threads_n)),
            "consumer_threads_per_warp": (threads_n, worker_width // threads_n),
            "consumer_warps_per_cta": (consumer_warps, 1),
        })
    return result


def _qkv_rope_with_cache_call(raw) -> dict[str, object]:
    """Encode fused Q/K normalization, RoPE, and paged-cache writes.

    The algorithm is expressed entirely in the local-buffer ABI.  Distribution
    affects active bounds and coordinate expressions; it does not select a
    model-, mesh-, or accelerator-specific kernel body.  Normalized suffix axes
    must be owner-local because this non-collective variant computes their
    statistics without a cross-owner reduction.
    """

    q = _buffer(raw, "inputs", "qkv", 0)
    k = _buffer(raw, "inputs", "qkv", 1)
    v = _buffer(raw, "inputs", "qkv", 2)
    return _encode_qkv_rope_with_cache(raw, q, k, v)


def _gather_reduce_qkv_rope_with_cache_call(raw) -> dict[str, object]:
    """Encode QKV RoPE directly over compact Sum-partial projection outputs."""

    partials = tuple(_buffer(raw, "inputs", "qkv", index) for index in range(3))
    attrs = raw.get("semantic_attrs", {})
    logical_type = _qkv_type_attribute(attrs, "logical_qkv_type")
    logical = tuple(
        {
            **dict(source),
            "abi": _logical_view_abi(source["abi"], field),
        }
        for source, field in zip(partials, logical_type.fields, strict=True)
    )
    return _encode_qkv_rope_with_cache(
        raw, *logical, partial_sources=partials
    )


def _encode_qkv_rope_with_cache(
    raw,
    q,
    k,
    v,
    *,
    partial_sources=None,
) -> dict[str, object]:
    """Build the common semantic context for materialized or partial QKV."""

    q_scale = _buffer(raw, "inputs", "q_scale")
    k_scale = _buffer(raw, "inputs", "k_scale")
    q_bias = _buffer(raw, "inputs", "q_bias")
    k_bias = _buffer(raw, "inputs", "k_bias")
    cosine = _buffer(raw, "inputs", "cos")
    sine = _buffer(raw, "inputs", "sin")
    state = tuple(_buffer(raw, "inputs", "state", index) for index in range(5))
    layer_id = _buffer(raw, "inputs", "layer_id")
    advance = _buffer(raw, "inputs", "advance_sequence")
    q_result = _buffer(raw, "outputs", "result_0")

    attrs = raw.get("semantic_attrs", {})
    qkv_layout = _attention_layout(attrs, "qkv_layout")
    output_layout = _attention_layout(attrs, "attention_layout")
    q_axis = _normalized_axis(attrs, "q_axis", len(qkv_layout))
    k_axis = _normalized_axis(attrs, "k_axis", len(qkv_layout))
    q_context = _qkv_norm_rope_context(
        q,
        q_scale,
        q_bias,
        cosine,
        sine,
        q_result,
        input_layout=qkv_layout,
        output_layout=output_layout,
        axis=q_axis,
        epsilon=attrs.get("q_epsilon"),
        use_mean=attrs.get("q_use_mean"),
        round_before_scale=bool(attrs.get("q_round_before_scale", False)),
        prefix="qkv_q",
        tile=raw["parameters"]["elements_per_program"],
    )
    k_context = _qkv_norm_rope_context(
        k,
        k_scale,
        k_bias,
        cosine,
        sine,
        None,
        input_layout=qkv_layout,
        output_layout=output_layout,
        axis=k_axis,
        epsilon=attrs.get("k_epsilon"),
        use_mean=attrs.get("k_use_mean"),
        round_before_scale=bool(attrs.get("k_round_before_scale", False)),
        prefix="qkv_k",
        tile=raw["parameters"]["elements_per_program"],
    )
    v_context = _attention_scalar_domain(v["abi"], qkv_layout, "qkv_v_offsets")

    if partial_sources is not None:
        q_context["partial_reduce"] = _partial_qkv_access(
            partial_sources[0], q_context["reduce_domain"], qkv_layout,
            "qkv_q_partial_member",
        )
        q_context["partial_input"] = _partial_qkv_access(
            partial_sources[0], q_context["apply_domain"], qkv_layout,
            "qkv_q_partial_member",
        )
        q_context["partial_partner"] = _partial_qkv_access(
            partial_sources[0], q_context["partner_domain"], qkv_layout,
            "qkv_q_partial_member",
        )
        k_context["partial_reduce"] = _partial_qkv_access(
            partial_sources[1], k_context["reduce_domain"], qkv_layout,
            "qkv_k_partial_member",
        )
        k_context["partial_input"] = _partial_qkv_access(
            partial_sources[1], k_context["apply_domain"], qkv_layout,
            "qkv_k_partial_member",
        )
        k_context["partial_partner"] = _partial_qkv_access(
            partial_sources[1], k_context["partner_domain"], qkv_layout,
            "qkv_k_partial_member",
        )
        v_context["partial_input"] = _partial_qkv_access(
            partial_sources[2], v_context, qkv_layout,
            "qkv_v_partial_member",
        )

    cache_abi = state[0]["abi"]
    cache_shape = _static_shape(cache_abi, "logical_shape")
    if len(cache_shape) != 6:
        raise CodegenError(
            "QKVRoPEWithCache requires paged cache "
            "[block, layer, key/value, block_offset, head, dimension]."
        )
    cache_lanes = int(cache_abi.get("scalar_lane_count", 1))
    if cache_lanes <= 0:
        raise CodegenError("QKVRoPEWithCache cache lanes must be positive.")
    cache_head_dim = cache_shape[-1] * cache_lanes
    if (
        q_context["head_dim"] != cache_head_dim
        or k_context["head_dim"] != cache_head_dim
        or v_context["scalar_shape"][qkv_layout.index("dim")] != cache_head_dim
    ):
        raise CodegenError(
            "QKVRoPEWithCache Q/K/V scalar head dimensions disagree with the cache."
        )
    if (
        k_context["apply_domain"]["global_extents"]["head"] != cache_shape[-2]
        or v_context["global_extents"]["head"] != cache_shape[-2]
    ):
        raise CodegenError(
            "QKVRoPEWithCache K/V head counts disagree with the cache."
        )

    layer_expression = _scalar_expression(layer_id)
    cache_position = "qkv_cache_position"
    cache_block = "qkv_cache_physical_block"
    cache_offset = "qkv_cache_block_offset"
    k_context["cache_offset"] = _paged_cache_scalar_offset(
        cache_abi,
        k_context["apply_domain"],
        cache_index=0,
        layer=layer_expression,
        physical_block=cache_block,
        block_offset=cache_offset,
    )
    v_context["cache_offset"] = _paged_cache_scalar_offset(
        cache_abi,
        v_context,
        cache_index=1,
        layer=layer_expression,
        physical_block=cache_block,
        block_offset=cache_offset,
    )
    v_context.update({
        "input": _pointer(v),
        "input_offset": v_context["offset"],
        "writer_active": _distributed_unique_writer_active(
            v["abi"], writer_ordinal=1
        ),
        "tile": _bounded_vector_tile(
            raw["parameters"]["elements_per_program"],
            v_context["capacity"],
            name="QKVRoPEWithCache V elements_per_program",
        ),
    })
    k_context["writer_active"] = _distributed_unique_writer_active(
        k["abi"], writer_ordinal=0
    )
    q_context["compute_active"] = q_context.get("writer_active", "True")
    k_context["compute_active"] = k_context["writer_active"]
    v_context["compute_active"] = v_context["writer_active"]

    headwise_partial = (
        partial_sources is not None
        and q_context["tile"] == q_context["reduction_capacity"]
        and k_context["tile"] == k_context["reduction_capacity"]
        and v_context["tile"] == v_context["capacity"]
        and q_context["head_dim"] % 2 == 0
        and k_context["head_dim"] % 2 == 0
    )

    return {
        "q": q_context,
        "k": k_context,
        "v": v_context,
        "kv_cache": _pointer(state[0]),
        "sequence_lengths": _pointer(state[2]),
        "slot_mapping": _pointer(state[3]),
        "block_table": _pointer(state[4]),
        "layer_id": layer_expression,
        "advance_sequence": _scalar_expression(advance),
        "block_size": cache_shape[3],
        "headwise_partial": headwise_partial,
    }


def _qkv_type_attribute(attrs, name: str) -> TupleType:
    value = attrs.get(name) if isinstance(attrs, Mapping) else None
    if isinstance(value, Mapping):
        value = type_from_data(value)
    if not isinstance(value, TupleType) or len(value.fields) != 3:
        raise CodegenError(
            f"GatherReduceQKVRoPEWithCache requires a three-field {name}."
        )
    return value


def _logical_view_abi(
    source_abi: Mapping[str, object], logical_type
) -> dict[str, object]:
    """Describe a zero-copy logical domain without changing its backing buffer."""

    if not isinstance(logical_type, DistributedType):
        raise CodegenError(
            "GatherReduceQKVRoPEWithCache logical views must be distributed."
        )
    tensor = logical_type.tensor
    shape = tuple(_fixed_dimension(value, "logical QKV view") for value in tensor.shape)
    lane_count = (
        prod(tensor.dtype.lanes, start=1)
        if isinstance(tensor.dtype, VectorType)
        else 1
    )
    if (
        str(source_abi.get("dtype")) != tensor.dtype.value
        or int(source_abi.get("scalar_lane_count", 1)) != lane_count
    ):
        raise CodegenError(
            "GatherReduceQKVRoPEWithCache logical view changes its backing dtype."
        )
    coordinates = tuple(
        f"shard_coord_{axis}" for axis in range(logical_type.placement.rank)
    )
    shard = local_shard_descriptor(logical_type, coordinates)
    local_shape = tuple(
        _fixed_dimension(value, "logical QKV local capacity")
        for value in shard.local_capacity_shape
    )
    strides = _dense_strides(shape)
    return {
        **dict(source_abi),
        "logical_shape": shape,
        "local_capacity_shape": local_shape,
        "active_shape_expressions": tuple(
            emit_dimension(value) for value in shard.active_shape
        ),
        "logical_coordinate_expressions": tuple(
            emit_dimension(axis.map_local_to_global(f"local_coord_{index}"))
            for index, axis in enumerate(shard.axes)
        ),
        "storage_strides": strides,
        "scalar_storage_strides": tuple(value * lane_count for value in strides),
        "coordinate_space": "local",
        "distributed_type": logical_type.to_data(),
    }


def _fixed_dimension(value, owner: str) -> int:
    try:
        return value.fixed_value
    except ValueError as error:
        raise CodegenError(f"Triton renderer requires a fixed {owner}.") from error


def _dense_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    running = 1
    result = [1] * len(shape)
    for axis in range(len(shape) - 1, -1, -1):
        result[axis] = running
        running *= shape[axis]
    return tuple(result)


def _partial_qkv_access(
    source,
    logical_domain: Mapping[str, object],
    layout: tuple[str, str, str],
    member: str,
) -> dict[str, object]:
    """Map a logical Q/K/V scalar to its packed partial owner and offset."""

    abi = source["abi"]
    if str(abi.get("storage_kind")) != "compact_per_owner":
        raise CodegenError(
            "GatherReduceQKVRoPEWithCache requires compact-per-owner partial input."
        )
    distributed = abi.get("distributed_type")
    if not isinstance(distributed, Mapping):
        raise CodegenError(
            "GatherReduceQKVRoPEWithCache partial input has no DistributedType ABI."
        )
    partial = distributed.get("partial")
    if (
        not isinstance(partial, Mapping)
        or str(partial.get("reduce_op")) != "sum"
    ):
        raise CodegenError(
            "GatherReduceQKVRoPEWithCache requires Sum partial input."
        )
    placement = distributed.get("placement")
    if not isinstance(placement, Mapping):
        raise CodegenError(
            "GatherReduceQKVRoPEWithCache partial input has no placement ABI."
        )
    hierarchy = tuple(int(value) for value in placement.get("hierarchy", ()))
    partial_axes = tuple(int(value) for value in partial.get("axes", ()))
    if (
        not hierarchy
        or not partial_axes
        or tuple(sorted(set(partial_axes))) != partial_axes
        or any(axis < 0 or axis >= len(hierarchy) for axis in partial_axes)
    ):
        raise CodegenError(
            "GatherReduceQKVRoPEWithCache has invalid partial placement axes."
        )

    source_shape = _static_shape(abi, "logical_shape")
    lane_count = int(abi.get("scalar_lane_count", 1))
    logical_extents = tuple(
        int(logical_domain["global_extents"][kind]) for kind in layout
    )
    logical_coordinates = tuple(
        str(logical_domain["global_by_kind"][kind]) for kind in layout
    )
    scalar_linear = _flatten_expression(logical_coordinates, logical_extents)
    source_physical_linear = (
        scalar_linear if lane_count == 1 else f"(({scalar_linear}) // {lane_count})"
    )
    source_global = _unflatten_expression(source_physical_linear, source_shape)
    lane = None if lane_count == 1 else f"(({scalar_linear}) % {lane_count})"

    mesh_coordinates = list(_mesh_coordinates(len(hierarchy)))
    remaining_member = f"({member})"
    for position, axis in enumerate(reversed(partial_axes)):
        extent = hierarchy[axis]
        final = position == len(partial_axes) - 1
        mesh_coordinates[axis] = (
            remaining_member if final else f"(({remaining_member}) % {extent})"
        )
        if not final:
            remaining_member = f"(({remaining_member}) // {extent})"

    policies = distributed.get("axis_policies")
    if not isinstance(policies, (tuple, list)) or len(policies) != len(source_shape):
        raise CodegenError(
            "GatherReduceQKVRoPEWithCache partial axis policies are incomplete."
        )
    local_coordinates = []
    for coordinate, extent, policy in zip(
        source_global, source_shape, policies, strict=True
    ):
        local_coordinates.append(
            _invert_split_policy(
                coordinate,
                str(extent),
                policy,
                hierarchy,
                mesh_coordinates,
                partial_axes,
            )
        )
    owner = _linear_owner_expression(mesh_coordinates, hierarchy)
    owner_stride = int(abi.get("component_stride_scalar_elements", 0))
    if owner_stride <= 0:
        raise CodegenError(
            "GatherReduceQKVRoPEWithCache partial input has no owner stride."
        )
    return {
        "storage": emit_storage_pointer(abi, str(source["runtime_argument"])),
        "owner": owner,
        "owner_stride": owner_stride,
        "offset": emit_local_scalar_offset(
            abi,
            local_coordinates,
            lane_coordinate=lane,
            shard_coordinates=mesh_coordinates,
        ),
        "member_count": prod(hierarchy[axis] for axis in partial_axes),
        "member_width": _next_power_of_two(
            prod(hierarchy[axis] for axis in partial_axes)
        ),
        "scalar_dtype": emit_triton_scalar_type(str(abi.get("scalar_dtype"))),
    }


def _next_power_of_two(value: int) -> int:
    if value <= 0:
        raise CodegenError("Triton partial-owner width must be positive.")
    return 1 << (value - 1).bit_length()


def _flatten_expression(
    coordinates: tuple[str, ...], extents: tuple[int, ...]
) -> str:
    if len(coordinates) != len(extents) or not coordinates:
        raise CodegenError("Cannot flatten an empty or rank-mismatched QKV domain.")
    expression = f"({coordinates[0]})"
    for coordinate, extent in zip(coordinates[1:], extents[1:], strict=True):
        expression = f"(({expression}) * {extent} + ({coordinate}))"
    return expression


def _unflatten_expression(
    linear: str, shape: tuple[int, ...]
) -> tuple[str, ...]:
    if not shape:
        raise CodegenError("Cannot unflatten a scalar into an empty QKV shape.")
    strides = _dense_strides(shape)
    return tuple(
        (
            f"(({linear}) // {stride})"
            if axis == 0
            else f"((({linear}) // {stride}) % {extent})"
        )
        for axis, (extent, stride) in enumerate(zip(shape, strides, strict=True))
    )


def _invert_split_policy(
    coordinate: str,
    parent_extent: str,
    policy,
    hierarchy: tuple[int, ...],
    mesh_coordinates: list[str],
    partial_axes: tuple[int, ...],
) -> str:
    if not isinstance(policy, Mapping):
        raise CodegenError("Serialized QKV axis policy must be a mapping.")
    if str(policy.get("kind")) != "split":
        return coordinate
    stages = policy.get("stages")
    if not isinstance(stages, (tuple, list)) or not stages:
        raise CodegenError("Serialized QKV split policy requires stages.")
    local = coordinate
    extent = parent_extent
    for stage in stages:
        if not isinstance(stage, Mapping):
            raise CodegenError("Serialized QKV split stage must be a mapping.")
        axes = tuple(int(value) for value in stage.get("hierarchy_axes", ()))
        if (
            not axes
            or any(axis < 0 or axis >= len(hierarchy) for axis in axes)
            or set(axes) & set(partial_axes)
        ):
            raise CodegenError(
                "QKV split axes must be valid and disjoint from partial axes."
            )
        count = prod(hierarchy[axis] for axis in axes)
        distribution = stage.get("distribution")
        if not isinstance(distribution, Mapping):
            raise CodegenError("Serialized QKV split stage has no distribution.")
        kind = str(distribution.get("kind"))
        if kind == "contiguous":
            granularity = distribution.get("granularity")
            capacity = (
                _serialized_fixed_dimension(granularity)
                if granularity is not None
                else f"(-(-({extent}) // {count}))"
            )
            owner = f"(({local}) // ({capacity}))"
            local = f"(({local}) - ({owner}) * ({capacity}))"
            extent = (
                f"tl.maximum(0, tl.minimum(({capacity}), "
                f"({extent}) - ({owner}) * ({capacity})))"
            )
        elif kind == "block_cyclic":
            block = int(distribution.get("block_size", 0))
            if block <= 0:
                raise CodegenError("QKV block-cyclic split requires a block size.")
            cycle = count * block
            owner = f"((({local}) // {block}) % {count})"
            local = f"((({local}) // {cycle}) * {block} + ({local}) % {block})"
            extent = (
                f"((({extent}) // {cycle}) * {block} + "
                f"tl.minimum(tl.maximum(({extent}) % {cycle} - "
                f"({owner}) * {block}, 0), {block}))"
            )
        else:
            raise CodegenError(f"Unsupported QKV split distribution {kind!r}.")
        remaining = owner
        for position, axis in enumerate(reversed(axes)):
            axis_extent = hierarchy[axis]
            final = position == len(axes) - 1
            value = remaining if final else f"(({remaining}) % {axis_extent})"
            current = mesh_coordinates[axis]
            if axis in partial_axes or (
                current not in _mesh_coordinates(len(hierarchy))
                and current != value
            ):
                raise CodegenError(
                    "A placement axis is assigned by multiple QKV policies."
                )
            mesh_coordinates[axis] = value
            if not final:
                remaining = f"(({remaining}) // {axis_extent})"
    return local


def _serialized_fixed_dimension(value) -> str:
    if isinstance(value, Mapping) and value.get("kind") == "fixed":
        fixed = int(value.get("value", 0))
    elif isinstance(value, int) and not isinstance(value, bool):
        fixed = value
    else:
        raise CodegenError(
            "Triton QKV contiguous granularity must be statically fixed."
        )
    if fixed <= 0:
        raise CodegenError("Triton QKV split granularity must be positive.")
    return str(fixed)


def _attention_layout(attrs, name: str) -> tuple[str, str, str]:
    raw = attrs.get(name) if isinstance(attrs, Mapping) else None
    if not isinstance(raw, (tuple, list)):
        raise CodegenError(f"QKVRoPEWithCache requires {name} metadata.")
    layout = tuple(str(value) for value in raw)
    if len(layout) != 3 or set(layout) != {"seq", "head", "dim"}:
        raise CodegenError(
            f"QKVRoPEWithCache has invalid {name} {layout!r}."
        )
    return layout


def _normalized_axis(attrs, name: str, rank: int) -> int:
    value = attrs.get(name) if isinstance(attrs, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int):
        raise CodegenError(f"QKVRoPEWithCache {name} must be an integer.")
    result = value + rank if value < 0 else value
    if result < 0 or result >= rank:
        raise CodegenError(
            f"QKVRoPEWithCache {name} {value} is outside rank {rank}."
        )
    return result


def _qkv_norm_rope_context(
    source,
    scale,
    bias,
    cosine,
    sine,
    result,
    *,
    input_layout,
    output_layout,
    axis,
    epsilon,
    use_mean,
    prefix,
    tile,
    round_before_scale=False,
) -> dict[str, object]:
    source_abi = source["abi"]
    _require_owner_local_reduction(source_abi, axis, prefix)
    scalar_shape = _attention_scalar_shape(source_abi, input_layout)
    outer_shape = scalar_shape[:axis]
    reduction_shape = scalar_shape[axis:]
    outer_coordinates = _unflattened_coordinates(
        outer_shape, f"{prefix}_outer_index"
    )
    reduction_coordinates = _unflattened_coordinates(
        reduction_shape, f"{prefix}_reduce_offsets"
    )
    apply_coordinates = _unflattened_coordinates(
        reduction_shape, f"{prefix}_element_offsets"
    )
    reduce_domain = _attention_scalar_coordinates(
        source_abi,
        input_layout,
        (*outer_coordinates, *reduction_coordinates),
    )
    apply_domain = _attention_scalar_coordinates(
        source_abi,
        input_layout,
        (*outer_coordinates, *apply_coordinates),
    )
    dimension = apply_domain["global_by_kind"]["dim"]
    local_dimension = apply_domain["local_by_kind"]["dim"]
    head_dim = apply_domain["global_extents"]["dim"]
    partner = (
        f"tl.where(({local_dimension}) < {head_dim} // 2, "
        f"({local_dimension}) + {head_dim} // 2, "
        f"({local_dimension}) - {head_dim} // 2)"
    )
    partner_coordinates = dict(apply_domain["local_by_kind"])
    partner_coordinates["dim"] = partner
    partner_domain = _attention_scalar_coordinates_from_kinds(
        source_abi,
        input_layout,
        partner_coordinates,
    )
    scale_offset = _aligned_parameter_offset(
        scale["abi"], apply_domain, input_layout, start_axis=axis,
        owner=f"{prefix} scale",
    )
    bias_offset = _aligned_parameter_offset(
        bias["abi"], apply_domain, input_layout, start_axis=axis,
        owner=f"{prefix} bias",
    )
    cos_offset = _aligned_parameter_offset(
        cosine["abi"], apply_domain, input_layout,
        start_axis=len(input_layout) - len(_static_shape(cosine["abi"], "logical_shape")),
        owner=f"{prefix} cosine",
    )
    sin_offset = _aligned_parameter_offset(
        sine["abi"], apply_domain, input_layout,
        start_axis=len(input_layout) - len(_static_shape(sine["abi"], "logical_shape")),
        owner=f"{prefix} sine",
    )
    partner_scale_offset = _aligned_parameter_offset(
        scale["abi"], partner_domain, input_layout, start_axis=axis,
        owner=f"{prefix} partner scale",
    )
    partner_bias_offset = _aligned_parameter_offset(
        bias["abi"], partner_domain, input_layout, start_axis=axis,
        owner=f"{prefix} partner bias",
    )
    capacity = prod(reduction_shape, start=1)
    try:
        epsilon_value = float(epsilon)
    except (TypeError, ValueError) as error:
        raise CodegenError(f"QKVRoPEWithCache {prefix} epsilon must be numeric.") from error
    if epsilon_value <= 0 or not isinstance(use_mean, bool):
        raise CodegenError(
            f"QKVRoPEWithCache {prefix} requires positive epsilon and boolean use_mean."
        )
    context = {
        "input": _pointer(source),
        "scale": _pointer(scale),
        "bias": _pointer(bias),
        "cosine": _pointer(cosine),
        "sine": _pointer(sine),
        "outer_capacity": prod(outer_shape, start=1),
        "reduction_capacity": capacity,
        "normalization_size": prod(
            apply_domain["global_extents"][input_layout[index]]
            for index in range(axis, len(input_layout))
        ),
        "reduce_active": reduce_domain["active"],
        "apply_active": apply_domain["active"],
        "reduce_input_offset": reduce_domain["offset"],
        "input_offset": apply_domain["offset"],
        "partner_offset": partner_domain["offset"],
        "scale_offset": scale_offset,
        "bias_offset": bias_offset,
        "partner_scale_offset": partner_scale_offset,
        "partner_bias_offset": partner_bias_offset,
        "cosine_offset": cos_offset,
        "sine_offset": sin_offset,
        "dimension": dimension,
        "head_dim": head_dim,
        "epsilon": repr(epsilon_value),
        "round_before_scale": round_before_scale,
        "input_type": _triton_dtype(str(source_abi["scalar_dtype"])),
        "use_mean": use_mean,
        "apply_domain": apply_domain,
        "reduce_domain": reduce_domain,
        "partner_domain": partner_domain,
        "tile": _bounded_vector_tile(
            tile,
            capacity,
            name=f"QKVRoPEWithCache {prefix} reduction tile",
        ),
    }
    if result is not None:
        output_offset, output_active = _attention_result_access(
            result["abi"], output_layout, apply_domain
        )
        context.update({
            "output": _pointer(result),
            "output_offset": output_offset,
            "output_active": output_active,
            "writer_active": _canonical_writer_active(result["abi"]),
        })
    return context


def _attention_scalar_domain(abi, layout, flat_coordinate):
    scalar_shape = _attention_scalar_shape(abi, layout)
    coordinates = _unflattened_coordinates(scalar_shape, flat_coordinate)
    result = _attention_scalar_coordinates(abi, layout, coordinates)
    return {**result, "capacity": prod(scalar_shape), "scalar_shape": scalar_shape}


def _attention_scalar_shape(abi, layout):
    shape = list(_static_shape(abi, "local_capacity_shape"))
    logical = _static_shape(abi, "logical_shape")
    if len(shape) != len(layout) or len(logical) != len(layout):
        raise CodegenError(
            "QKVRoPEWithCache attention operands must match their rank-three layout."
        )
    lanes = int(abi.get("scalar_lane_count", 1))
    if lanes <= 0:
        raise CodegenError("QKVRoPEWithCache vector lane count must be positive.")
    shape[layout.index("dim")] *= lanes
    return tuple(shape)


def _attention_scalar_coordinates(abi, layout, scalar_coordinates):
    return _attention_scalar_coordinates_from_kinds(
        abi,
        layout,
        {kind: scalar_coordinates[index] for index, kind in enumerate(layout)},
    )


def _attention_scalar_coordinates_from_kinds(abi, layout, local_by_kind):
    lanes = int(abi.get("scalar_lane_count", 1))
    dimension_axis = layout.index("dim")
    physical_local = []
    lane_coordinate = None
    for axis, kind in enumerate(layout):
        coordinate = str(local_by_kind[kind])
        if axis == dimension_axis and lanes != 1:
            physical_local.append(f"(({coordinate}) // {lanes})")
            lane_coordinate = f"(({coordinate}) % {lanes})"
        else:
            physical_local.append(coordinate)
    global_by_kind = {}
    for axis, kind in enumerate(layout):
        coordinate = emit_logical_coordinate(abi, axis, physical_local)
        if axis == dimension_axis and lanes != 1:
            coordinate = f"(({coordinate}) * {lanes} + ({lane_coordinate}))"
        global_by_kind[kind] = coordinate
    local_shape = _static_shape(abi, "local_capacity_shape")
    logical_shape = _static_shape(abi, "logical_shape")
    active_parts = []
    global_extents = {}
    for axis, kind in enumerate(layout):
        lane_factor = lanes if axis == dimension_axis else 1
        active_parts.append(
            f"(({local_by_kind[kind]}) < ({emit_active_extent(abi, axis)}) * {lane_factor})"
        )
        global_extents[kind] = logical_shape[axis] * lane_factor
    return {
        "local_by_kind": dict(local_by_kind),
        "global_by_kind": global_by_kind,
        "global_extents": global_extents,
        "physical_local_coordinates": tuple(physical_local),
        "lane_coordinate": lane_coordinate,
        "active": " & ".join(active_parts) or "True",
        "offset": emit_local_scalar_offset(
            abi, physical_local, lane_coordinate=lane_coordinate
        ),
        "local_capacity_shape": local_shape,
    }


def _require_owner_local_reduction(abi, axis, owner):
    local_shape = _static_shape(abi, "local_capacity_shape")
    logical_shape = _static_shape(abi, "logical_shape")
    mappings = tuple(str(value) for value in abi["logical_coordinate_expressions"])
    for reduction_axis in range(axis, len(local_shape)):
        if (
            local_shape[reduction_axis] != logical_shape[reduction_axis]
            or mappings[reduction_axis] != f"local_coord_{reduction_axis}"
        ):
            raise CodegenError(
                f"QKVRoPEWithCache {owner} normalized suffix is sharded; "
                "select a gather-reduce variant or insert Boxing."
            )


def _aligned_parameter_offset(
    abi,
    source_domain,
    source_layout,
    *,
    start_axis,
    owner,
):
    shape = _static_shape(abi, "logical_shape")
    if start_axis < 0 or start_axis + len(shape) != len(source_layout):
        raise CodegenError(
            f"QKVRoPEWithCache {owner} rank does not align with its source suffix."
        )
    lanes = int(abi.get("scalar_lane_count", 1))
    dimension_parameter_axis = None
    local_coordinates = []
    global_coordinates = []
    for parameter_axis, extent in enumerate(shape):
        source_kind = source_layout[start_axis + parameter_axis]
        local_coordinate = source_domain["local_by_kind"][source_kind]
        global_coordinate = source_domain["global_by_kind"][source_kind]
        scalar_extent = extent
        if source_kind == "dim" and lanes != 1:
            dimension_parameter_axis = parameter_axis
            scalar_extent *= lanes
        if scalar_extent == 1:
            local_coordinate = global_coordinate = "0"
        if source_kind == "dim" and lanes != 1:
            local_coordinates.append(f"(({local_coordinate}) // {lanes})")
            global_coordinates.append(f"(({global_coordinate}) // {lanes})")
        else:
            local_coordinates.append(str(local_coordinate))
            global_coordinates.append(str(global_coordinate))
    lane_coordinate = None
    if lanes != 1:
        if dimension_parameter_axis is None:
            raise CodegenError(
                f"QKVRoPEWithCache {owner} vector lanes do not map to dimension."
            )
        dimension = source_domain["local_by_kind"]["dim"]
        lane_coordinate = f"(({dimension}) % {lanes})"
    if str(abi.get("coordinate_space")) == "canonical_global":
        return emit_global_scalar_offset(
            abi, global_coordinates, lane_coordinate=lane_coordinate
        )
    return emit_local_scalar_offset(
        abi, local_coordinates, lane_coordinate=lane_coordinate
    )


def _attention_result_access(abi, layout, source_domain):
    shape = _static_shape(abi, "logical_shape")
    if len(shape) != len(layout):
        raise CodegenError("QKVRoPEWithCache Q output rank/layout mismatch.")
    lanes = int(abi.get("scalar_lane_count", 1))
    local_coordinates = []
    global_coordinates = []
    lane_coordinate = None
    active = []
    for axis, kind in enumerate(layout):
        local = source_domain["local_by_kind"][kind]
        global_value = source_domain["global_by_kind"][kind]
        lane_factor = lanes if kind == "dim" else 1
        expected = shape[axis] * lane_factor
        if expected != source_domain["global_extents"][kind]:
            raise CodegenError(
                "QKVRoPEWithCache Q output logical shape differs from Q input."
            )
        if kind == "dim" and lanes != 1:
            local_coordinates.append(f"(({local}) // {lanes})")
            global_coordinates.append(f"(({global_value}) // {lanes})")
            lane_coordinate = f"(({local}) % {lanes})"
        else:
            local_coordinates.append(str(local))
            global_coordinates.append(str(global_value))
        active.append(
            f"(({local}) < ({emit_active_extent(abi, axis)}) * {lane_factor})"
        )
    offset = (
        emit_global_scalar_offset(
            abi, global_coordinates, lane_coordinate=lane_coordinate
        )
        if str(abi.get("coordinate_space")) == "canonical_global"
        else emit_local_scalar_offset(
            abi, local_coordinates, lane_coordinate=lane_coordinate
        )
    )
    return offset, " & ".join(active) or "True"


def _paged_cache_scalar_offset(
    abi,
    source_domain,
    *,
    cache_index,
    layer,
    physical_block,
    block_offset,
):
    lanes = int(abi.get("scalar_lane_count", 1))
    dimension = source_domain["global_by_kind"]["dim"]
    coordinates = (
        physical_block,
        layer,
        str(cache_index),
        block_offset,
        source_domain["global_by_kind"]["head"],
        dimension if lanes == 1 else f"(({dimension}) // {lanes})",
    )
    lane_coordinate = None if lanes == 1 else f"(({dimension}) % {lanes})"
    return emit_local_scalar_offset(
        abi, coordinates, lane_coordinate=lane_coordinate
    )


def _rotary_embedding_call(raw) -> dict[str, object]:
    state = tuple(
        _buffer(raw, "inputs", "state", index) for index in range(5)
    )
    cosine = _buffer(raw, "outputs", "result_0")
    sine = _buffer(raw, "outputs", "result_1")
    cosine_abi = cosine["abi"]
    domain = _local_domain(cosine_abi, "rotary_offsets")
    shape = _static_shape(cosine_abi, "logical_shape")
    attrs = raw.get("semantic_attrs", {})
    return {
        "sequence_lengths": _pointer(state[2]),
        "cosine": _pointer(cosine),
        "sine": _pointer(sine),
        "local_capacity": domain["capacity"],
        "active": domain["active"],
        "local_coordinates": domain["local_coordinates"],
        "logical_coordinates": domain["logical_coordinates"],
        "cosine_offset": emit_local_scalar_offset(
            cosine_abi, domain["local_coordinates"]
        ),
        "sine_offset": _access_in_result_domain(
            sine["abi"], cosine_abi, domain
        ),
        "sequence_size": shape[-2] if len(shape) > 1 else 1,
        "head_dim": int(attrs["head_dim"]),
        "theta": repr(float(attrs["theta"])),
        "attention_scaling": repr(float(attrs["attention_scaling"])),
        "tile": _bounded_vector_tile(
            raw["parameters"]["elements_per_program"],
            domain["capacity"],
            name="rotary_embedding.elements_per_program",
        ),
    }


def _rope_call(raw) -> dict[str, object]:
    source = _buffer(raw, "inputs", "input")
    cosine = _buffer(raw, "inputs", "cos")
    sine = _buffer(raw, "inputs", "sin")
    result = _buffer(raw, "outputs", "result")
    result_abi = result["abi"]
    # VectorizedRoPE stores one or more VectorType lanes after the physical
    # tensor rank.  Iterate scalar payload coordinates so the same local-shard
    # renderer handles both nn.rope and ntt.vectorized_rope.
    domain = _scalar_local_domain(result_abi, "rope_offsets")
    shape = _static_shape(result_abi, "logical_shape")
    if len(shape) < 3:
        raise CodegenError("RoPE TIR requires [sequence, head, dimension].")
    local_coordinates = domain["local_coordinates"]
    logical_coordinates = domain["logical_coordinates"]
    result_lane_count = int(result_abi.get("scalar_lane_count", 1))
    lane_coordinate = domain.get("lane_coordinate")
    dimension = (
        logical_coordinates[-1]
        if result_lane_count == 1
        else (
            f"(({logical_coordinates[-1]}) * {result_lane_count} "
            f"+ ({lane_coordinate}))"
        )
    )
    head_dim = shape[-1] * result_lane_count
    partner_dimension = (
        f"tl.where(({dimension}) < {head_dim} // 2, "
        f"({dimension}) + {head_dim} // 2, "
        f"({dimension}) - {head_dim} // 2)"
    )

    def source_offset(scalar_dimension: str) -> str:
        abi = source["abi"]
        source_lane_count = int(abi.get("scalar_lane_count", 1))
        coordinates = list(
            logical_coordinates
            if str(abi["coordinate_space"]) == "canonical_global"
            else local_coordinates
        )
        source_lane = None
        if source_lane_count == 1:
            coordinates[-1] = scalar_dimension
        else:
            coordinates[-1] = f"(({scalar_dimension}) // {source_lane_count})"
            source_lane = f"(({scalar_dimension}) % {source_lane_count})"
        if str(abi["coordinate_space"]) == "canonical_global":
            return emit_global_scalar_offset(
                abi, coordinates, lane_coordinate=source_lane
            )
        return emit_local_scalar_offset(
            abi, coordinates, lane_coordinate=source_lane
        )

    def table_offset(binding: Mapping[str, object]) -> str:
        abi = binding["abi"]
        table_shape = _static_shape(abi, "logical_shape")
        if len(table_shape) > len(shape):
            raise CodegenError("RoPE rotary table rank exceeds the result rank.")
        table_lane_count = int(abi.get("scalar_lane_count", 1))
        result_coordinates = (
            logical_coordinates
            if str(abi["coordinate_space"]) == "canonical_global"
            else local_coordinates
        )
        start = len(shape) - len(table_shape)
        coordinates = [
            "0" if extent == 1 else result_coordinates[start + axis]
            for axis, extent in enumerate(table_shape)
        ]
        table_lane = None
        if table_lane_count == 1:
            coordinates[-1] = dimension
        else:
            coordinates[-1] = f"(({dimension}) // {table_lane_count})"
            table_lane = f"(({dimension}) % {table_lane_count})"
        if str(abi["coordinate_space"]) == "canonical_global":
            return emit_global_scalar_offset(
                abi, coordinates, lane_coordinate=table_lane
            )
        return emit_local_scalar_offset(
            abi, coordinates, lane_coordinate=table_lane
        )

    return {
        "source": _pointer(source),
        "cosine": _pointer(cosine),
        "sine": _pointer(sine),
        "result": _pointer(result),
        "local_capacity": domain["capacity"],
        "active": domain["active"],
        "dimension": dimension,
        "head_dim": head_dim,
        "source_offset": source_offset(dimension),
        "input_type": _triton_dtype(str(source["abi"]["scalar_dtype"])),
        "partner_offset": source_offset(partner_dimension),
        "cosine_offset": table_offset(cosine),
        "sine_offset": table_offset(sine),
        "result_offset": emit_local_scalar_offset(
            result_abi,
            local_coordinates,
            lane_coordinate=lane_coordinate,
        ),
        "tile": int(raw["parameters"]["elements_per_program"]),
    }


def _cache_update_call(raw) -> dict[str, object]:
    slots = _buffer(raw, "inputs", "slots")
    state = tuple(
        _buffer(raw, "inputs", "state", index) for index in range(5)
    )
    layer_id = _buffer(raw, "inputs", "layer_id")
    advance = _buffer(raw, "inputs", "advance_sequence")
    slots_abi = slots["abi"]
    domain = _local_domain(slots_abi, "cache_offsets")
    shape = _static_shape(slots_abi, "logical_shape")
    cache_shape = _static_shape(state[0]["abi"], "logical_shape")
    attrs = raw.get("semantic_attrs", {})
    return {
        "slots": _pointer(slots),
        "kv_cache": _pointer(state[0]),
        "sequence_lengths": _pointer(state[2]),
        "slot_mapping": _pointer(state[3]),
        "block_table": _pointer(state[4]),
        "layer_id": _scalar_expression(layer_id),
        "advance_sequence": _scalar_expression(advance),
        "cache_index": 0 if attrs.get("cache_kind") == "key" else 1,
        "num_layers": cache_shape[1],
        "num_kv_heads": shape[-2],
        "head_dim": shape[-1],
        "block_size": cache_shape[3],
        "local_capacity": domain["capacity"],
        "active": domain["active"],
        "logical_head": domain["logical_coordinates"][-2],
        "logical_dimension": domain["logical_coordinates"][-1],
        "slots_offset": emit_local_scalar_offset(
            slots_abi, domain["local_coordinates"]
        ),
        "tile": int(raw["parameters"]["elements_per_program"]),
    }


def _paged_attention_partial_call(raw) -> dict[str, object]:
    query = _buffer(raw, "inputs", "q")
    state = tuple(
        _buffer(raw, "inputs", "state", index) for index in range(5)
    )
    layer_id = _buffer(raw, "inputs", "layer_id")
    partial_max = _buffer(raw, "outputs", "result_0")
    partial_sum = _buffer(raw, "outputs", "result_1")
    partial_accumulator = _buffer(raw, "outputs", "result_2")
    max_abi = partial_max["abi"]
    max_domain = _local_domain(max_abi, "attention_state_offsets")
    query_abi = query["abi"]
    cache_abi = state[0]["abi"]
    query_shape = _static_shape(query_abi, "logical_shape")
    cache_shape = _static_shape(cache_abi, "logical_shape")
    attrs = raw.get("semantic_attrs", {})
    layout = tuple(str(value) for value in attrs["layout"])
    head_axis = layout.index("head")
    dim_axis = layout.index("dim")
    query_head_dim = query_shape[dim_axis] * int(
        query_abi.get("scalar_lane_count", 1)
    )
    cache_head_dim = cache_shape[-1] * int(
        cache_abi.get("scalar_lane_count", 1)
    )
    if query_head_dim != cache_head_dim:
        raise CodegenError(
            "Paged-attention query and KV-cache scalar head dimensions differ: "
            f"{query_head_dim} != {cache_head_dim}."
        )
    split_axis = int(attrs["split_hierarchy_axis"])
    partial_axis = _single_partial_hierarchy_axis(max_abi)
    if split_axis != partial_axis:
        raise CodegenError(
            "Paged-attention partial-state ABI disagrees with its split axis."
        )
    state_coordinates = list(max_domain["local_coordinates"])
    state_coordinates[layout.index("dim")] = "0"

    def head_pointer(binding: Mapping[str, object]) -> str:
        offset = emit_local_scalar_offset(
            binding["abi"], state_coordinates
        )
        return f"({_pointer(binding)} + ({offset}))"

    result = {
        # The local loop selects one head.  Pass pointers already rebased to
        # that head so the microkernel only iterates its scalar dimension.
        # This is required both for compact head shards and for partial-state
        # components containing more than one local head.
        "query": head_pointer(query),
        "kv_cache": _pointer(state[0]),
        "slot_mapping": _pointer(state[3]),
        "block_table": _pointer(state[4]),
        "layer_id": _scalar_expression(layer_id),
        "partial_max": head_pointer(partial_max),
        "partial_sum": head_pointer(partial_sum),
        "partial_accumulator": head_pointer(partial_accumulator),
        "num_layers": cache_shape[1],
        "num_query_heads": query_shape[head_axis],
        "num_kv_heads": cache_shape[-2],
        "head_dim": query_head_dim,
        "block_size": cache_shape[3],
        "scale": repr(float(attrs["scale"])),
        "token_tile": int(raw["parameters"].get("token_tile", 0)),
        "context_shards": int(attrs["split_count"]),
        "local_capacity": max_domain["capacity"],
        "active": max_domain["active"],
        "context_shard": _mesh_coordinate(split_axis, _mesh_rank(max_abi)),
        "head": max_domain["logical_coordinates"][head_axis],
    }
    variant = str(raw.get("variant", ""))
    if variant in {"decode_t16", "decode_t32"}:
        return result
    if variant != "mma_tma_smem_pipeline":
        raise CodegenError(
            f"PagedAttentionPartial variant {variant!r} has no local-shard ABI."
        )

    parameters = raw.get("parameters", {})
    pipeline = raw.get("transfer_pipeline")
    workspaces = raw.get("shared_workspaces")
    if (
        not isinstance(parameters, Mapping)
        or not isinstance(pipeline, Mapping)
        or not isinstance(workspaces, (tuple, list))
        or len(workspaces) != 2
    ):
        raise CodegenError(
            "Paged-attention MMA requires a typed two-channel transfer pipeline."
        )
    block_m = int(parameters["block_m"])
    block_n = int(parameters["block_n"])
    block_k = int(parameters["block_k"])
    num_stages = int(parameters["num_stages"])
    consumer_warps = int(parameters["consumer_warps"])
    worker_width = int(parameters["worker_width"])
    query_lanes = tuple(int(value) for value in query_abi["scalar_lane_shape"])
    cache_lanes = tuple(int(value) for value in cache_abi["scalar_lane_shape"])
    local_query_shape = _static_shape(query_abi, "local_capacity_shape")
    cache_strides = tuple(
        int(value) for value in cache_abi["scalar_storage_strides"]
    )
    if (
        block_m != 1
        or block_k != query_head_dim
        or block_n <= 0
        or cache_shape[3] % block_n
        or num_stages <= 0
        or consumer_warps != 8
        or worker_width != 32
        or query_lanes != (8,)
        or cache_lanes != (8,)
        or str(query_abi["scalar_dtype"]) != "bfloat16"
        or str(cache_abi["scalar_dtype"]) != "bfloat16"
        or local_query_shape[layout.index("seq")] != 1
        or local_query_shape[head_axis] != 1
        or len(cache_strides) != 6
    ):
        raise CodegenError(
            "Paged-attention MMA requires one owner-local BF16x8 decode head, "
            "a complete K dimension, eight consumer warps, and a page-aligned "
            "attention tile."
        )
    descriptor_shape = (
        cache_shape[0],
        cache_shape[1],
        cache_shape[3],
        cache_shape[4],
        cache_head_dim,
    )
    descriptor_strides = (
        cache_strides[0],
        cache_strides[1],
        cache_strides[3],
        cache_strides[4],
        1,
    )
    descriptor_block_shape = (1, 1, block_n, 1, cache_head_dim)
    expected_shared_shape = (
        num_stages,
        *descriptor_block_shape,
    )
    if tuple(tuple(value["shape"]) for value in workspaces) != (
        expected_shared_shape,
        expected_shared_shape,
    ):
        raise CodegenError(
            "Paged-attention MMA Shared K/V stages disagree with selected tiles."
        )
    cache_offset_bytes = (
        int(cache_abi["pool_byte_offset"])
        if str(cache_abi.get("storage")) in {"rdata", "workspace"}
        else 0
    )
    value_offset_bytes = (
        cache_offset_bytes
        + cache_strides[2] * int(cache_abi["scalar_itemsize"])
    )
    descriptor_source = str(state[0]["runtime_argument"])
    result.update({
        "host_tensor_descriptor_requests": (
            {
                "parameter": "key_descriptor",
                "source": descriptor_source,
                "kind": "single",
                "offset_bytes": cache_offset_bytes,
                "dtype": str(cache_abi["scalar_dtype"]),
                "shape": descriptor_shape,
                "strides": descriptor_strides,
                "block_shape": descriptor_block_shape,
                "source_shape_axes": tuple(() for _ in descriptor_shape),
                "padding": "zero",
            },
            {
                "parameter": "value_descriptor",
                "source": descriptor_source,
                "kind": "single",
                "offset_bytes": value_offset_bytes,
                "dtype": str(cache_abi["scalar_dtype"]),
                "shape": descriptor_shape,
                "strides": descriptor_strides,
                "block_shape": descriptor_block_shape,
                "source_shape_axes": tuple(() for _ in descriptor_shape),
                "padding": "zero",
            },
        ),
        "key_descriptor": "key_descriptor",
        "value_descriptor": "value_descriptor",
        "descriptor_block_shape": descriptor_block_shape,
        "pipeline_contract": dict(pipeline),
        "shared_workspaces": tuple(workspaces),
        "pipeline_channel_field_names": {
            "key": ("key",),
            "value": ("value",),
        },
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "num_stages": num_stages,
        "q_head_group_size": query_shape[head_axis] // cache_shape[-2],
        "consumer_warps": consumer_warps,
        "worker_width": worker_width,
        "producer_warps": int(parameters["producer_warps"]),
        "producer_registers": int(parameters["producer_registers"]),
    })
    if result["q_head_group_size"] <= 0:
        raise CodegenError(
            "Paged-attention MMA requires at least one query head per KV head."
        )
    return result


def _paged_attention_combine_call(raw) -> dict[str, object]:
    partial_max = _buffer(raw, "inputs", "max_state")
    partial_sum = _buffer(raw, "inputs", "sum_state")
    partial_accumulator = _buffer(raw, "inputs", "acc_state")
    result = _buffer(raw, "outputs", "result")
    max_abi = partial_max["abi"]
    result_abi = result["abi"]
    attrs = raw.get("semantic_attrs", {})
    layout = tuple(str(value) for value in attrs["layout"])
    head_axis = layout.index("head")
    dim_axis = layout.index("dim")
    domain = _attention_scalar_domain(
        result_abi, layout, "attention_output_offsets"
    )
    shape = _static_shape(result_abi, "logical_shape")
    lane_count = int(result_abi.get("scalar_lane_count", 1))
    split_axis = int(attrs["split_hierarchy_axis"])
    partial_axis = _single_partial_hierarchy_axis(max_abi)
    if split_axis != partial_axis:
        raise CodegenError(
            "Paged-attention combine ABI disagrees with its split axis."
        )
    max_coordinates = _attention_partial_state_coordinates(
        max_abi, layout, domain, statistics=True
    )
    sum_coordinates = _attention_partial_state_coordinates(
        partial_sum["abi"], layout, domain, statistics=True
    )
    accumulator_coordinates = _attention_partial_state_coordinates(
        partial_accumulator["abi"], layout, domain, statistics=False
    )
    owner = _partial_group_owner_expression(max_abi, "attention_part")
    return {
        "partial_max": emit_storage_pointer(
            max_abi, str(partial_max["runtime_argument"])
        ),
        "partial_sum": emit_storage_pointer(
            partial_sum["abi"], str(partial_sum["runtime_argument"])
        ),
        "partial_accumulator": emit_storage_pointer(
            partial_accumulator["abi"],
            str(partial_accumulator["runtime_argument"]),
        ),
        "partial_owner": owner,
        "partial_max_owner_stride": int(
            max_abi["component_stride_scalar_elements"]
        ),
        "partial_sum_owner_stride": int(
            partial_sum["abi"]["component_stride_scalar_elements"]
        ),
        "partial_accumulator_owner_stride": int(
            partial_accumulator["abi"]["component_stride_scalar_elements"]
        ),
        "partial_max_offset": emit_local_scalar_offset(
            max_abi, max_coordinates
        ),
        "partial_sum_offset": emit_local_scalar_offset(
            partial_sum["abi"], sum_coordinates
        ),
        "partial_accumulator_offset": emit_local_scalar_offset(
            partial_accumulator["abi"], accumulator_coordinates
        ),
        "result": _pointer(result),
        "num_query_heads": shape[head_axis],
        "head_dim": shape[dim_axis] * lane_count,
        "context_shards": int(attrs["split_count"]),
        "local_capacity": domain["capacity"],
        "active": domain["active"],
        "writer_active": _canonical_writer_active(result_abi),
        "head": domain["global_by_kind"]["head"],
        "dimension": domain["global_by_kind"]["dim"],
        "result_offset": domain["offset"],
        "tile": int(raw["parameters"]["elements_per_program"]),
    }


def _attention_partial_state_coordinates(
    abi: Mapping[str, object],
    layout: tuple[str, ...],
    output_domain: Mapping[str, object],
    *,
    statistics: bool,
) -> tuple[str, ...]:
    """Remap a distributed output lane into its partial-state component.

    PagedAttentionCombine may further shard the result dimension while the
    online-softmax accumulator keeps that dimension broadcast within every
    partial owner.  In that case the state address needs the output's logical
    coordinate, not its owner-local coordinate.  Axes that remain sharded in
    the state (normally query-head) keep the local coordinate because the
    selected owner base has already been applied separately.
    """

    logical_shape = _static_shape(abi, "logical_shape")
    local_shape = _static_shape(abi, "local_capacity_shape")
    if len(layout) != len(logical_shape) or len(local_shape) != len(logical_shape):
        raise CodegenError(
            "Paged-attention partial-state rank differs from its semantic layout."
        )
    coordinates = []
    for axis, kind in enumerate(layout):
        if statistics and kind == "dim":
            coordinates.append("0")
        elif local_shape[axis] == logical_shape[axis]:
            coordinates.append(str(output_domain["global_by_kind"][kind]))
        else:
            coordinates.append(str(output_domain["local_by_kind"][kind]))
    return tuple(coordinates)


def _greedy_sample_call(raw) -> dict[str, object]:
    logits = _buffer(raw, "inputs", "logits")
    result = _buffer(raw, "outputs", "result")
    partial_max = _buffer(raw, "workspaces", "partial_max")
    partial_index = _buffer(raw, "workspaces", "partial_index")
    logits_abi = logits["abi"]
    local_shape = _static_shape(logits_abi, "local_capacity_shape")
    coordinates = ("batch_local", "vocab_offsets")
    output_domain = _local_domain(result["abi"], "result_batch")
    distributed = logits_abi.get("distributed_type")
    hierarchy = tuple(distributed["placement"]["hierarchy"]) if isinstance(distributed, Mapping) else ()
    owner_coordinates = tuple(
        f"((greedy_owners // {prod(hierarchy[axis + 1:])}) % {extent})"
        for axis, extent in enumerate(hierarchy)
    )
    return {
        "logits": _pointer(logits),
        "result": _pointer(result),
        "partial_max": _pointer(partial_max),
        "partial_index": _pointer(partial_index),
        "local_capacity": local_shape[-1],
        "batch_capacity": local_shape[0],
        "active": " & ".join(
            f"(({coordinate}) < ({emit_active_extent(logits_abi, axis)}))"
            for axis, coordinate in enumerate(coordinates)
        ),
        "logical_index": emit_logical_coordinate(logits_abi, 1, coordinates),
        "logits_offset": emit_local_scalar_offset(
            logits_abi, coordinates
        ),
        "partial_batch_logical": emit_logical_coordinate(
            logits_abi, 0, ("partial_batch", "0"), shard_coordinates=owner_coordinates or None,
        ),
        "result_capacity": output_domain["capacity"],
        "result_active": output_domain["active"],
        "result_logical_batch": output_domain["logical_coordinates"][0],
        "result_offset": emit_local_scalar_offset(result["abi"], output_domain["local_coordinates"]),
        "result_writer": _canonical_writer_active(result["abi"]),
        "vocab_size": _static_shape(logits_abi, "logical_shape")[-1],
        "tile": int(raw["parameters"]["vocab_tile"]),
    }


def _scalar_axis_extent(abi: Mapping[str, object], axis: int) -> int:
    shape = _static_shape(abi, "local_capacity_shape")
    index = axis if axis >= 0 else len(shape) + axis
    lane_count = int(abi.get("scalar_lane_count", 1))
    return shape[index] * lane_count


def _semantic_weight_block_shape(
    raw: Mapping[str, object],
    *,
    scales: tuple[Mapping[str, object], ...],
    global_n: int,
    global_k: int,
    owner: str,
) -> tuple[int, int]:
    """Read block-scale geometry from the semantic weight-format contract."""

    attrs = raw.get("semantic_attrs", {})
    if not isinstance(attrs, Mapping):
        raise CodegenError(f"{owner} semantic attributes must be a mapping.")
    values = []
    for name in ("weight_block_n", "weight_block_k"):
        value = attrs.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise CodegenError(f"{owner} requires a positive semantic {name}.")
        values.append(value)
    block_n, block_k = values
    expected = (
        (global_n + block_n - 1) // block_n,
        (global_k + block_k - 1) // block_k,
    )
    for scale in scales:
        actual = _static_shape(scale, "logical_shape")
        if actual != expected:
            raise CodegenError(
                f"{owner} block-scale tensor shape {actual} does not match "
                f"semantic groups {expected}."
            )
    return block_n, block_k


def _scalar_logical_axis_extent(abi: Mapping[str, object], axis: int) -> int:
    shape = _static_shape(abi, "logical_shape")
    index = axis if axis >= 0 else len(shape) + axis
    lane_count = int(abi.get("scalar_lane_count", 1))
    return shape[index] * lane_count


def _scalar_last_axis_domain(
    abi: Mapping[str, object],
    scalar_coordinate: str,
    *,
    owner: str,
) -> dict[str, object]:
    """Describe one scalarized row of a possibly-vector physical tensor."""

    shape = _static_shape(abi, "local_capacity_shape")
    if not shape or prod(shape[:-1], start=1) != 1:
        raise CodegenError(f"{owner} requires exactly one owner-local row.")
    lane_count = int(abi.get("scalar_lane_count", 1))
    if lane_count <= 0:
        raise CodegenError(f"{owner} has an invalid scalar lane count.")
    physical = (
        scalar_coordinate
        if lane_count == 1
        else f"(({scalar_coordinate}) // {lane_count})"
    )
    lane = (
        None
        if lane_count == 1
        else f"(({scalar_coordinate}) % {lane_count})"
    )
    coordinates = (*("0" for _ in shape[:-1]), physical)
    zero_coordinates = tuple("0" for _ in shape)
    distributed = abi.get("distributed_type")
    if isinstance(distributed, Mapping):
        placement = distributed.get("placement")
        hierarchy = (
            placement.get("hierarchy")
            if isinstance(placement, Mapping)
            else None
        )
        mesh_rank = len(hierarchy) if isinstance(hierarchy, (tuple, list)) else 2
    else:
        mesh_rank = 2
    zero_shards = tuple("0" for _ in range(mesh_rank))
    active_extents = tuple(
        emit_active_extent(abi, axis) for axis in range(len(shape))
    )
    active_terms = [
        f"(0 < ({active_extents[axis]}))"
        for axis in range(len(shape) - 1)
    ]
    active_terms.append(
        f"(({physical}) < ({active_extents[-1]}))"
    )
    logical_physical = emit_logical_coordinate(
        abi, len(shape) - 1, coordinates
    )
    global_scalar = (
        logical_physical
        if lane_count == 1
        else f"(({logical_physical}) * {lane_count} + ({lane}))"
    )
    return {
        "capacity": shape[-1] * lane_count,
        "active": " & ".join(active_terms),
        "owner_active": " & ".join(active_terms[:-1]) or "True",
        "physical_active_extent": active_extents[-1],
        "lane_count": lane_count,
        "offset": emit_local_scalar_offset(
            abi, coordinates, lane_coordinate=lane
        ),
        # Split an affine distributed access into an owner base, computed once
        # by the caller, and an owner-local offset suitable for a value-only
        # inner-stage ABI. Contiguous and block-cyclic SBP coordinate
        # expressions keep shard and local terms additive by contract.
        "owner_base_offset": emit_local_scalar_offset(
            abi,
            zero_coordinates,
            lane_coordinate="0" if lane_count != 1 else None,
        ),
        "local_offset": emit_local_scalar_offset(
            abi,
            coordinates,
            lane_coordinate=lane,
            shard_coordinates=zero_shards,
        ),
        "global_scalar": global_scalar,
    }


def _dense_weight_access(
    binding: Mapping[str, object],
    *,
    packed_layout: str | None,
    transpose_b: bool,
    local_n: str,
    local_k: str,
    global_n: str,
    global_k: str,
) -> tuple[str, str]:
    """Address a dense RHS in the coordinate space declared by its ABI.

    Canonical-global storage is indexed with logical N/K coordinates. A
    compact owner component is already boxed into a dense local tensor, so
    both the pointer and coordinates must remain owner-local. Mixing these
    conventions aliases K groups with N owners when both dimensions contain
    more than one physical group.
    """

    abi = binding["abi"]
    coordinate_space = str(abi.get("coordinate_space"))
    if coordinate_space == "canonical_global":
        pointer = emit_storage_pointer(abi, str(binding["runtime_argument"]))
        n_coordinate = global_n
        k_coordinate = global_k
        emit_offset = emit_global_scalar_offset
    elif coordinate_space == "local":
        pointer = _pointer(binding)
        n_coordinate = local_n
        k_coordinate = local_k
        emit_offset = emit_local_scalar_offset
    else:
        raise CodegenError(
            "TIR DenseMatMul RHS has an unsupported coordinate space "
            f"{coordinate_space!r}."
        )

    if packed_layout is None:
        coordinates = (
            (n_coordinate, k_coordinate)
            if transpose_b
            else (k_coordinate, n_coordinate)
        )
        return pointer, emit_offset(abi, coordinates)

    if packed_layout != "k_major_n8_k16":
        raise CodegenError(
            f"TIR DenseMatMul has unsupported packed layout {packed_layout!r}."
        )
    shape = _static_shape(abi, "logical_shape")
    lanes = tuple(int(value) for value in abi.get("scalar_lane_shape", ()))
    if lanes:
        if len(shape) != 2 or lanes != (8, 2, 8):
            raise CodegenError(
                "Typed k_major_n8_k16 RHS must be [K/16,N/8] of "
                "VectorType(N8,KPack2,KVector8)."
            )
        lane_coordinate = (
            f"(({n_coordinate}) % 8) * 16 + (({k_coordinate}) % 16)"
        )
        return pointer, emit_offset(
            abi,
            (f"({k_coordinate}) // 16", f"({n_coordinate}) // 8"),
            lane_coordinate=lane_coordinate,
        )

    if len(shape) != 4:
        raise CodegenError(
            "Scalar-physical k_major_n8_k16 RHS must have rank four."
        )
    payload_width = shape[3]
    payload = f"(({n_coordinate}) % 8) * 16 + (({k_coordinate}) % 16)"
    return pointer, emit_offset(
        abi,
        (
            f"({k_coordinate}) // 16",
            f"({n_coordinate}) // 8",
            f"({payload}) // {payload_width}",
            f"({payload}) % {payload_width}",
        ),
    )


def _verify_k_major_n8_k16_weight(
    abi: Mapping[str, object], global_n: int, global_k: int
) -> None:
    shape = _static_shape(abi, "logical_shape")
    lanes = tuple(int(value) for value in abi.get("scalar_lane_shape", ()))
    scalar_physical = (
        len(shape) == 4
        and shape[0] * 16 == global_k
        and shape[1] * 8 == global_n
        and shape[2] * shape[3] == 128
        and int(abi.get("scalar_lane_count", 1)) == 1
    )
    typed_vector = (
        len(shape) == 2
        and lanes == (8, 2, 8)
        and shape[0] * 16 == global_k
        and shape[1] * 8 == global_n
        and int(abi.get("scalar_lane_count", 1)) == 128
    )
    if not (scalar_physical or typed_vector):
        raise CodegenError(
            "TIR k_major_n8_k16 weight ABI disagrees with its logical N/K."
        )


def _bounded_vector_tile(value: object, capacity: int, *, name: str) -> int:
    """Fit a power-of-two Triton vector to a statically bounded local domain.

    Implementation parameters describe the largest vector the kernel may use.
    Materializing a larger vector than the entire local domain only increases
    register pressure; transcendental kernels can otherwise spill even when
    all but a handful of lanes are masked.
    """

    if isinstance(value, bool):
        raise CodegenError(f"{name} must be an integer positive power of two.")
    try:
        requested = int(value)
    except (TypeError, ValueError) as error:
        raise CodegenError(f"{name} must be an integer.") from error
    if requested <= 0 or requested & (requested - 1):
        raise CodegenError(f"{name} must be a positive power of two.")
    if capacity <= 0:
        raise CodegenError(f"{name} requires a positive local capacity.")
    capacity_tile = 1 << (int(capacity) - 1).bit_length()
    return min(requested, capacity_tile)


def _single_partial_hierarchy_axis(abi: Mapping[str, object]) -> int:
    distributed = abi.get("distributed_type")
    if not isinstance(distributed, Mapping):
        raise CodegenError("Partial-state buffer has no DistributedType ABI.")
    partial = distributed.get("partial")
    if not isinstance(partial, Mapping):
        raise CodegenError("Partial-state buffer has no partial SBP contract.")
    axes = tuple(int(value) for value in partial.get("axes", ()))
    if len(axes) != 1:
        raise CodegenError(
            "Paged attention requires exactly one partial hierarchy axis."
        )
    return axes[0]


def _mesh_coordinates(rank: int) -> tuple[str, ...]:
    if rank <= 0:
        raise CodegenError("Triton distributed ABI requires a positive mesh rank.")
    if rank == 2:
        return ("shard_y", "shard_x")
    return tuple(f"shard_coord{axis}" for axis in range(rank))


def _mesh_rank(abi: Mapping[str, object]) -> int:
    distributed = abi.get("distributed_type")
    if not isinstance(distributed, Mapping):
        raise CodegenError("Distributed buffer has no DistributedType ABI.")
    placement = distributed.get("placement")
    if not isinstance(placement, Mapping):
        raise CodegenError("Distributed buffer has no placement ABI.")
    hierarchy = placement.get("hierarchy")
    if not isinstance(hierarchy, (tuple, list)) or not hierarchy:
        raise CodegenError("Distributed buffer placement is empty.")
    return len(hierarchy)


def _mesh_coordinate(axis: int, rank: int) -> str:
    coordinates = _mesh_coordinates(rank)
    if axis < 0 or axis >= rank:
        raise CodegenError("Hierarchy axis is outside its placement.")
    return coordinates[axis]


def _partial_group_owner_expression(
    abi: Mapping[str, object], member: str
) -> str:
    distributed = abi["distributed_type"]
    assert isinstance(distributed, Mapping)
    placement = distributed.get("placement")
    if not isinstance(placement, Mapping):
        raise CodegenError("Partial-state buffer has no placement ABI.")
    hierarchy = tuple(int(value) for value in placement.get("hierarchy", ()))
    if not hierarchy:
        raise CodegenError("Partial-state buffer placement is empty.")
    axis = _single_partial_hierarchy_axis(abi)
    if axis >= len(hierarchy):
        raise CodegenError("Partial hierarchy axis is outside its placement.")
    coordinates = list(_mesh_coordinates(len(hierarchy)))
    coordinates[axis] = member
    if len(hierarchy) == 2:
        if axis == 0:
            return f"(({member}) * {hierarchy[1]} + (shard_x))"
        return f"(shard_y * {hierarchy[1]} + ({member}))"
    return _linear_owner_expression(coordinates, hierarchy)


def _partial_group_owner_expression_for_axes(
    abi: Mapping[str, object],
    axes: tuple[int, ...],
    member: str,
) -> str:
    distributed = abi.get("distributed_type")
    if not isinstance(distributed, Mapping):
        raise CodegenError("Partial buffer has no DistributedType ABI.")
    partial = distributed.get("partial")
    actual_axes = (
        ()
        if not isinstance(partial, Mapping)
        else tuple(int(value) for value in partial.get("axes", ()))
    )
    if actual_axes != axes:
        raise CodegenError(
            "Partial buffer axes disagree with the selected collective contract."
        )
    placement = distributed.get("placement")
    if not isinstance(placement, Mapping):
        raise CodegenError("Partial buffer has no placement ABI.")
    hierarchy = tuple(int(value) for value in placement.get("hierarchy", ()))
    if not hierarchy:
        raise CodegenError("Partial-group owner placement is empty.")
    if (
        not axes
        or tuple(sorted(set(axes))) != axes
        or any(axis < 0 or axis >= len(hierarchy) for axis in axes)
    ):
        raise CodegenError(
            "Partial-group owner axes must be sorted, unique, and inside the placement."
        )
    return _group_owner_expression_for_axes(abi, axes, member)


def _group_owner_expression_for_axes(
    abi: Mapping[str, object],
    axes: tuple[int, ...],
    member: str,
) -> str:
    """Map a dense group member to a placement owner, preserving other axes."""

    distributed = abi.get("distributed_type")
    if not isinstance(distributed, Mapping):
        raise CodegenError("Owner group buffer has no DistributedType ABI.")
    placement = distributed.get("placement")
    if not isinstance(placement, Mapping):
        raise CodegenError("Owner group buffer has no placement ABI.")
    hierarchy = tuple(int(value) for value in placement.get("hierarchy", ()))
    if (
        not axes
        or tuple(sorted(set(axes))) != axes
        or any(axis < 0 or axis >= len(hierarchy) for axis in axes)
    ):
        raise CodegenError(
            "Owner group axes must be sorted, unique, and inside the placement."
        )
    if axes == tuple(range(len(hierarchy))):
        return f"({member})"
    if len(hierarchy) == 2:
        if axes == (0,):
            return f"(({member}) * {hierarchy[1]} + shard_x)"
        if axes == (1,):
            return f"(shard_y * {hierarchy[1]} + ({member}))"
    coordinates = list(_mesh_coordinates(len(hierarchy)))
    remaining = f"({member})"
    for position, axis in enumerate(reversed(axes)):
        extent = hierarchy[axis]
        is_most_major = position == len(axes) - 1
        coordinates[axis] = (
            remaining if is_most_major else f"(({remaining}) % {extent})"
        )
        if not is_most_major:
            remaining = f"(({remaining}) // {extent})"
    return _linear_owner_expression(coordinates, hierarchy)


def _mesh_group_member_expression_for_axes(
    abi: Mapping[str, object], axes: tuple[int, ...]
) -> tuple[str, int]:
    """Return this CTA's dense ordinal within selected placement axes."""

    distributed = abi.get("distributed_type")
    placement = (
        None if not isinstance(distributed, Mapping) else distributed.get("placement")
    )
    if not isinstance(placement, Mapping):
        raise CodegenError("Mesh group member has no placement ABI.")
    hierarchy = tuple(int(value) for value in placement.get("hierarchy", ()))
    if (
        not axes
        or tuple(sorted(set(axes))) != axes
        or any(axis < 0 or axis >= len(hierarchy) for axis in axes)
    ):
        raise CodegenError(
            "Mesh group axes must be sorted, unique, and inside the placement."
        )
    count = prod((hierarchy[axis] for axis in axes), start=1)
    if axes == tuple(range(len(hierarchy))):
        return "(shard_index)", count
    coordinates = _mesh_coordinates(len(hierarchy))
    expression = f"({coordinates[axes[0]]})"
    for axis in axes[1:]:
        expression = f"({expression} * {hierarchy[axis]} + {coordinates[axis]})"
    return expression, count


def _linear_owner_expression(
    coordinates: list[str], hierarchy: tuple[int, ...]
) -> str:
    if len(coordinates) != len(hierarchy) or not hierarchy:
        raise CodegenError("Owner coordinates do not match the placement hierarchy.")
    if len(hierarchy) == 1:
        return f"({coordinates[0]})"
    expression = f"({coordinates[0]})"
    for coordinate, extent in zip(coordinates[1:], hierarchy[1:], strict=True):
        expression = f"({expression} * {extent} + ({coordinate}))"
    return expression


def _canonical_writer_active(abi: Mapping[str, object]) -> str:
    if str(abi.get("storage_kind")) != "canonical_global":
        return "True"
    return _distributed_unique_writer_active(abi)


def _distributed_unique_writer_active(
    abi: Mapping[str, object], *, writer_ordinal: int = 0,
    allow_redundant_axes: tuple[int, ...] = (),
) -> str:
    if writer_ordinal < 0:
        raise CodegenError("Distributed writer ordinal must be non-negative.")
    distributed = abi.get("distributed_type")
    if not isinstance(distributed, Mapping):
        return "True"
    placement = distributed.get("placement")
    if not isinstance(placement, Mapping):
        raise CodegenError("Canonical distributed buffer has no placement ABI.")
    hierarchy = tuple(int(value) for value in placement.get("hierarchy", ()))
    if not hierarchy:
        raise CodegenError("Triton canonical writer placement is empty.")
    used: set[int] = set()
    for policy in distributed.get("axis_policies", ()):
        if not isinstance(policy, Mapping) or policy.get("kind") != "split":
            continue
        stages = policy.get("stages")
        if not isinstance(stages, (tuple, list)) or not stages:
            raise CodegenError(
                "Serialized split policy requires a non-empty stages sequence."
            )
        for stage in stages:
            if not isinstance(stage, Mapping):
                raise CodegenError("Serialized split stage must be a mapping.")
            axes = stage.get("hierarchy_axes")
            if not isinstance(axes, (tuple, list)) or not axes:
                raise CodegenError(
                    "Serialized split stage requires hierarchy_axes."
                )
            for value in axes:
                axis = int(value)
                if axis < 0 or axis >= len(hierarchy):
                    raise CodegenError(
                        "Serialized split stage hierarchy axis is outside its placement."
                    )
                used.add(axis)
    allowed = set(int(axis) for axis in allow_redundant_axes)
    if any(axis < 0 or axis >= len(hierarchy) for axis in allowed) or allowed & used:
        raise CodegenError(
            "Allowed redundant writer axes must be unused placement axes."
        )
    unused_axes = [
        axis
        for axis in range(len(hierarchy))
        if axis not in used and axis not in allowed
    ]
    redundant_owner_count = prod(hierarchy[axis] for axis in unused_axes)
    ordinal = writer_ordinal % redundant_owner_count
    selected_coordinates: dict[int, int] = {}
    for axis in reversed(unused_axes):
        selected_coordinates[axis] = ordinal % hierarchy[axis]
        ordinal //= hierarchy[axis]
    conditions = [
        f"({_mesh_coordinate(axis, len(hierarchy))} == {selected_coordinates[axis]})"
        for axis in unused_axes
    ]
    return " & ".join(conditions) if conditions else "True"


def _distributed_unique_writer_index(
    abi: Mapping[str, object],
    *, allow_redundant_axes: tuple[int, ...] = (),
) -> tuple[str, int]:
    """Return the compact ordinal and count of unique distributed writers.

    Split hierarchy axes identify distinct logical shards.  Broadcast axes are
    redundant physical owners and are fixed by
    :func:`_distributed_unique_writer_active`.  Compacting the former axes
    gives collective workspaces one entry per value shard rather than one
    sparse entry per placement owner.  This is the Python equivalent of
    nncase's owner-local statistics layout and is independent of mesh rank or
    concrete extents.
    """

    distributed = abi.get("distributed_type")
    if not isinstance(distributed, Mapping):
        return "0", 1
    placement = distributed.get("placement")
    if not isinstance(placement, Mapping):
        raise CodegenError("Distributed writer index has no placement ABI.")
    hierarchy = tuple(int(value) for value in placement.get("hierarchy", ()))
    if not hierarchy:
        raise CodegenError("Distributed writer index placement is empty.")
    used: set[int] = set()
    for policy in distributed.get("axis_policies", ()):
        if not isinstance(policy, Mapping) or policy.get("kind") != "split":
            continue
        stages = policy.get("stages")
        if not isinstance(stages, (tuple, list)) or not stages:
            raise CodegenError(
                "Serialized split policy requires a non-empty stages sequence."
            )
        for stage in stages:
            if not isinstance(stage, Mapping):
                raise CodegenError("Serialized split stage must be a mapping.")
            axes = stage.get("hierarchy_axes")
            if not isinstance(axes, (tuple, list)) or not axes:
                raise CodegenError(
                    "Serialized split stage requires hierarchy_axes."
                )
            for value in axes:
                axis = int(value)
                if axis < 0 or axis >= len(hierarchy):
                    raise CodegenError(
                        "Serialized split stage hierarchy axis is outside its placement."
                    )
                used.add(axis)
    allowed = set(allow_redundant_axes)
    if any(axis < 0 or axis >= len(hierarchy) for axis in allowed) or allowed & used:
        raise CodegenError(
            "Allowed redundant writer axes must be unused placement axes."
        )
    ordered_axes = tuple(sorted(used | allowed))
    if not ordered_axes:
        return "0", 1
    coordinates = _mesh_coordinates(len(hierarchy))
    expression = f"({coordinates[ordered_axes[0]]})"
    count = hierarchy[ordered_axes[0]]
    for axis in ordered_axes[1:]:
        expression = f"({expression} * {hierarchy[axis]} + ({coordinates[axis]}))"
        count *= hierarchy[axis]
    return expression, count


def _scalar_expression(binding: Mapping[str, object]) -> str:
    if str(binding["abi"].get("storage")) != "scalar":
        raise CodegenError(
            f"TIR formal {binding.get('formal')!r} is not a scalar ABI value."
        )
    value = binding.get("runtime_argument")
    if not isinstance(value, str):
        raise CodegenError("TIR scalar binding has no runtime expression.")
    return value


def _triton_dtype(dtype: str) -> str:
    try:
        return {
            "bfloat16": "tl.bfloat16",
            "float16": "tl.float16",
            "float32": "tl.float32",
            "int8": "tl.int8",
            "int16": "tl.int16",
            "int32": "tl.int32",
            "int64": "tl.int64",
            "uint8": "tl.uint8",
            "uint16": "tl.uint16",
            "uint32": "tl.uint32",
            "uint64": "tl.uint64",
        }[dtype]
    except KeyError as error:
        raise CodegenError(
            f"Elementwise cast has unsupported Triton dtype {dtype!r}."
        ) from error


def _local_domain(
    abi: Mapping[str, object], flat_coordinate: str
) -> dict[str, object]:
    """Build one dense owner-local iteration domain from a buffer ABI.

    This is deliberately independent of the concrete SBP.  Split staging only
    changes active bounds and local-to-global address expressions; ordinary
    kernel loops always start at local coordinate zero.
    """

    shape = _static_shape(abi, "local_capacity_shape")
    local_coordinates = _unflattened_coordinates(shape, flat_coordinate)
    active = " & ".join(
        f"(({coordinate}) < ({emit_active_extent(abi, axis)}))"
        for axis, coordinate in enumerate(local_coordinates)
    ) or "True"
    logical_coordinates = tuple(
        emit_logical_coordinate(abi, axis, local_coordinates)
        for axis in range(len(shape))
    )
    return {
        "capacity": prod(shape),
        "local_coordinates": local_coordinates,
        "logical_coordinates": logical_coordinates,
        "active": active,
    }


def _scalar_local_domain(
    abi: Mapping[str, object], flat_coordinate: str
) -> dict[str, object]:
    """Build an owner-local domain whose induction variable counts scalars.

    ``local_capacity_shape`` describes physical vector elements.  Kernels that
    perform scalar arithmetic must additionally iterate the element-type lanes
    and pass the lane coordinate to physical addressing.
    """

    shape = _static_shape(abi, "local_capacity_shape")
    lane_count = int(abi.get("scalar_lane_count", 1))
    if lane_count <= 0:
        raise CodegenError("A scalar local domain requires positive vector lanes.")
    physical_flat = (
        flat_coordinate
        if lane_count == 1
        else f"(({flat_coordinate}) // {lane_count})"
    )
    lane_coordinate = (
        None
        if lane_count == 1
        else f"(({flat_coordinate}) % {lane_count})"
    )
    local_coordinates = _unflattened_coordinates(shape, physical_flat)
    active = " & ".join(
        f"(({coordinate}) < ({emit_active_extent(abi, axis)}))"
        for axis, coordinate in enumerate(local_coordinates)
    ) or "True"
    logical_coordinates = tuple(
        emit_logical_coordinate(abi, axis, local_coordinates)
        for axis in range(len(shape))
    )
    return {
        "capacity": prod(shape) * lane_count,
        "local_coordinates": local_coordinates,
        "logical_coordinates": logical_coordinates,
        "lane_coordinate": lane_coordinate,
        "active": active,
    }


def _same_local_mapping(
    lhs: Mapping[str, object], rhs: Mapping[str, object]
) -> bool:
    return (
        tuple(lhs["local_capacity_shape"])
        == tuple(rhs["local_capacity_shape"])
        and tuple(lhs["logical_coordinate_expressions"])
        == tuple(rhs["logical_coordinate_expressions"])
    )


def _same_suffix_local_mapping(
    operand: Mapping[str, object], result: Mapping[str, object]
) -> bool:
    """Return whether a broadcast suffix operand shares the result mapping."""

    operand_shape = tuple(operand["local_capacity_shape"])
    result_shape = tuple(result["local_capacity_shape"])
    rank = len(operand_shape)
    if rank > len(result_shape) or operand_shape != result_shape[-rank:]:
        return False
    operand_mapping = tuple(
        re.sub(r"local_coord_\d+", "local_coord", str(value))
        for value in operand["logical_coordinate_expressions"]
    )
    result_mapping = tuple(
        re.sub(r"local_coord_\d+", "local_coord", str(value))
        for value in result["logical_coordinate_expressions"][-rank:]
    )
    return operand_mapping == result_mapping


def _same_axis_mapping(
    lhs: Mapping[str, object],
    rhs: Mapping[str, object],
    lhs_axes,
    rhs_axes,
) -> bool:
    lhs_values = tuple(lhs["logical_coordinate_expressions"])
    rhs_values = tuple(rhs["logical_coordinate_expressions"])
    lhs_mapping = tuple(
        re.sub(r"local_coord_\d+", "local_coord", str(lhs_values[index]))
        for index in lhs_axes
    )
    rhs_mapping = tuple(
        re.sub(r"local_coord_\d+", "local_coord", str(rhs_values[index]))
        for index in rhs_axes
    )
    return lhs_mapping == rhs_mapping


def _contiguous_local_offset_in_domain(
    operand_abi: Mapping[str, object],
    domain: Mapping[str, object],
    *,
    lane_coordinate: str | None,
) -> str:
    """Address a compact operand using another layout's logical coordinates.

    This is the regular inverse of ``LocalShardDescriptor.map_local_to_global``
    for contiguous shard regions.  It is used by gather/reduce/scatter kernels
    whose work assignment follows the destination, while each partial owner
    still stores a dense compact source component.  Block-cyclic or otherwise
    non-contiguous layouts require a distinct implementation and are rejected
    rather than being treated as dense by accident.
    """

    distributed_data = operand_abi.get("distributed_type")
    if not isinstance(distributed_data, Mapping):
        raise CodegenError(
            "A compact cross-layout access requires a DistributedType ABI."
        )
    operand_type = type_from_data(distributed_data)
    if not isinstance(operand_type, DistributedType):
        raise CodegenError(
            "A compact cross-layout access must describe a distributed tensor."
        )
    descriptor = local_shard_descriptor(
        operand_type, _mesh_coordinates(operand_type.placement.rank)
    )
    region = descriptor.contiguous_region
    if region is None:
        raise CodegenError(
            "Cross-layout partial access requires contiguous source shards; "
            "select a block-cyclic implementation for this distribution."
        )
    logical_coordinates = tuple(domain["logical_coordinates"])
    if len(logical_coordinates) != len(region.offset):
        raise CodegenError(
            "Cross-layout partial access rank differs from its work domain."
        )
    local_coordinates = tuple(
        f"(({logical}) - ({emit_dimension(offset)}))"
        for logical, offset in zip(
            logical_coordinates, region.offset, strict=True
        )
    )
    return emit_local_scalar_offset(
        operand_abi,
        local_coordinates,
        lane_coordinate=lane_coordinate,
    )


def _access_in_result_domain(
    operand_abi: Mapping[str, object],
    result_abi: Mapping[str, object],
    domain: Mapping[str, object],
    *,
    lane_coordinate: str | None = None,
) -> str:
    """Address an elementwise operand from the result's local coordinates."""

    if str(operand_abi["coordinate_space"]) == "canonical_global":
        return emit_global_scalar_offset(
            operand_abi,
            domain["logical_coordinates"],
            lane_coordinate=lane_coordinate,
        )
    if not _same_local_mapping(operand_abi, result_abi):
        raise CodegenError(
            "Elementwise local operands use different owner mappings; insert "
            "an explicit Boxing before code generation."
        )
    return emit_local_scalar_offset(
        operand_abi,
        domain["local_coordinates"],
        lane_coordinate=lane_coordinate,
    )

_FAMILY_ENCODERS = {
    "embedding": _embedding_call,
    "rms_norm": _rms_norm_call,
    "block_fp8": _block_fp8_call,
    "distributed_boxing": _boxing_call,
    "gdn_convolution": _gdn_convolution_call,
    "gdn_recurrent": _gdn_recurrent_call,
    "add_norm_stats": _add_norm_stats_call,
    "gather_reduce_add_norm_stats": _gather_reduce_add_norm_stats_call,
    "gather_reduce_add_norm_apply": _gather_reduce_add_norm_apply_call,
    "gather_reduce_norm_apply": _gather_reduce_norm_apply_call,
    "norm_apply": _norm_apply_call,
    "matmul_glu": _matmul_glu_call,
    "elementwise": _elementwise_call,
    "norm_stats": _norm_stats_call,
    "dense_matmul": _dense_matmul_call,
    "dense_matmul_glu": _dense_matmul_glu_call,
    "qkv_parallel_linear": _qkv_parallel_linear_call,
    "qkv_rope_with_cache": _qkv_rope_with_cache_call,
    "gather_reduce_qkv_rope_with_cache": _gather_reduce_qkv_rope_with_cache_call,
    "rotary_embedding": _rotary_embedding_call,
    "rope": _rope_call,
    "update_paged_attention_kv_cache": _cache_update_call,
    "paged_attention_partial": _paged_attention_partial_call,
    "paged_attention_combine": _paged_attention_combine_call,
    "greedy_sample": _greedy_sample_call,
}


__all__ = ["prepare_kernel_calls"]

# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Family/variant call encoders for bufferized Triton PrimFunctions."""

from __future__ import annotations

import itertools
import keyword
import re
from collections.abc import Mapping
from math import prod

from triton.flagmega.codegen.triton.broadcast_renderer import broadcast_to_call
from triton.flagmega.codegen.triton.concat_renderer import concat_call
from triton.flagmega.codegen.triton.delta_rule_block_renderer import delta_rule_block_update_call
from triton.flagmega.codegen.triton.delta_rule_decay_renderer import delta_rule_log_prefix_call
from triton.flagmega.codegen.triton.delta_rule_gates_renderer import delta_rule_gates_call
from triton.flagmega.codegen.triton.delta_rule_renderer import delta_rule_coefficients_call
from triton.flagmega.codegen.triton.descriptor_abi import device_descriptor_request
from triton.flagmega.codegen.triton.dimension_expression import emit_dimension
from triton.flagmega.codegen.triton.grouped_coordinates import tiles_stay_within_groups
from triton.flagmega.codegen.triton.kda_renderer import (
    kda_convolution_call,
    kda_gated_recurrent_call,
    kda_recurrent_call,
)
from triton.flagmega.codegen.triton.l2_normalization_renderer import l2_normalization_call
from triton.flagmega.codegen.triton.mhc_renderer import mhc_post_call, mhc_pre_call
from triton.flagmega.codegen.triton.mla_renderer import mla_projection_call
from triton.flagmega.codegen.triton.physical_access import (
    emit_active_extent,
    emit_buffer_pointer,
    emit_global_scalar_offset,
    emit_local_scalar_offset,
    emit_logical_coordinate,
    emit_owner_base,
    emit_remote_storage_pointer,
    emit_storage_pointer,
)
from triton.flagmega.codegen.triton.reduce_sum_renderer import reduce_sum_call
from triton.flagmega.codegen.triton.routing_renderer import routing_call
from triton.flagmega.codegen.triton.softmax_renderer import softmax_call
from triton.flagmega.codegen.triton.sparse_experts.down import sparse_experts_down_call
from triton.flagmega.codegen.triton.sparse_experts.gate_up import sparse_experts_gate_up_call
from triton.flagmega.codegen.triton.sparse_experts.routes import sparse_experts_routes_call
from triton.flagmega.codegen.triton.tensor_descriptor_planner import (
    packed_distributed_tensor_map_table_request,
    packed_owner_prefix_tensor_map_table_request,
)
from triton.flagmega.codegen.triton.tensor_transform_renderers import (
    gather_call,
    tensor_transform_call,
    vector_relayout_call,
)
from triton.flagmega.codegen.triton.top_k_renderer import top_k_call
from triton.flagmega.errors import CodegenError, IRSchemaError
from triton.flagmega.ir import (
    DistributedType,
    block_owner_axes,
    boxing_crossing_chip_axes,
    is_fully_replicated,
    is_local_shard_subview,
    local_shard_descriptor,
    type_from_data,
)
from triton.flagmega.ir.ops.math._block_scaled import (
    block_scaled_mma_supports_precision,
    normalize_block_scaled_precision,
    normalize_glu_activation_quantization,
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
        call["participation_active"] = _exclusive_participation_active(raw)
        from triton.flagmega.codegen.triton.fusion import decode_fusion_attrs, require_fusion
        from triton.flagmega.ir.op_fusion import has_ops
        if has_ops(raw.get("semantic_attrs", {})):
            from triton.flagmega.codegen.triton.tensor_transform_renderers import _tensor_type
            from triton.flagmega.ir import Node, get_definition
            attrs = decode_fusion_attrs(raw["semantic_attrs"])
            inputs = tuple(Node(f"arg{i}", "builtin.var", (), _tensor_type(_buffer(raw, "inputs", parameter.name)["abi"]))
                           for i, parameter in enumerate(get_definition(raw["semantic_op"]).input_parameters))
            output_type = _tensor_type(_buffer(raw, "outputs", "result")["abi"])
            require_fusion(Node("call", raw["semantic_op"], tuple(value.id for value in inputs), output_type,
                                attrs=attrs), inputs, family=family)
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


def _static_pipe_commits(call) -> int:
    """Return commits per invocation when the stage's tile grid is static.

    A counted loop can reuse one pipe instance only if it can advance the
    stage's iteration base by exactly the number of commits a trip issues, so
    a stage whose stream length is decided at run time (paged attention closes
    at a dynamic `sequence`) must report 0 rather than a wrong guess.
    """

    def positive(key):
        value = call.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        return None

    explicit_n, explicit_k = positive("num_n_tiles"), positive("num_k_tiles")
    if explicit_n or explicit_k:
        return (explicit_n or 1) * (explicit_k or 1)

    # Tile-descriptor stages walk `tl.range(0, capacity, step)` on each axis,
    # so the commit count is the product of the per-axis trip counts.  Both
    # axis step names appear across templates (`tile_n`/`block_n` and
    # `tile_k`/`block_k`), and an axis is only counted when both its capacity
    # and its step are statically known.
    tiles, known = 1, False
    for capacity_key, step_keys in (
        ("local_n_capacity", ("tile_n", "block_n")),
        ("local_k_capacity", ("tile_k", "block_k")),
    ):
        capacity = positive(capacity_key)
        step = next((positive(key) for key in step_keys if positive(key)), None)
        if capacity is None or step is None:
            continue
        tiles *= -(-capacity // step)
        known = True
    return tiles if known else 0


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
    if not isinstance(pipeline, Mapping) or not isinstance(workspaces, (tuple, list)):
        raise CodegenError("A transfer-pipelined TIR call requires typed Shared workspace ABI.")
    channels = pipeline.get("channels")
    consumer_indices = pipeline.get("consumer_shared_workspace_indices", ())
    if not isinstance(channels, (tuple, list)) or not channels:
        raise CodegenError("A transfer pipeline must expose at least one channel.")
    if not isinstance(consumer_indices, (tuple, list)):
        raise CodegenError("Transfer-pipeline consumer Shared workspace indices must be a sequence.")
    parameters = raw.get("parameters", {})
    for name in ("num_stages", "producer_warps", "producer_registers"):
        value = parameters.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise CodegenError(f"Transfer-pipeline parameter {name!r} must be a positive integer.")
        call[name] = value

    def workspace(index: object, owner: str) -> Mapping[str, object]:
        if isinstance(index, bool) or not isinstance(index, int):
            raise CodegenError(f"{owner} workspace index must be an integer.")
        try:
            value = workspaces[index]
        except IndexError as error:
            raise CodegenError(f"{owner} workspace index {index} is outside the Shared ABI.") from error
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
        raise CodegenError("pipeline_channel_field_names must map channel names to field names.")
    for channel in channels:
        if not isinstance(channel, Mapping):
            raise CodegenError("Transfer-pipeline channel must be a mapping.")
        name = str(channel.get("name", ""))
        _require_pipeline_identifier(name, "Transfer-pipeline channel")
        indices = channel.get("shared_workspace_indices")
        if not isinstance(indices, (tuple, list)) or not indices:
            raise CodegenError(f"Transfer-pipeline channel {name!r} owns no Shared workspace.")
        owned = tuple(workspace(index, f"Channel {name!r}") for index in indices)
        raw_fields = field_overrides.get(name)
        if raw_fields is None:
            field_names = ((name, ) if len(owned) == 1 else tuple(str(value["name"]) for value in owned))
        elif isinstance(raw_fields, (tuple, list)):
            field_names = tuple(str(value) for value in raw_fields)
        else:
            raise CodegenError(f"Transfer-pipeline channel {name!r} field override must be a sequence.")
        if len(field_names) != len(owned):
            raise CodegenError(f"Transfer-pipeline channel {name!r} has {len(owned)} workspaces "
                               f"but {len(field_names)} pipe field names.")
        for field_name in field_names:
            _require_pipeline_identifier(field_name, f"Transfer-pipeline channel {name!r} field")
        if len(set(field_names)) != len(field_names):
            raise CodegenError(f"Transfer-pipeline channel {name!r} has duplicate pipe fields.")
        reader = f"pipeline_{name}_reader"
        writer = f"pipeline_{name}_writer"
        encoded_channels.append({
            "name":
            name,
            "reader_parameter":
            reader,
            "writer_parameter":
            writer,
            "workspace_names":
            tuple(str(value["name"]) for value in owned),
            "fields":
            tuple({
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
            raise CodegenError("Transfer-pipeline auxiliary consumer must be a mapping.")
        channel_indices = auxiliary.get("channel_indices")
        workspace_indices = auxiliary.get("consumer_shared_workspace_indices", ())
        if (not isinstance(channel_indices, (tuple, list)) or not channel_indices
                or not isinstance(workspace_indices, (tuple, list))):
            raise CodegenError("Auxiliary consumer channel/workspace indices must be sequences.")
        try:
            selected_channels = tuple(encoded_channels[int(index)] for index in channel_indices)
        except (IndexError, TypeError, ValueError) as error:
            raise CodegenError("Auxiliary consumer references an invalid transfer channel.") from error
        consumer_by_name = {value["name"]: value for value in encoded_consumer_workspaces}
        selected_workspaces = []
        for index in workspace_indices:
            value = workspace(index, "Auxiliary consumer")
            try:
                selected_workspaces.append(consumer_by_name[value["name"]])
            except KeyError as error:
                raise CodegenError("Auxiliary consumer references a Shared workspace not "
                                   "owned by the primary consumer.") from error
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
            "consumer_shared_workspace_indices": tuple(int(value) for value in workspace_indices),
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
    # A counted loop reuses one pipe instance for every trip, so it must know
    # how many iterations a single invocation consumes to keep the iteration
    # stream gap-free across trips.  Only the renderer knows the tile counts
    # that decide it, so publish the product on the encoded call.
    # Only a stage whose tile counts are known publishes a commit count. A
    # stage whose stream length is decided at run time (paged attention closes
    # at a dynamic `sequence`) must report nothing rather than a wrong guess:
    # a counted loop can only reuse one pipe instance when it can advance the
    # iteration base by exactly the number of commits each trip issues.
    call["pipeline_commits_per_invocation"] = _static_pipe_commits(call)
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
                for dependency in binding.get("address_arguments", ()):
                    if dependency not in arguments:
                        arguments.append(dependency)
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
    if len(local_shape) != 2:
        raise CodegenError(
            "TIR embedding requires a rank-2 (row, feature) local result."
        )
    # M==1 is decode; M>1 (prefill) loops the leading token rows, each loading
    # its own index and copying the matching vocabulary row.
    local_m = local_shape[0]
    row_coordinate = "embedding_row" if local_m != 1 else "0"
    lanes = int(result_abi.get("scalar_lane_count", 1))
    if (tuple(weight["abi"].get("scalar_lane_shape", ())) != tuple(result_abi.get("scalar_lane_shape", ()))):
        raise CodegenError("Embedding weight and output must have identical vector element lanes.")
    local_feature = "local_offsets" if lanes == 1 else f"(local_offsets // {lanes})"
    lane_coordinate = None if lanes == 1 else f"(local_offsets % {lanes})"
    global_feature = emit_logical_coordinate(
        result_abi, 1, (row_coordinate, local_feature)
    )
    weight_abi = weight["abi"]
    weight_space = str(weight_abi["coordinate_space"])
    if weight_space == "canonical_global":
        # The table spans the whole vocabulary, so the row address is global.
        weight_offset = emit_global_scalar_offset(
            weight_abi, ("token_id", global_feature), lane_coordinate=lane_coordinate,
        )
    elif weight_space in {"local", "parent_shard_local"}:
        # A chip-compact table still holds every vocabulary row: the
        # distributed provider keeps that axis broadcast because a token id is a
        # runtime value no single owner can answer for.  Only the feature axis
        # is split, and it is projected exactly onto the output's feature
        # policy, so this owner reads the features it also writes and the local
        # feature coordinate addresses them directly.
        weight_offset = emit_local_scalar_offset(
            weight_abi, ("token_id", local_feature), lane_coordinate=lane_coordinate,
        )
    else:
        raise CodegenError(
            "Embedding lookup requires a canonical or chip-local vocabulary "
            f"table, got coordinate space {weight_space!r}."
        )
    indices_abi = indices["abi"]
    padding_idx = raw.get("semantic_attrs", {}).get("padding_idx")
    encoding = raw.get("semantic_attrs", {}).get("weight_encoding")
    if encoding not in (None, "IQ4_XS", "Q4_K"):
        raise CodegenError(f"Unsupported native GGUF embedding encoding {encoding!r}.")
    hidden = _static_shape(weight_abi, "logical_shape")[1]
    if encoding is not None:
        block_bytes = 136 if encoding == "IQ4_XS" else 144
        if hidden % block_bytes:
            raise CodegenError("Native GGUF embedding has an incomplete packed row.")
        hidden = hidden // block_bytes * 256
    if encoding is not None and hidden % 256:
        raise CodegenError("Native GGUF quantized embedding requires hidden size divisible by 256.")
    return {
        "indices": _pointer(indices),
        "weight": _pointer(weight),
        "result": _pointer(result),
        "local_m": local_m,
        "indices_offset": emit_local_scalar_offset(indices_abi, (row_coordinate,)),
        "local_capacity": local_shape[-1] * lanes,
        "active": f"({emit_active_extent(result_abi, 1)}) * {lanes}",
        "weight_offset": weight_offset,
        "global_feature": global_feature,
        "result_offset": emit_local_scalar_offset(
            result_abi, (row_coordinate, local_feature), lane_coordinate=lane_coordinate,
        ),
        "tile":
        int(raw["parameters"]["elements_per_program"]),
        "vocab_size":
        _static_shape(weight_abi, "logical_shape")[0],
        "padding_idx":
        -1 if padding_idx is None else int(padding_idx),
        "weight_encoding":
        encoding,
        "weight_block_bytes":
        136 if encoding == "IQ4_XS" else 144 if encoding == "Q4_K" else 0,
        "weight_hidden":
        hidden,
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
    lanes = tuple(raw["semantic_attrs"].get("output_lanes", ()))
    if (tuple(result_abi.get("scalar_lane_shape", ())) != lanes
            or int(result_abi.get("scalar_lane_count", 1)) != prod(lanes)
            or result_abi["scalar_dtype"] != value_abi["scalar_dtype"]):
        raise CodegenError("TIR block-FP8 output ABI disagrees with output_lanes or activation dtype.")
    value_shape = _static_shape(value_abi, "logical_shape")
    result_local = _static_shape(result_abi, "local_capacity_shape")
    if len(value_shape) != 2 or len(result_local) != 2:
        raise CodegenError("TIR block-FP8 renderer requires rank-two GEMV buffers.")
    local_n = "local_n_offsets"
    output_domain = _scalar_last_axis_domain(result_abi, local_n, owner="TIR block-FP8 output")
    value_domain = _scalar_last_axis_domain(value_abi, "0", owner="TIR block-FP8 activation")
    global_n = output_domain["global_scalar"]
    global_k_size = _scalar_logical_axis_extent(value_abi, -1)
    global_n_size = _scalar_logical_axis_extent(result_abi, -1)
    block_n, block_k = _semantic_weight_block_shape(
        raw,
        scales=(scale_abi, ),
        global_n=global_n_size,
        global_k=global_k_size,
        owner="TIR block-FP8",
    )
    encoding = raw.get("semantic_attrs", {}).get("weight_encoding")
    if encoding not in (None, "IQ4_XS", "Q4_K"):
        raise CodegenError(f"Unsupported native GGUF weight encoding {encoding!r}.")
    try:
        precision = normalize_block_scaled_precision(raw["semantic_attrs"], "TIR block-FP8")
    except IRSchemaError as error:
        raise CodegenError(str(error)) from error
    if precision and (value_abi["scalar_dtype"] not in {"float16", "bfloat16", "float32"}
                      or weight["abi"]["scalar_dtype"] != "float8_e4m3fn"
                      or scale_abi["scalar_dtype"] != "float32"
                      or int(value_abi.get("scalar_lane_count", 1)) != 1
                      or int(scale_abi.get("scalar_lane_count", 1)) != 1):
        raise CodegenError("TIR block-FP8 precision overrides require scalar F16/BF16/F32 activations, "
                           "FP8 weights and scalar FP32 scales.")
    if precision.get("activation_quantization") == "dynamic_fp8_sm90" and value_abi["scalar_dtype"] != "bfloat16":
        raise CodegenError("SM90 reciprocal activation quantization requires scalar BF16 input.")
    if raw.get("variant") == "mma" and not block_scaled_mma_supports_precision(
            precision, value_abi["scalar_dtype"], block_k=block_k):
        raise CodegenError("TIR block-FP8 MMA does not support the declared computation precision.")
    dequant_mma_splits = raw["parameters"].get("dequant_mma_splits", 1)
    if (type(dequant_mma_splits) is not int or dequant_mma_splits <= 0
            or dequant_mma_splits & (dequant_mma_splits - 1)):
        raise CodegenError("TIR block-FP8 dequant_mma_splits must be a positive power of two.")
    if dequant_mma_splits != 1 and (
            raw.get("variant") != "mma" or precision.get("activation_quantization") != "none"
            or value_domain["capacity"] % (dequant_mma_splits * block_k)):
        raise CodegenError("TIR block-FP8 dequant_mma_splits requires complete dequantized MMA scale blocks.")
    if encoding is None:
        weight_row = emit_local_scalar_offset(weight["abi"], (local_n, "0"))
    else:
        bytes_per_block = 136 if encoding == "IQ4_XS" else 144
        if global_k_size % 256:
            raise CodegenError("Native GGUF quantized GEMV requires K divisible by 256.")
        weight_row = f"({global_n}) * ({global_k_size} // 256) * {bytes_per_block}"
    scale_row = emit_global_scalar_offset(
        scale_abi,
        (f"({global_n}) // {block_n}", f"({value_domain['global_scalar']}) // {block_k}"),
    )
    return {
        "value": f"({_pointer(value)} + ({value_domain['owner_base_offset']}))",
        "weight": _pointer(weight),
        "scale": _pointer(scale),
        "result": _pointer(result),
        "local_n_capacity": output_domain["capacity"],
        "active_n": emit_active_extent(result_abi, 1),
        "n_mask": output_domain["active"],
        "global_n": global_n,
        "weight_row_offset": weight_row,
        "scale_row_offset": scale_row,
        "destination_offset": output_domain["offset"],
        "k": value_domain["capacity"],
        "block_k": block_k,
        "tile_n": int(raw["parameters"]["tile_n"]),
        "weight_encoding": encoding,
        "dequant_mma_splits": dequant_mma_splits,
        "scaled_accumulation": _scaled_accumulation_mode(
            raw, precision.get("activation_quantization", "dynamic_fp8"), block_k
        ),
        "activation_quantization": precision.get("activation_quantization", "dynamic_fp8"),
        "weight_dequant_type": _triton_dtype(precision["weight_dequant_dtype"])
        if precision.get("weight_dequant_dtype") is not None else "None",
    }


def _scaled_accumulation_mode(raw, activation_quantization, block_k):
    """Validate a selected promotion schedule independently of FP8 storage.

    The FMA schedule combines both scales before promoting each raw MMA tile.
    Dequantized BF16 GEMM has no scaled promotion and keeps its own accumulator.
    """
    mode = raw["parameters"].get("scaled_accumulation", "compensated")
    if not isinstance(mode, str) or mode not in {"compensated", "fma"}:
        raise CodegenError("TIR scaled_accumulation must be compensated or fma.")
    if mode == "fma" and (
        raw.get("variant") != "mma"
        or activation_quantization not in {"none", "dynamic_fp8_sm90"}
        or activation_quantization == "dynamic_fp8_sm90" and block_k < 32
    ):
        raise CodegenError("TIR scaled FMA requires SM90 MMA quantization and a complete MMA tile.")
    return mode


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
    gate_encoding = raw.get("semantic_attrs", {}).get("gate_weight_encoding")
    up_encoding = raw.get("semantic_attrs", {}).get("up_weight_encoding")
    if gate_encoding != up_encoding:
        raise CodegenError("Native GGUF MatMulGlu requires matching gate/up encodings.")
    if gate_encoding not in (None, "IQ4_XS", "Q4_K"):
        raise CodegenError(f"Unsupported native GGUF MatMulGlu encoding {gate_encoding!r}.")
    try:
        precision = normalize_glu_activation_quantization(raw.get("semantic_attrs", {}), "TIR MatMulGlu")
    except IRSchemaError as error:
        raise CodegenError(str(error)) from error
    if precision and (value["abi"]["scalar_dtype"] != "bfloat16"
                      or int(value["abi"].get("scalar_lane_count", 1)) != 1
                      or any(weight["abi"]["scalar_dtype"] != "float8_e4m3fn"
                             for weight in (gate_weight, up_weight))
                      or any(scale["abi"]["scalar_dtype"] != "float32"
                             or int(scale["abi"].get("scalar_lane_count", 1)) != 1
                             for scale in (gate_scale, up_scale))):
        raise CodegenError("SM90 reciprocal MatMulGlu requires scalar BF16 activations, FP8 weights and FP32 scales.")
    if gate_encoding is None:
        weight_row_offset = emit_local_scalar_offset(gate_weight["abi"], (local_n, "0"))
        up_weight_row_offset = emit_local_scalar_offset(up_weight["abi"], (local_n, "0"))
    else:
        bytes_per_block = 136 if gate_encoding == "IQ4_XS" else 144
        if global_k_size % 256:
            raise CodegenError("Native GGUF quantized MatMulGlu requires K divisible by 256.")
        weight_row_offset = f"({global_n}) * ({global_k_size} // 256) * {bytes_per_block}"
        up_weight_row_offset = weight_row_offset
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
        "weight_row_offset": weight_row_offset,
        "up_weight_row_offset": up_weight_row_offset,
        "gate_clamp_max": repr(raw.get("semantic_attrs", {}).get("gate_clamp_max")),
        "up_clamp": repr(raw.get("semantic_attrs", {}).get("up_clamp")),
        "scaled_accumulation": _scaled_accumulation_mode(
            raw, precision.get("activation_quantization", "dynamic_fp8"), block_k
        ),
        "activation_quantization": precision.get("activation_quantization", "dynamic_fp8"),
        "scale_row_offset": emit_global_scalar_offset(
            gate_scale["abi"],
            (f"({global_n}) // {block_n}", "0"),
        ),
        "destination_offset": emit_local_scalar_offset(result_abi, ("0", local_n)),
        "k": global_k_size,
        "block_k": block_k,
        "tile_n": int(raw["parameters"]["tile_n"]),
        "weight_encoding": gate_encoding,
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
    crossing = _boxing_leaf_crossing_chip_axes(source_abi, result_abi)
    if crossing:
        # A source-domain scatter only visits this chip's own elements. Each
        # private destination instance needs the complete destination region,
        # so peer reads must invert the source map from destination coordinates.
        mode = "canonical_copy" if result_space == "canonical_global" else "gather"
        iteration_abi = result_abi
        logical_abi = result_abi
    elif source_space == "local" and result_space == "canonical_global":
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
    elif _boxing_is_local_narrowing(source_abi, result_abi):
        mode = "local_gather"
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
    remote = (
        _cross_chip_boxing_source(
            source_abi, crossing, mode=mode,
            logical_coordinates=logical_coordinates,
            lane_coordinate=lane_coordinate,
        )
        if crossing else {}
    )
    if crossing:
        remote["remote_local_base"] = emit_storage_pointer(
            source_abi, str(source["runtime_argument"])
        )
        source_offset = remote["remote_source_offset"]
    elif mode == "local_gather":
        source_offset = _owner_local_operand_offset(source_abi, logical_coordinates, lane_coordinate)
    elif mode in {"scatter", "local_copy"}:
        source_offset = emit_local_scalar_offset(
            source_abi, local_coordinates, lane_coordinate=lane_coordinate
        )
    else:
        source_offset = emit_global_scalar_offset(
            source_abi, logical_coordinates, lane_coordinate=lane_coordinate
        )
    if mode in {"gather", "local_copy", "local_gather"}:
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
        **remote,
        "writer_active": (
            _chip_local_unique_writer_active(iteration_abi)
            if result_space == "canonical_global"
            else "True"
        ),
        "tile": _bounded_vector_tile(
            raw["parameters"]["tile"], scalar_capacity, name="Boxing tile"
        ),
    }


def _boxing_is_local_narrowing(source_abi, result_abi):
    """Prove containment before inverting a compact source's owner map."""
    types = []
    for abi in (source_abi, result_abi):
        if abi.get("coordinate_space") not in {"local", "parent_shard_local"}:
            return False
        data = abi.get("distributed_type")
        if not isinstance(data, Mapping) or data.get("kind") != "distributed":
            return False
        types.append(type_from_data(data))
    return is_local_shard_subview(*types)


_REMOTE_PEER_VARIABLE = "boxing_peer"


def _boxing_leaf_crossing_chip_axes(source_abi, result_abi) -> tuple[int, ...]:
    """Chip axes whose peer instances one leaf must read, from its ABIs.

    Every chip owns a private instance of a distributed buffer at identical
    pool offsets, so a leaf whose destination region is not contained in the
    same chip's source region has to address the peers explicitly.
    """

    source = source_abi.get("distributed_type")
    if not isinstance(source, Mapping) or source.get("kind") != "distributed":
        return ()
    source_type = type_from_data(source)
    result = result_abi.get("distributed_type")
    result_type = (
        type_from_data(result)
        if isinstance(result, Mapping) and result.get("kind") == "distributed"
        # Boxing preserves the logical tensor, so a plain destination is the
        # source's logical type: it is materialized whole on every chip.
        else source_type.tensor
    )
    return boxing_crossing_chip_axes(source_type, result_type)


def _placement_abi(abi: Mapping[str, object]) -> tuple[tuple[int, ...], str]:
    distributed = abi.get("distributed_type")
    placement = (
        None if not isinstance(distributed, Mapping) else distributed.get("placement")
    )
    if not isinstance(placement, Mapping):
        raise CodegenError("Cross-chip Boxing source has no placement ABI.")
    hierarchy = tuple(int(value) for value in placement.get("hierarchy", ()))
    levels = str(placement.get("hierarchy_levels", ""))
    if not hierarchy or len(levels) != len(hierarchy):
        raise CodegenError("Cross-chip Boxing placement ABI is incomplete.")
    return hierarchy, levels


def _group_member_coordinates(
    hierarchy: tuple[int, ...], axes: tuple[int, ...], member: str
) -> tuple[str, ...]:
    """Decompose a dense group ordinal into per-axis placement coordinates."""

    coordinates: dict[int, str] = {}
    remaining = f"({member})"
    for position, axis in enumerate(reversed(axes)):
        extent = hierarchy[axis]
        is_most_major = position == len(axes) - 1
        coordinates[axis] = (
            remaining if is_most_major else f"(({remaining}) % {extent})"
        )
        if not is_most_major:
            remaining = f"(({remaining}) // {extent})"
    return tuple(coordinates[axis] for axis in axes)


def _replicated_owner_coordinates(
    hierarchy: tuple[int, ...], levels: str
) -> list[str]:
    """Owner coordinates for axes that do not distinguish source components.

    A block axis' owners share one chip instance, so any of their components
    holds the replicated value.  A chip/node axis does not: owner ``0``'s
    component only exists inside chip ``0``'s instance, so a replicated chip
    axis must keep this chip's own coordinate.
    """

    mesh = _mesh_coordinates(len(hierarchy))
    return [
        mesh[axis] if levels[axis] in {"c", "d"} else "0"
        for axis in range(len(hierarchy))
    ]


def _source_owner_coordinates(
    abi,
    logical_coordinates,
    *,
    group_coordinates: Mapping[int, str] | None = None,
):
    """Invert a value's owner map into local and owner-coordinate expressions.

    ``group_coordinates`` pins the axes whose owner a reduction group selects
    explicitly: a partial axis' component is chosen by the group member, not by
    the logical coordinate, so the caller's expression replaces this chip's own
    coordinate on those axes.
    """

    distributed = abi["distributed_type"]
    hierarchy, levels = _placement_abi(abi)
    owner_coordinates = _replicated_owner_coordinates(hierarchy, levels)
    for axis, coordinate in (group_coordinates or {}).items():
        if axis < 0 or axis >= len(hierarchy):
            raise CodegenError("Owner group axis is outside its placement.")
        owner_coordinates[axis] = coordinate
    exclusive = distributed.get("exclusive")
    if isinstance(exclusive, Mapping):
        # An E value exists only on its selected owner, so a reader outside
        # that owner group addresses the owner's coordinates, not its own.
        axes = tuple(int(value) for value in exclusive.get("axes", ()))
        owners = exclusive.get("owner_coordinates")
        owners = (0,) * len(axes) if owners is None else tuple(int(value) for value in owners)
        if not axes or len(axes) != len(owners) or any(
            axis < 0 or axis >= len(hierarchy) for axis in axes
        ):
            raise CodegenError("Cross-chip Boxing source has invalid E owner axes.")
        for axis, owner in zip(axes, owners, strict=True):
            owner_coordinates[axis] = str(owner)
    coordinates = []
    for coordinate, extent, policy in zip(
        logical_coordinates, _static_shape(abi, "logical_shape"),
        distributed["axis_policies"], strict=True,
    ):
        local, owners = _invert_split_axis(coordinate, str(extent), policy, hierarchy)
        coordinates.append(local)
        for axis, owner in owners:
            owner_coordinates[axis] = owner
    return tuple(coordinates), owner_coordinates, hierarchy


def _device_rank_expression(
    hierarchy: tuple[int, ...], levels: str, owner_coordinates
) -> tuple[str, int]:
    """Linearize chip coordinates into one owner's FlagCX device rank.

    ``_mesh_context`` derives every ``c`` coordinate from that rank with these
    strides, so this expression inverts the generated ``shard_coord``
    definitions and keeps both sides of the ABI in agreement.
    """

    axes = [
        axis
        for axis, level in enumerate(levels)
        if level == "c" and hierarchy[axis] > 1
    ]
    terms = []
    for position, axis in enumerate(axes):
        stride = prod((hierarchy[other] for other in axes[position + 1:]), start=1)
        coordinate = owner_coordinates[axis]
        terms.append(
            f"({coordinate})" if stride == 1 else f"({coordinate}) * {stride}"
        )
    extent = prod((hierarchy[axis] for axis in axes), start=1)
    return (" + ".join(terms) if terms else "0"), extent


def _cross_chip_remote_source(
    source_abi: Mapping[str, object],
    crossing: tuple[int, ...],
    *,
    peer_expression: str,
    peer_count: int,
    window_byte_offset: str = "boxing_source_byte_offset",
) -> dict[str, object]:
    """Describe the peer-instance read one crossing leaf has to perform."""

    _, levels = _placement_abi(source_abi)
    node_axes = tuple(axis for axis in crossing if levels[axis] == "d")
    if node_axes:
        raise CodegenError(
            "A Boxing that reads a peer node's instance requires an inter-node "
            f"transfer; placement axes {node_axes} are node level."
        )
    if peer_count <= 1:
        raise CodegenError(
            "A cross-chip Boxing requires a chip axis with several owners."
        )
    return {
        "remote_source": emit_remote_storage_pointer(
            source_abi, peer=peer_expression,
            window_byte_offset=window_byte_offset,
        ),
        "remote_peers": peer_count,
        "remote_value_type": _triton_dtype(str(source_abi["scalar_dtype"])),
        # The peer instances are addressed inside one FlagCX symmetric window,
        # so the runtime has to register this pool as that window.
        "remote_window_pool": str(source_abi.get("memory_space")),
    }


def _split_hierarchy_axes(distributed: Mapping[str, object]) -> set[int]:
    """Placement axes whose owners hold different elements of a value."""

    return {
        int(axis)
        for policy in distributed.get("axis_policies", ())
        if isinstance(policy, Mapping) and policy.get("kind") == "split"
        for stage in policy.get("stages", ())
        for axis in stage.get("hierarchy_axes", ())
    }


def _partial_axes(abi: Mapping[str, object]) -> tuple[int, ...]:
    """The reduction group a partial operand's ABI declares."""

    distributed = abi.get("distributed_type")
    if not isinstance(distributed, Mapping):
        return ()
    partial = distributed.get("partial")
    if not isinstance(partial, Mapping):
        return ()
    return tuple(int(axis) for axis in partial.get("axes", ()))


def _split_stage_map(
    distributed: Mapping[str, object],
) -> dict[int, tuple[int, object]]:
    """Map each split placement axis to its tensor axis and distribution."""

    result: dict[int, tuple[int, object]] = {}
    for tensor_axis, policy in enumerate(distributed.get("axis_policies", ())):
        if not isinstance(policy, Mapping) or policy.get("kind") != "split":
            continue
        for stage in policy.get("stages", ()):
            if not isinstance(stage, Mapping):
                continue
            distribution = stage.get("distribution")
            key = (
                None
                if not isinstance(distribution, Mapping)
                else tuple(sorted(distribution.items(), key=lambda item: item[0]))
            )
            for axis in stage.get("hierarchy_axes", ()):
                result[int(axis)] = (tensor_axis, key)
    return result


def _relayouts_destination_owners(
    source_abi: Mapping[str, object],
    result_abi: Mapping[str, object],
) -> bool:
    """Whether the destination's split owner map differs from the source's.

    A block axis the destination splits names one writer per owner: that owner
    holds those elements and every later reader addresses them through the same
    map.  A source-domain sweep instead assigns elements to blocks by the
    *source's* map, so on such an axis it elects the wrong writer -- either a
    block that owned those elements before the relayout, or (when the axis is
    in the reduction group) a single elected block standing in for owners the
    destination still distinguishes.  Both leave each block reading elements a
    peer wrote, while the hazard analysis reads the declared destination type,
    sees one consistent owner map, and proves a block-local hazard that needs
    no chip rendezvous.

    An axis the destination leaves broadcast is not affected: every owner there
    holds every element, so the elected writer is a valid one.
    """

    source_distribution = source_abi.get("distributed_type")
    result_distribution = result_abi.get("distributed_type")
    if not isinstance(source_distribution, Mapping) or not isinstance(
        result_distribution, Mapping
    ):
        return False
    if source_distribution.get("placement") != result_distribution.get("placement"):
        return False
    try:
        hierarchy, levels = _placement_abi(source_abi)
    except CodegenError:
        return False
    source_stages = _split_stage_map(source_distribution)
    result_stages = _split_stage_map(result_distribution)
    return any(
        axis in result_stages and source_stages.get(axis) != result_stages[axis]
        for axis in range(len(hierarchy))
        if levels[axis] == "b" and hierarchy[axis] > 1
    )


def _partial_group_axis_split(
    hierarchy: tuple[int, ...], levels: str, axes: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Separate a reduction group into peer-chip axes and same-chip axes."""

    chip = tuple(
        axis
        for axis in axes
        if levels[axis] in {"c", "d"} and hierarchy[axis] > 1
    )
    return chip, tuple(axis for axis in axes if axis not in set(chip))


def _element_selected_chip_axes(
    source_abi: Mapping[str, object],
    crossing: tuple[int, ...],
    *,
    axes: tuple[int, ...],
) -> tuple[int, ...]:
    """Crossing chip axes the reduction group does not contain.

    A crossing chip axis inside the group makes every peer contribute to the
    same destination element, so the leaf still enumerates its destination by
    enumerating its own source region.  An axis outside the group instead picks
    one source chip per destination element, and then this chip's source region
    names only its own share of that destination: the leaf has to iterate the
    destination domain.  Which domain is iterated is decided before the leaf's
    offsets are built, so the two kinds are told apart here.
    """

    if not crossing:
        return ()
    hierarchy, levels = _placement_abi(source_abi)
    group = set(_partial_group_axis_split(hierarchy, levels, axes)[0])
    return tuple(axis for axis in crossing if axis not in group)


def _element_selected_partial_source(
    source_abi: Mapping[str, object],
    crossing: tuple[int, ...],
    *,
    axes: tuple[int, ...],
    routed: bool,
    logical_coordinates,
    peer_variable: str = _REMOTE_PEER_VARIABLE,
    window_byte_offset: str = "boxing_source_byte_offset",
) -> tuple[dict[str, object], tuple[int, ...], None]:
    """Remote descriptor of a partial leaf whose peers hold distinct elements.

    When no member of the reduction group lives on a peer chip, the crossing
    axes split the source instead: one destination element is a reduction over
    a single chip's components, and which chip that is follows from the
    element's logical coordinate.  The peer loop therefore visits each chip once
    and a per-element predicate keeps the elements it owns, while the group is
    reduced inside that peer's instance exactly as a chip-local leaf reduces its
    own.  The component index stays the caller's routed decomposition: only the
    pool the group is read from changes.
    """

    hierarchy, levels = _placement_abi(source_abi)
    split = _split_hierarchy_axes(source_abi["distributed_type"])
    unselected = tuple(axis for axis in crossing if axis not in split)
    if unselected:
        raise CodegenError(
            "A partial Boxing that reads peer chips must either reduce those "
            f"peers or split its source over them, but placement axes "
            f"{unselected} do neither."
        )
    if not routed:
        raise CodegenError(
            "A partial Boxing whose source chip varies per element must iterate "
            "its destination domain to reach every element of it."
        )
    _, owner_coordinates, _ = _source_owner_coordinates(
        source_abi, logical_coordinates
    )
    rank, extent = _device_rank_expression(hierarchy, levels, owner_coordinates)
    return (
        {
            # Each element selects its own source chip, so the peer loop visits
            # every chip and a mask keeps the elements this iteration owns.  A
            # peer holding none of them contributes the reduction's neutral,
            # which is the identity of the combine, so the sweep is exact.
            "remote_owner": rank,
            **_cross_chip_remote_source(
                source_abi, crossing,
                peer_expression=peer_variable, peer_count=extent,
                window_byte_offset=window_byte_offset,
            ),
        },
        # The whole group is reduced inside the selected peer's instance.
        axes,
        None,
    )


def _cross_chip_partial_source(
    source_abi: Mapping[str, object],
    crossing: tuple[int, ...],
    *,
    axes: tuple[int, ...],
    routed: bool,
    logical_coordinates,
    member_variable: str,
    peer_variable: str = _REMOTE_PEER_VARIABLE,
    window_byte_offset: str = "boxing_source_byte_offset",
) -> tuple[dict[str, object], tuple[int, ...], tuple[str, str] | None]:
    """Remote descriptor, same-chip group axes, and (base, member) owner offsets.

    A partial value's components are one per placement owner, but only the
    owners of one chip are materialized inside that chip's instance.  When the
    reduction group itself spans chips, the reduction runs as a scalar loop over
    the group's chip members - one peer instance per iteration - with the
    group's remaining (block) owners reduced inside each instance as before.
    When it does not, the crossing axes select one peer per destination element
    instead; a ``None`` offset pair asks the caller for its own owner
    decomposition. The base follows each output element; only the member
    varies along the reduction axis. They must not share one vector index:
    broadcasting a value-dependent owner onto the reduction axis creates
    a value-by-value cross product instead of one address per group member.
    """

    hierarchy, levels = _placement_abi(source_abi)
    chip_axes, local_axes = _partial_group_axis_split(hierarchy, levels, axes)
    if not chip_axes:
        return _element_selected_partial_source(
            source_abi, crossing, axes=axes, routed=routed,
            logical_coordinates=logical_coordinates,
            peer_variable=peer_variable, window_byte_offset=window_byte_offset,
        )
    outside = tuple(axis for axis in crossing if axis not in set(chip_axes))
    if outside:
        raise CodegenError(
            "A partial Boxing that reads peer chips must select its peers from "
            f"its reduction group, but placement axes {outside} "
            "cross chips outside it."
        )
    element_selected = tuple(
        axis
        for axis in sorted(_split_hierarchy_axes(source_abi["distributed_type"]))
        if levels[axis] == "c" and hierarchy[axis] > 1 and axis not in set(chip_axes)
    )
    if element_selected:
        # ``shard_id`` is a scalar, so one loop iteration reads one peer for
        # every element of the tile. A split chip axis outside the group would
        # need a different peer per element while reducing, which is a separate
        # lowering rather than a variation of this loop.
        raise CodegenError(
            "A partial Boxing cannot reduce peer chips while placement axes "
            f"{element_selected} select a source chip per element."
        )
    chip_coordinates = dict(
        zip(
            chip_axes,
            _group_member_coordinates(hierarchy, chip_axes, peer_variable),
            strict=True,
        )
    )
    if routed:
        # A routed destination selects the source owner per logical element, so
        # its split axes come from the inversion rather than from this chip.
        _, owner_coordinates, _ = _source_owner_coordinates(
            source_abi, logical_coordinates, group_coordinates=chip_coordinates,
        )
    else:
        # A matching map reduces this chip's own region: every axis outside the
        # group keeps the reader's own coordinate.
        owner_coordinates = list(_mesh_coordinates(len(hierarchy)))
        for axis, coordinate in chip_coordinates.items():
            owner_coordinates[axis] = coordinate
    # Only chip axes choose the instance, so the rank is complete before the
    # group's in-instance owners are substituted.
    rank, _ = _device_rank_expression(hierarchy, levels, owner_coordinates)
    if any(
        coordinate in rank
        for coordinate in _mesh_coordinates(len(hierarchy))
    ):
        # A chip axis outside the group keeps this reader's own coordinate, so
        # the rank is a runtime i64 mesh value and needs the i32 shard-id
        # narrowing. A rank built only from the unrolled peer ordinal stays a
        # Python integer, which selects the compile-time shard-id lowering.
        rank = f"({rank}).to(tl.int32)"
    member_coordinates = ["0"] * len(hierarchy)
    for axis, coordinate in zip(
        local_axes,
        _group_member_coordinates(hierarchy, local_axes, member_variable),
        strict=True,
    ):
        owner_coordinates[axis] = "0"
        member_coordinates[axis] = coordinate
    component_base = _linear_owner_expression(owner_coordinates, hierarchy)
    component_member = (
        _linear_owner_expression(member_coordinates, hierarchy)
        if local_axes else member_variable
    )
    return (
        _cross_chip_remote_source(
            source_abi, crossing,
            peer_expression=rank,
            peer_count=prod((hierarchy[axis] for axis in chip_axes), start=1),
            window_byte_offset=window_byte_offset,
        ),
        local_axes,
        (component_base, component_member),
    )


def _cross_chip_boxing_source(
    source_abi: Mapping[str, object],
    crossing: tuple[int, ...],
    *,
    mode: str,
    logical_coordinates,
    lane_coordinate: str | None,
) -> dict[str, object]:
    """Remote-read descriptor and source offset for a non-reduction leaf."""

    if mode not in {"gather", "canonical_copy"}:
        raise CodegenError(
            "A Boxing leaf that reads peer chips must iterate its destination "
            f"domain, but mode {mode!r} iterates its source."
        )
    storage_kind = str(source_abi.get("storage_kind"))
    if storage_kind not in {
        "canonical_global", "compact_local", "compact_per_owner", "compact_per_chip",
    }:
        raise CodegenError(
            "A cross-chip Boxing source has unsupported storage kind "
            f"{storage_kind!r}."
        )
    local_coordinates, owner_coordinates, hierarchy = _source_owner_coordinates(
        source_abi, logical_coordinates
    )
    if str(source_abi.get("coordinate_space")) == "canonical_global":
        # Canonical storage keeps every logical element at the same offset on
        # every chip, so the peer's instance needs no coordinate inversion.
        offset = emit_global_scalar_offset(
            source_abi, logical_coordinates, lane_coordinate=lane_coordinate
        )
    else:
        offset = emit_local_scalar_offset(
            source_abi, local_coordinates, lane_coordinate=lane_coordinate,
            shard_coordinates=owner_coordinates,
        )
        if storage_kind in {"compact_per_owner", "compact_per_chip"}:
            if storage_kind == "compact_per_chip":
                storage_type = type_from_data(
                    source_abi.get("distributed_backing_type") or source_abi["distributed_type"]
                )
                axes = block_owner_axes(storage_type)
                owner = (_linear_owner_expression(
                    [owner_coordinates[axis] for axis in axes],
                    tuple(hierarchy[axis] for axis in axes),
                ) if axes else "0")
            else:
                owner = _linear_owner_expression(owner_coordinates, hierarchy)
            # The peer pointer addresses the table base. Select the physical
            # component belonging to this element before its dense local offset.
            base = emit_owner_base(source_abi, "0", owner_index=owner)
            offset = f"({base} + ({offset}))"
    _, levels = _placement_abi(source_abi)
    rank, extent = _device_rank_expression(hierarchy, levels, owner_coordinates)
    return {
        "remote_source_offset": offset,
        # Each element selects its own source chip, so the peer loop visits
        # every chip and a mask keeps the elements this iteration owns.
        "remote_owner": rank,
        **_cross_chip_remote_source(
            source_abi, crossing,
            peer_expression=_REMOTE_PEER_VARIABLE, peer_count=extent,
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
    # A plain tensor also exposes the complete logical address space, even
    # though its distributed-storage enum is compact_local. Physical sharing
    # distinguishes one chip-wide tensor from a private full-shaped replica.
    canonical_result = str(result_abi.get("storage_kind")) == "canonical_global" or (
        result_abi.get("distributed_type") is None
        and result_abi.get("coordinate_space") == "canonical_global"
        and result_abi.get("memory_sharing_scope") == "chip"
    )
    routed = False
    if not canonical_result:
        # A private destination needs the reduced value on every owner, not
        # just the unique writer used for shared canonical storage. Matching
        # logical maps permit the same source-group reduction directly into
        # its local (or parent-backed local) destination coordinates.
        source_distribution = source_abi.get("distributed_type") or {}
        result_distribution = result_abi.get("distributed_type") or {}
        matching_placement = (
            not result_distribution
            or source_distribution.get("placement") == result_distribution.get("placement")
        )
        matching_map = all(
            tuple(source_abi[key]) == tuple(result_abi[key])
            for key in (
                "local_capacity_shape", "logical_coordinate_expressions", "active_shape_expressions",
            )
        )
        if (
            str(result_abi.get("coordinate_space")) not in {"local", "parent_shard_local", "canonical_global"}
            or not matching_placement
        ):
            raise CodegenError(
                "Partial Boxing requires a destination on the source placement."
            )
        routed = not matching_map
    elif _relayouts_destination_owners(source_abi, result_abi):
        # Addressing shared canonical storage by logical coordinate makes a
        # source-domain sweep write every byte of the destination, so the bytes
        # look complete.  But the sweep writes them under the *source's* owner
        # map, so a destination axis the source splits differently gets its
        # elements from the wrong block.  Every later reader addresses the
        # destination's map, and the hazard analysis -- reading the declared
        # destination type -- proves a block-local hazard, so no chip
        # rendezvous separates them.  Enumerating the destination keeps each
        # block the writer of its own shard, which is what the result type
        # promises its readers.
        routed = True
    owner_stride = int(source_abi.get("component_stride_scalar_elements", 0))
    if owner_stride <= 0:
        raise CodegenError("Partial Boxing source has no owner component stride.")
    distributed = source_abi.get("distributed_type")
    assert isinstance(distributed, Mapping)
    partial = distributed.get("partial")
    assert isinstance(partial, Mapping)
    axes = tuple(int(value) for value in partial.get("axes", ()))
    placement = distributed.get("placement")
    if not isinstance(placement, Mapping):
        raise CodegenError("Partial Boxing source has no placement ABI.")
    hierarchy = tuple(int(value) for value in placement.get("hierarchy", ()))
    crossing = _boxing_leaf_crossing_chip_axes(source_abi, result_abi)
    # A crossing chip axis outside the reduction group makes this chip's source
    # region cover only its own share of the destination, so enumerating that
    # region would leave the peers' elements unwritten.  Which domain the leaf
    # iterates decides every offset below it, so it is settled before them.
    routed = routed or bool(
        _element_selected_chip_axes(source_abi, crossing, axes=axes)
    )
    iteration_abi = result_abi if routed else source_abi
    local_shape = _static_shape(iteration_abi, "local_capacity_shape")
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
        emit_logical_coordinate(iteration_abi, axis, local_coordinates)
        for axis in range(len(local_shape))
    )
    active = " & ".join(
        f"(({coordinate}) < ({emit_active_extent(iteration_abi, axis)}))"
        for axis, coordinate in enumerate(local_coordinates)
    ) or "True"
    remote: dict[str, object] = {}
    group_axes = axes
    partial_owner: str | None = None
    partial_owner_base: str | None = None
    if crossing:
        # The group's chip members live in peer instances of this pool, so they
        # are visited one peer at a time and only the remaining owners stay in
        # the in-instance owner vector.
        remote, group_axes, owner_offsets = _cross_chip_partial_source(
            source_abi, crossing, axes=axes, routed=routed,
            logical_coordinates=logical_coordinates,
            member_variable="boxing_partial_member",
        )
        remote["remote_local_base"] = emit_storage_pointer(
            source_abi, str(source["runtime_argument"])
        )
        if owner_offsets is not None:
            partial_owner_base, partial_owner = owner_offsets
    if routed:
        # The destination owns the iteration domain. Each logical element
        # selects its source shard; only partial axes form the reduction group.
        source_coordinates, source_owner = _compact_source_coordinates(
            source_abi, logical_coordinates
        )
        source_offset = emit_local_scalar_offset(
            source_abi, source_coordinates, lane_coordinate=lane_coordinate,
        )
        if partial_owner is None:
            # The element's own owner fixes every axis outside the group, and
            # the group's ordinal fixes the rest, so the two sum to the source
            # component. A remote peer keeps the same decomposition: the peer
            # only changes which instance of the pool it is read from.
            source_offset = f"(({source_owner}) * {owner_stride} + ({source_offset}))"
            partial_owner = _group_owner_expression_for_axes(
                source_abi, axes, "boxing_partial_member",
                preserved_coordinates=tuple("0" for _ in hierarchy),
            )
    else:
        source_offset = emit_local_scalar_offset(
            source_abi, local_coordinates, lane_coordinate=lane_coordinate,
        )
        if partial_owner is None:
            partial_owner = _partial_group_owner_expression_for_axes(
                source_abi, axes, "boxing_partial_member"
            )
    if partial_owner_base is not None:
        # This offset varies with the destination value, so it belongs beside
        # its local element offset, on the load's value axis. The independent
        # partial_owner vector remains on its reduction axis.
        source_offset = f"(({partial_owner_base}) * {owner_stride} + ({source_offset}))"
    owner_count = prod((hierarchy[axis] for axis in group_axes), start=1)
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
        "source_offset": source_offset,
        "result_offset": (
            emit_global_scalar_offset(
                result_abi, logical_coordinates, lane_coordinate=lane_coordinate,
            )
            if canonical_result else emit_local_scalar_offset(
                result_abi, local_coordinates, lane_coordinate=lane_coordinate,
            )
        ),
        "active": active,
        **remote,
        # Writer election follows the domain the leaf enumerates.  A
        # source-domain sweep assigns elements by the source's owner map, so
        # the source's replica axes name the redundant writers.  A routed leaf
        # enumerates the destination, so the destination's own map does: its
        # split axes each keep one writer per owner, including axes the
        # reduction folded away on the source side.
        "writer_active": (
            _reduced_destination_writer_active(
                result_abi if routed else source_abi
            )
            if canonical_result else "True"
        ),
        "partial_owner_count": owner_count,
        "partial_owner_tile": owner_tile,
        "placement_owner_count": placement_owner_count,
        "partial_owner": partial_owner,
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


def _flatten_leading_logical_row(
    global_shape: tuple[int, ...], logical_coordinates: tuple[str, ...]
) -> str:
    """Flatten the leading (non-last) logical coordinates into a single global
    row index, matching a token-replicated per-owner partials workspace whose
    row dimension counts the flattened leading extents of the logical shape."""
    leading = global_shape[:-1]
    terms = []
    stride = 1
    for axis in range(len(leading) - 1, -1, -1):
        if leading[axis] != 1:
            terms.append(f"(({logical_coordinates[axis]}) * {stride})")
        stride *= leading[axis]
    return " + ".join(terms) if terms else "0"


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
        # Logical channel ownership and physical storage are independent:
        # inputs/weights may be owner-local while the result is a canonical
        # shared tensor (or vice versa). Resolve each operand's own ABI.
        "source_offset": emit_local_scalar_offset(qkv["abi"], ("token_index", local_channel)),
        "result_offset": emit_local_scalar_offset(result_abi, ("token_index", local_channel)),
        "weight_offset": emit_local_scalar_offset(
            weight["abi"], (local_channel, *("0" for _ in weight["abi"]["logical_shape"][1:]))
        ),
        "weight_kernel_stride": int(weight["abi"]["scalar_storage_strides"][-1]),
        "round_products": bool(attrs.get("round_products", True)),
        "round_before_activation": bool(attrs.get("round_before_activation", True)),
        "current_first": attrs.get("accumulation_order", "current_first") == "current_first",
        "tokens": local_shape[0],
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
    b_encoding = attrs.get("b_weight_encoding")
    a_encoding = attrs.get("a_weight_encoding")
    for encoding in (b_encoding, a_encoding):
        if encoding not in (None, "IQ4_XS", "Q4_K"):
            raise CodegenError(f"Unsupported native GGUF recurrent encoding {encoding!r}.")
    distributed_data = result_abi.get("distributed_type")
    uniform_projection_rows = distributed_data is not None and tiles_stay_within_groups(
        type_from_data(distributed_data),
        1,
        int(raw["parameters"]["tile_state"][1]),
        int(attrs["value_head_dim"]),
    )
    result_context = {
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
        "z_offset": emit_local_scalar_offset(z["abi"], ("0", local_value)),
        "result_offset": emit_local_scalar_offset(result_abi, ("0", local_value)),
        "local_capacity": local_shape[-1],
        "active_values": emit_active_extent(result_abi, 1),
        "global_value": global_value,
        "global_projection_value": emit_logical_coordinate(result_abi, 1, ("0", "recurrent_start")),
        "value_heads": int(attrs["num_value_heads"]),
        "key_heads": int(attrs["num_key_heads"]),
        "key_dim": int(attrs["key_head_dim"]),
        "value_dim": int(attrs["value_head_dim"]),
        "projection_dim": _static_shape(projection_input["abi"], "logical_shape")[-1],
        "repeats": int(attrs["num_value_heads"]) // int(attrs["num_key_heads"]),
        "epsilon": repr(float(attrs["epsilon"])),
        "value_tile": int(raw["parameters"]["tile_state"][1]),
        "head_block": int(raw["parameters"]["tile_state"][0]),
        "projection_tile": int(raw["parameters"]["projection_tile"]),
        "uniform_projection_rows": uniform_projection_rows,
        "query_scale": repr(1.0 / (int(attrs["key_head_dim"])**0.5)),
        "qk_norm_add": attrs.get("qk_norm_mode", "clamp") == "add",
        "qk_norm_epsilon": repr(float(attrs.get("qk_norm_epsilon", 1e-12))),
        "round_normalized_qk": bool(attrs.get("round_normalized_qk", False)),
        "round_beta": bool(attrs.get("round_beta", True)),
        "round_core": bool(attrs.get("round_core", False)),
        "writer_active": _distributed_unique_writer_active(result_abi),
        "b_weight_encoding": b_encoding,
        "a_weight_encoding": a_encoding,
        "tiled_value_heads": attrs.get("value_head_order", "grouped") == "tiled",
    }
    if raw.get("variant") == "state_smem_pipeline":
        from triton.flagmega.ir.tir import tir_from_data
        partition = raw["transfer_pipeline"]["channels"][0].get("inplace_partition")
        partition = None if partition is None else tir_from_data(partition)
        if (partition is None or partition.source_field_path != ("recurrent", ) or partition.source_row_rank != 3
                or partition.output_index != 0 or partition.output_axis != 1
                or partition.tile_rows != result_context["value_tile"]):
            raise CodegenError("GDN state pipeline requires the verified recurrent row partition.")
        result_context.update({
            name: raw["parameters"][name]
            for name in ("num_stages", "producer_warps", "producer_registers", "consumer_warps")
        })
    return result_context


def _chip_local_owner_group(abi: Mapping[str, object], owner_count: int) -> tuple[str, int]:
    """Dense CTA ordinal in one device's allocation, not the logical mesh.

    Removing c/d coordinates explicitly also handles interleaved rank and
    block axes. A modulo of the flattened logical owner is not equivalent.
    Callers must establish that their reduction does not span device ranks.
    """
    distributed = abi.get("distributed_type")
    if not isinstance(distributed, Mapping):
        return "shard_index", owner_count
    placement = distributed.get("placement", {})
    hierarchy = tuple(int(value) for value in placement.get("hierarchy", ()))
    levels = str(placement.get("hierarchy_levels", "")) or "b" * len(hierarchy)
    if len(levels) != len(hierarchy):
        raise CodegenError("Chip-local owner group has inconsistent placement levels.")
    axes = tuple(axis for axis, level in enumerate(levels) if level not in {"c", "d"})
    if not axes:
        return "0", 1
    return _mesh_group_member_expression_for_axes(abi, axes)


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
    owner_count = int(raw["parameters"].get("owner_count", 1))
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
        # The owner-local square sum is a Sum-partial only when the RMS reduction
        # axis is split across owners. When the whole normalized feature axis is
        # owner-local, NormStats materializes the statistics (partial is None):
        # the same local square sum is then already the complete per-token
        # statistic and is written straight to the output's own (token-sharded)
        # slot. Reject only a non-Sum partial, which this kernel cannot combine.
        if isinstance(stats_partial, Mapping) and stats_partial.get("reduce_op") != "sum":
            raise CodegenError(
                "AddNormStats local_partial_rms can only combine a Sum-partial "
                "statistics result."
            )
        # M==1 (decode) flattens one owner-local row; M>1 (prefill) iterates an
        # explicit row loop (add_stats_row) with one RMS partial statistic per
        # leading row.
        result_local_shape = _static_shape(result_abi, "local_capacity_shape")
        local_m = prod(result_local_shape[:-1], start=1)
        row_coordinate = "add_stats_row" if local_m != 1 else None
        domain = _scalar_local_domain(
            result_abi, "add_stats_local_offsets", row_coordinate=row_coordinate
        )
        local_capacity = int(domain["capacity"])
        # A token-sharded result pads its owner-local rows: a non-divisible token
        # count leaves inactive rows on some owners, and -- with fewer tokens than
        # token owners -- a single-capacity shard (local_m == 1) may own no token
        # at all. The per-element value stores are masked, but the once-per-row
        # statistics store is not, so an inactive row would publish its zero
        # square sum past the canonical statistics buffer. Gate that store on the
        # row's active extent too, indexing the loop row for M>1 and local row 0
        # for the single-capacity case.
        stats_row_active = " & ".join(
            f"(({coordinate}) < ({emit_active_extent(result_abi, index)}))"
            for index, coordinate in enumerate(
                _unflattened_coordinates(
                    result_local_shape[:-1],
                    "add_stats_row" if local_m != 1 else "0",
                )
            )
        ) or "True"
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
        if prod(stats_shape, start=1) != local_m:
            raise CodegenError(
                "AddNormStats local_partial_rms requires one local statistics "
                "element per result row."
            )
        stats_coordinates = (
            _unflattened_coordinates(stats_shape, "add_stats_row")
            if local_m != 1
            else ("0",) * len(stats_shape)
        )
        return {
            "source": _pointer(source),
            "residual": _pointer(residual),
            "result": _pointer(result),
            "stats": _pointer(stats),
            "partials": None,
            "mode": "local_partial_rms",
            "local_m": local_m,
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
                stats_abi, stats_coordinates
            ),
            "stats_writer_active": _canonical_writer_active(stats_abi),
            "stats_row_active": stats_row_active,
            "tile": _bounded_vector_tile(
                raw["parameters"]["tile"],
                local_capacity,
                name="AddNormStats local partial tile",
            ),
            "output_type": _triton_dtype(str(result_abi["scalar_dtype"])),
            "elements": prod(shape),
            "owner_count": owner_count,
        }
    facts = raw.get("facts", {})
    if not isinstance(facts, Mapping):
        raise CodegenError("AddNormStats implementation facts must be a mapping.")
    participant_scope = str(facts.get("participant_scope", "all_programs"))
    if participant_scope not in {"all_programs", "single_program"}:
        raise CodegenError(
            "AddNormStats participant_scope must be all_programs or single_program."
        )
    attrs = raw.get("semantic_attrs", {})
    axis = int(attrs.get("axis", -1))
    axis = axis + len(shape) if axis < 0 else axis
    # One RMS statistic per OUTER row: the dimensions before `axis` are
    # independent normalization groups (tokens), while the suffix shape[axis:]
    # is reduced per row. elements // norm_size is the owner-local outer-row
    # count; a single-outer-row input keeps norm_size == elements.
    norm_size = prod(shape[axis:], start=1) * lane_count
    owner_index, local_owner_count = _chip_local_owner_group(result_abi, owner_count)
    return {
        "source": _pointer(source),
        "residual": _pointer(residual),
        "result": _pointer(result),
        "stats": _pointer(stats),
        "partials": None if partials is None else _pointer(partials),
        "elements": prod(shape) * lane_count,
        "norm_size": norm_size,
        "tile": int(raw["parameters"]["tile"]),
        "owner_count": local_owner_count,
        "owner_index": owner_index,
        "mode": "global_rms",
        "participant_active": (
            f"({owner_index} == 0)"
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
    # M==1 (decode) reduces one owner-local row; M>1 (prefill) iterates an
    # explicit row loop (gather_row), one RMS partial per (owner, row).
    partial_local_shape = _static_shape(partial_abi, "local_capacity_shape")
    local_m = prod(partial_local_shape[:-1], start=1)
    row_coordinate = "gather_row" if local_m != 1 else None
    # The partials workspace holds one RMS partial per (owner, global token):
    # the partial input is token-replicated, so its local_m rows span the full
    # token axis. The statistics OUTPUT, however, is token-sharded, with only
    # stats_local_m local tokens per owner. The finalize must therefore iterate
    # the OUTPUT's local tokens and, for each, read the partial at that token's
    # global index -- not reuse the partial's row count (which would wrap).
    stats_local_shape = _static_shape(stats_abi, "local_capacity_shape")
    stats_local_m = prod(stats_local_shape[:-1], start=1)
    # The finalize reads the partial for each OUTPUT token at its global index.
    # With multiple output rows the token comes from the gather_stats_row loop;
    # with a single-row (token-sharded) output there is no loop, so the one row
    # is local row 0 -- its logical coordinate still names the owned token.
    stats_finalize_row = "gather_stats_row" if stats_local_m != 1 else "0"
    stats_finalize_domain = _scalar_local_domain(
        stats_abi, "0", row_coordinate=stats_finalize_row
    )
    stats_partial_token = _flatten_leading_logical_row(
        _static_shape(stats_abi, "logical_shape"),
        stats_finalize_domain["logical_coordinates"],
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
    # With row_coordinate the domain's capacity is one owner-local row; the row
    # loop (gather_row) supplies the leading coordinate. M==1 keeps the flat
    # whole-tensor domain (scalar_capacity == the single row).
    gather_domain = _scalar_local_domain(
        partial_abi, "gather_local_offsets", row_coordinate=row_coordinate
    )
    scalar_capacity = int(gather_domain["capacity"])
    lane_coordinate = gather_domain["lane_coordinate"]
    local_coordinates = gather_domain["local_coordinates"]
    logical_coordinates = gather_domain["logical_coordinates"]
    active = gather_domain["active"]
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
    private_result = str(result_abi.get("storage_kind")) != "canonical_global"
    private_context = {}
    if private_result:
        result_domain = _scalar_local_domain(result_abi, "gather_result_offsets")
        result_lane = result_domain["lane_coordinate"]
        private_context = {
            "result_capacity": result_domain["capacity"],
            "result_active": result_domain["active"],
            "result_tile": _bounded_vector_tile(
                raw["parameters"]["tile"], result_domain["capacity"],
                name="GatherReduceAddNormStats private result tile",
            ),
            "finalize_collective_offset": emit_global_scalar_offset(
                collective["abi"], result_domain["logical_coordinates"], lane_coordinate=result_lane,
            ),
            "finalize_residual_offset": _access_in_result_domain(
                residual_abi, result_abi, result_domain, lane_coordinate=result_lane,
            ),
            "finalize_result_offset": emit_local_scalar_offset(
                result_abi, result_domain["local_coordinates"], lane_coordinate=result_lane,
            ),
        }
    return {
        "private_result": private_result,
        **private_context,
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
            stats_abi, stats_reduction_axes, "gather_stats_lane", allow_empty=True
        ),
        "stats_reduction_count": stats_reduction_count,
        "stats_reduction_width": _bounded_vector_tile(
            raw["parameters"].get("partial_reduction_width"),
            stats_reduction_count,
            name="GatherReduceAddNormStats statistics reduction_width",
        ),
        "stats_store_active": _canonical_writer_active(stats_abi),
        "participant_active": _distributed_unique_writer_active(
            partial_abi, allow_redundant_axes=partial_axes
        ),
        "work_partition_index": work_partition_index,
        "work_partition_count": work_partition_count,
        "local_capacity": scalar_capacity,
        "local_m": local_m,
        "stats_local_m": stats_local_m,
        "stats_partial_token": stats_partial_token,
        "stats_partials_store_offset": (
            "shard_index" if local_m == 1
            else f"shard_index * {local_m} + gather_row"
        ),
        "stats_partials_owner_stride": local_m,
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
        "result_offset": None if private_result else _access_in_result_domain(
            result_abi, partial_abi, domain, lane_coordinate=lane_coordinate,
        ),
        "collective_offset": emit_global_scalar_offset(
            collective["abi"],
            logical_coordinates,
            lane_coordinate=lane_coordinate,
        ),
        # Write each statistics row into the OUTPUT's own (token-sharded) local
        # slot via its local-to-physical mapping. For stats_local_m==1 the single
        # local row 0 still belongs to a different global token on each owner, so
        # a global (0,0,0) destination would make every owner clobber stats[0]
        # and leave the other tokens unwritten -- stats_finalize_domain already
        # names local row 0 for that case (row_coordinate "0").
        "stats_offset": emit_local_scalar_offset(
            stats_abi,
            stats_finalize_domain["local_coordinates"],
            lane_coordinate=stats_finalize_domain["lane_coordinate"],
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
    # The statistics workspace carries one (components, owner) slab per leading
    # token: a flat (components, owner) buffer for M==1 (decode), or a prepended
    # row axis for M>1 (prefill), matching the kernel's apply_row loop. Both
    # phases loop over the FINER of the partial-input / value-output local
    # domains (by scalar capacity, selected below); equal total capacity need
    # not mean equal outer-row count (a split-input / split-output reduce
    # scatter), so size and validate the workspace from that selected domain,
    # not the output, or the row-indexed store runs off the workspace.
    _partial_local = _static_shape(partial_abi, "local_capacity_shape")
    _value_local = _static_shape(value_abi, "local_capacity_shape")
    _domain_local = (
        _partial_local
        if prod(_partial_local, start=1) <= prod(_value_local, start=1)
        else _value_local
    )
    workspace_rows = prod(_domain_local[:axis], start=1)
    expected_stats_shape = (
        (stats_components, owner_count)
        if workspace_rows == 1
        else (workspace_rows, stats_components, owner_count)
    )
    if (
        _static_shape(stats_abi, "logical_shape") != expected_stats_shape
        or str(stats_abi.get("scalar_dtype")) != "float32"
        or int(stats_abi.get("scalar_lane_count", 1)) != 1
    ):
        raise CodegenError(
            "GatherReduceAddNormApply statistics workspace must be float32 "
            f"{list(expected_stats_shape)}."
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
    # Normalization is over the suffix shape[axis:]; the dimensions before axis
    # are independent statistics rows (tokens). M==1 (decode) is one owner-local
    # row; M>1 (prefill) iterates an explicit token loop (apply_row), one RMS per
    # outer row, each reducing the WHOLE owner-local suffix (not just the last
    # axis). The row-aware domain splits leading rows from the flattened suffix.
    local_rows = prod(local_shape[:axis], start=1)
    row_coordinate = "apply_row" if local_rows != 1 else None
    gather_domain = _scalar_local_domain(
        domain_abi, "gather_local_offsets",
        row_coordinate=row_coordinate, reduction_axis=axis,
    )
    scalar_capacity = int(gather_domain["capacity"])
    lane_coordinate = gather_domain["lane_coordinate"]
    local_coordinates = gather_domain["local_coordinates"]
    logical_coordinates = gather_domain["logical_coordinates"]
    active = gather_domain["active"]
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
    # A token's RMS reduces the square sums of every owner holding a DISTINCT
    # piece of that token's normalized suffix -- the work-partitioned gather
    # axes and any axes splitting the suffix shape[axis:] -- but NOT the axes
    # that split the OUTER (token) rows, whose owners hold different tokens, nor
    # the broadcast axes that merely replicate the same piece. A partial axis can
    # itself become the output token split (a reduce-scatter result); that axis
    # splits the domain, so it is excluded from partition_axes and must stay
    # preserved. Using partition_axes (partial axes NOT splitting the domain)
    # rather than every partial axis keeps that case correct. Each owner
    # publishes its square sum at its own full-mesh slot; the finalize reduces
    # the group sharing this CTA's token (outer) coordinate (mirrors the sibling
    # stats kernel's preserved/reduction-axis reduction).
    suffix_split_axes = {
        int(mesh_axis)
        for policy in domain_distributed["axis_policies"][axis:]
        if isinstance(policy, Mapping) and policy.get("kind") == "split"
        for stage in policy.get("stages", ())
        for mesh_axis in stage.get("hierarchy_axes", ())
    }
    stats_reduction_axes = tuple(
        sorted(set(partition_axes) | suffix_split_axes)
    )
    stats_reduction_count = prod(
        (hierarchy[index] for index in stats_reduction_axes), start=1
    )
    stats_partial_owner = _group_owner_expression_for_axes(
        domain_abi, stats_reduction_axes, "norm_owner_offsets", allow_empty=True,
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
        "stats_owner_count": stats_reduction_count,
        "stats_partial_owner": stats_partial_owner,
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
        "local_m": local_rows,
        # Per-row block of the statistics workspace: one (components, owner_count)
        # slab per leading token. M==1 keeps the flat single-row layout.
        "row_stats_stride": (2 if use_mean else 1) * owner_count,
        "stats_square_base": owner_count if use_mean else 0,
        "round_before_scale": bool(attrs.get("round_before_scale", False)),
        "output_type": _triton_dtype(str(value_abi["scalar_dtype"])),
        "norm_output_type": _triton_dtype(str(norm_abi["scalar_dtype"])),
    }


def _norm_apply_call(raw) -> dict[str, object]:
    return _norm_apply_context(raw, stats_formal="stats")


def _gated_norm_apply_call(raw) -> dict[str, object]:
    """Reuse the norm_apply access plan and add the gate in the result domain."""
    algorithm = raw["parameters"].get("inverse_std_algorithm", "rsqrt")
    if not isinstance(algorithm, str) or algorithm not in {"rsqrt", "sqrt_div"}:
        raise CodegenError(
            "TIR GatedNormApply inverse_std_algorithm must be rsqrt or sqrt_div; "
            f"got {algorithm!r}."
        )
    return {
        **_norm_apply_context(raw, stats_formal="stats", gate_formal="gate"),
        "inverse_std_algorithm": algorithm,
    }


def _norm_apply_context(
    raw, *, stats_formal: str, gate_formal: str | None = None
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
    gate_context: dict[str, object] = {}
    if gate_formal is not None:
        gate = _buffer(raw, "inputs", gate_formal)
        gate_abi = gate["abi"]
        if (
            int(gate_abi.get("scalar_lane_count", 1)) != lane_count
            or tuple(gate_abi.get("scalar_lane_shape", ())) != lane_shape
        ):
            raise CodegenError(
                "TIR GatedNormApply gate vector lanes must match the result."
            )
        if _static_shape(gate_abi, "logical_shape") != logical_shape:
            raise CodegenError(
                "TIR GatedNormApply gate shape must equal the result shape."
            )
        activation = str(attrs.get("activation", "sigmoid"))
        if activation not in ("sigmoid", "silu", "swish"):
            raise CodegenError(
                f"TIR GatedNormApply activation {activation!r} is not supported."
            )
        gate_context = {
            "gate": _pointer(gate),
            "gate_offset": _access_in_result_domain(
                gate_abi, result_abi, domain, lane_coordinate=lane_coordinate,
            ),
            "gate_type": _triton_dtype(str(gate_abi["scalar_dtype"])),
            "activation": activation,
        }
    return {
        **gate_context,
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
        "input_type": _triton_dtype(str(source_abi["scalar_dtype"])),
        "output_type": _triton_dtype(str(result_abi["scalar_dtype"])),
    }


def _gather_reduce_norm_apply_call(raw) -> dict[str, object]:
    partial_reduction = raw["parameters"].get("partial_reduction", "peer_sum")
    if not isinstance(partial_reduction, str) or partial_reduction not in {"peer_sum", "global_tree"}:
        raise CodegenError("GatherReduceNormApply partial_reduction must be peer_sum or global_tree.")
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
    _, levels = _placement_abi(partial_abi)
    chip_axes, local_axes = _partial_group_axis_split(hierarchy, levels, partial_axes)
    remote = {}
    if chip_axes:
        remote, local_axes, owners = _cross_chip_partial_source(
            partial_abi, chip_axes, axes=partial_axes, routed=False,
            logical_coordinates=stats_logical_coordinates,
            member_variable="norm_partial_lane", peer_variable="norm_partial_peer",
            window_byte_offset="norm_partial_byte_offset",
        )
        assert owners is not None
        partial_owner = f"(({owners[0]}) + ({owners[1]}))"
        remote["remote_local_base"] = emit_storage_pointer(
            partial_abi, str(partial["runtime_argument"])
        )
    else:
        partial_owner = _partial_group_owner_expression_for_axes(
            partial_abi, partial_axes, "norm_partial_lane"
        )
    call = {
        "norm_partial_reduction": partial_reduction,
        "global_reduction_width": 1 << (max(1, prod(
            (hierarchy[index] for index in partial_axes), start=1)) - 1).bit_length(),
        "source": _pointer(source),
        "scale": _pointer(scale),
        "bias": _pointer(bias),
        "result": _pointer(result),
        "partial_stats": emit_storage_pointer(
            partial_abi, str(partial["runtime_argument"])
        ),
        "partial_owner": partial_owner,
        "partial_owner_count": prod(
            (hierarchy[index] for index in local_axes), start=1
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
        "input_type": _triton_dtype(str(source_abi["scalar_dtype"])),
        "output_type": _triton_dtype(str(result_abi["scalar_dtype"])),
        **remote,
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
    binary = variant in {"add", "mul", "div"}
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
    # Canonical storage is shared across broadcast owners. In-place outputs
    # require a single reader/writer, not redundant read-modify-write kernels.
    active = " & ".join(
        f"({value})" for value in (domain["active"], lhs_active, rhs_active, _canonical_writer_active(result_abi))
        if value != "True"
    ) or "True"
    from triton.flagmega.codegen.triton.fusion import elementwise_program
    from triton.flagmega.ir.op_fusion import has_ops
    fusion_lines, fusion_value = elementwise_program(raw) if has_ops(raw.get("semantic_attrs", {})) else ((), "")
    return {
        "fusion_lines": fusion_lines,
        "fusion_value": fusion_value,
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
    abi_lanes = tuple(int(value) for value in abi.get("scalar_lane_shape", ()))
    abi_lane_count = int(abi.get("scalar_lane_count", 1))
    if kind == "scalar":
        if axes or lanes:
            raise CodegenError("Scalar elementwise schedule cannot carry vector axes.")
    elif kind == "axes":
        if len(axes) != len(lanes) or (not abi_lanes and len(set(axes)) != len(axes)):
            raise CodegenError("Elementwise vector axes and lanes are inconsistent.")
        if any(axis < 0 or axis >= len(shape) for axis in axes):
            raise CodegenError("Elementwise vector axis is outside the local rank.")
        if any(lane < (1 if abi_lanes else 2) for lane in lanes):
            raise CodegenError("Elementwise vector lanes must be positive (nontrivial for scalar schedules).")
    elif kind == "typed_elements":
        if axes or not abi_lanes or lanes != abi_lanes or any(lane <= 0 for lane in lanes):
            raise CodegenError("Typed element traversal must match the physical vector element lanes without axes.")
    else:
        raise CodegenError(
            f"Elementwise call cannot consume vector schedule kind {kind!r}."
        )
    if abi_lanes:
        if kind not in {"axes", "typed_elements"} or lanes != abi_lanes:
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
    from triton.flagmega.codegen.triton.fusion import decode_fusion_attrs, vector_axes
    from triton.flagmega.ir.op_fusion import has_ops
    if (variant != "cast" and not has_ops(attrs)) or not isinstance(attrs, Mapping):
        raise CodegenError(
            "Elementwise operands with different VectorType lanes require an "
            "explicit vectorized cast contract."
        )
    axes = vector_axes(decode_fusion_attrs(attrs), len(_static_shape(result_abi, "local_capacity_shape")))
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
        or not axes
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
    from triton.flagmega.ir.ops.ntt.vectorized_cast import cast_vector_axes
    input_axes, output_axes = cast_vector_axes(input_lanes, output_lanes, axes, len(base_coordinates))
    components: list[str] = [""] * len(input_lanes)
    # Pack permits multiple lane groups on the same logical axis. Recover
    # that scalar coordinate in group order before splitting it into the
    # input groups; independently overwriting coordinates loses outer lanes.
    for axis in dict.fromkeys(input_axes):
        output_groups = [index for index, value in enumerate(output_axes) if value == axis]
        input_groups = [index for index, value in enumerate(input_axes) if value == axis]
        scalar_coordinate = base_coordinates[axis]
        for index in output_groups:
            scalar_coordinate = f"(({scalar_coordinate}) * {output_lanes[index]} + ({output_components[index]}))"
        remainder = scalar_coordinate
        for index in reversed(input_groups):
            components[index] = f"(({remainder}) % {input_lanes[index]})"
            remainder = f"(({remainder}) // {input_lanes[index]})"
        coordinates[axis] = remainder
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
    reduction_layout = raw["parameters"].get("reduction_layout", "default")
    if not isinstance(reduction_layout, str) or reduction_layout not in {"default", "contiguous_8"}:
        raise CodegenError("TIR NormStats reduction_layout must be default or contiguous_8.")
    reduction_algorithm = raw["parameters"].get("reduction_algorithm", "tree")
    if not isinstance(reduction_algorithm, str) or reduction_algorithm not in {"tree", "fma_1024"}:
        raise CodegenError("TIR NormStats reduction_algorithm must be tree or fma_1024.")
    if reduction_algorithm != "tree" and reduction_layout != "default":
        raise CodegenError("TIR NormStats contiguous_8 requires the tree reduction algorithm.")
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
    if reduction_algorithm == "fma_1024" and use_mean:
        raise CodegenError("TIR NormStats fma_1024 supports RMS statistics only.")
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
        "norm_stats_reduction_layout": reduction_layout,
        "norm_stats_reduction_algorithm": reduction_algorithm,
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


def _dense_matmul_mma_call(raw) -> dict[str, object]:
    """Render a bf16 WMMA GEMM (M>1) dense_matmul: M is a tile, not a row loop,
    so the weight tile is read once and reused across BM rows via tl.dot."""
    source = _buffer(raw, "inputs", "lhs")
    weight = _buffer_from_formals(raw, "inputs", ("rhs", "weight"))
    result = _buffer(raw, "outputs", "result")
    source_abi = source["abi"]
    weight_abi = weight["abi"]
    result_abi = result["abi"]
    if raw["parameters"].get("packed_layout") is not None:
        raise CodegenError("DenseMatMul mma requires a logical (unpacked) RHS.")
    source_shape = _static_shape(source_abi, "local_capacity_shape")
    result_shape = _static_shape(result_abi, "local_capacity_shape")
    local_m = prod(source_shape[:-1], start=1)
    local_k = source_shape[-1] * int(source_abi.get("scalar_lane_count", 1))
    local_n = result_shape[-1] * int(result_abi.get("scalar_lane_count", 1))
    global_k = _scalar_logical_axis_extent(source_abi, -1)
    global_n = _scalar_logical_axis_extent(result_abi, -1)
    weight_shape = _static_shape(weight_abi, "logical_shape")
    transpose_b = bool(raw.get("semantic_attrs", {}).get("transpose_b", False))
    expected = (global_n, global_k) if transpose_b else (global_k, global_n)
    if len(weight_shape) != 2 or weight_shape != expected:
        raise CodegenError("TIR DenseMatMul mma RHS shape disagrees with MxK@KxN contract.")
    # 2D owner-local tile offsets: M as arange rows (dense_m_offsets[:, None]),
    # the contracted/output axis as the last axis (dense_*_offsets[None, :]).
    source_domain = _scalar_last_axis_domain(
        source_abi, "dense_local_k_offsets[None, :]",
        owner="DenseMatMul mma lhs", row_coordinate="dense_m_offsets[:, None]")
    result_domain = _scalar_last_axis_domain(
        result_abi, "dense_local_n_offsets[None, :]",
        owner="DenseMatMul mma result", row_coordinate="dense_m_offsets[:, None]")
    # M-independent (1D) global N/K coordinates for the weight tile (row=0).
    global_k_axis = _scalar_last_axis_domain(
        source_abi, "dense_local_k_offsets", owner="DenseMatMul mma lhs K", row_coordinate="0")
    global_n_axis = _scalar_last_axis_domain(
        result_abi, "dense_local_n_offsets", owner="DenseMatMul mma result N", row_coordinate="0")
    weight_pointer, weight_offset = _dense_weight_access(
        weight, packed_layout=None, transpose_b=transpose_b,
        local_n="dense_local_n_offsets[:, None]",
        local_k="dense_local_k_offsets[None, :]",
        global_n="dense_global_n[:, None]",
        global_k="dense_global_k[None, :]",
    )
    block_k = _bounded_vector_tile(raw["parameters"]["block_k"], local_k, name="mma K tile")
    tile_n = _bounded_vector_tile(raw["parameters"]["tile_n"], local_n, name="mma N tile")
    return {
        "source": _pointer(source),
        "weight": weight_pointer,
        "result": _pointer(result),
        "local_m_capacity": local_m,
        # tl.arange / the accumulator shape must be a power of two, but M (prefill
        # token rows, or a standalone matmul's M=3/M=17) need not be. Round the M
        # tile up to the next power of two; m_mask (< m_active_extent) drops the
        # padding rows on load and store, so results are unaffected.
        "local_m_padded": 1 << (local_m - 1).bit_length(),
        "local_k_capacity": local_k,
        "local_n_capacity": local_n,
        "tile_n": tile_n,
        "block_k": block_k,
        "source_offset": source_domain["offset"],
        "result_offset": result_domain["offset"],
        "weight_offset": weight_offset,
        "global_n": global_n_axis["global_scalar"],
        "global_k": global_k_axis["global_scalar"],
        "m_active_extent": emit_active_extent(source_abi, 0),
        "k_active_extent": emit_active_extent(source_abi, len(source_shape) - 1),
        "n_active_extent": emit_active_extent(result_abi, len(result_shape) - 1),
        "output_type": _triton_dtype(str(result_abi["scalar_dtype"])),
    }


def _dense_matmul_call(raw) -> dict[str, object]:
    if str(raw.get("semantic_op", "")) == "ntt.matmul_norm_stats":
        return _dense_matmul_norm_stats_call(raw)
    if str(raw["variant"]) == "mma":
        return _dense_matmul_mma_call(raw)
    source = _buffer(raw, "inputs", "lhs")
    weight = _buffer_from_formals(raw, "inputs", ("rhs", "weight"))
    result = _buffer(raw, "outputs", "result")
    source_abi = source["abi"]
    weight_abi = weight["abi"]
    result_abi = result["abi"]
    source_rows = _static_shape(source_abi, "local_capacity_shape")[:-1]
    if source_rows != _static_shape(result_abi, "local_capacity_shape")[:-1]:
        raise CodegenError("DenseMatMul lhs/result owner-local row domains must match.")
    local_m = prod(source_rows, start=1)
    row_coordinate = "dense_local_m" if local_m != 1 else None
    source_domain = _scalar_last_axis_domain(
        source_abi, "dense_local_k_offsets", owner="DenseMatMul lhs", row_coordinate=row_coordinate
    )
    result_domain = _scalar_last_axis_domain(
        result_abi, "dense_local_n_offsets", owner="DenseMatMul result", row_coordinate=row_coordinate
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
        # A VectorizedMatMul RHS is the zero-copy N-major lane view of the
        # logical weight: its ABI keeps the packed extents plus the lane
        # count of the packed axis, which is axis zero under transpose_b and
        # axis one otherwise.
        weight_lanes = int(weight_abi.get("scalar_lane_count", 1))
        if weight_lanes != 1:
            if str(raw.get("semantic_op", "")) != "math.vectorized_matmul":
                raise CodegenError(
                    "A typed-vector DenseMatMul RHS requires the semantic "
                    "math.vectorized_matmul contract."
                )
            rhs_axes = tuple(int(axis) for axis in attrs.get("rhs_axes", ()))
            n_axis = 0 if transpose_b else 1
            if rhs_axes != (n_axis,) or len(weight_abi.get("scalar_lane_shape", ())) != 1:
                raise CodegenError(
                    "TIR DenseMatMul generic vector RHS must pack only its logical N axis."
                )
            packed_n = weight_shape[0] if transpose_b else weight_shape[1]
            if global_n != packed_n * weight_lanes:
                raise CodegenError(
                    "TIR DenseMatMul RHS shape disagrees with its logical MxK @ KxN "
                    f"contract: {weight_shape} (lanes {weight_lanes}) != N {global_n}, K {global_k}."
                )
        else:
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
        "packed_k_major_mma": None,
        "packed_k_major_threaded_fma": None,
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
    if variant == "packed_k_major_mma":
        if (source_abi["scalar_dtype"] != "bfloat16" or weight_abi["scalar_dtype"] != "bfloat16"
                or result_abi["scalar_dtype"] != "bfloat16"):
            raise CodegenError("Packed BF16 MMA requires BF16 source, weight and result.")
        if source_domain["capacity"] != global_k:
            raise CodegenError("Packed BF16 MMA requires the complete logical reduction on each owner.")
        # Tensor Core dimensions stay legal even for smaller owner tails.
        block_k = max(32, block_k)
        tile_n = max(16, tile_n)
    descriptor_requests = ()
    descriptor_variants = {
        "tensor_descriptor_gemv",
        "tensor_descriptor_smem_pipeline_gemv",
        "tensor_descriptor_smem_pipeline_aux_gemv",
        "packed_tensor_descriptor_smem_pipeline_gemv",
    }
    if variant in descriptor_variants:
        descriptor_kind = str(
            raw["parameters"].get("descriptor_kind", "single")
        )
        # A descriptor table names one resident owner component per entry, so it
        # describes the local shape of whatever storage the weight has. A single
        # map spans the canonical tensor and therefore needs canonical storage.
        chip_local_table = (
            descriptor_kind == "table"
            and str(weight_abi.get("storage_kind"))
            in {"compact_per_chip", "compact_per_owner"}
        )
        if (
            not chip_local_table
            and str(weight_abi.get("coordinate_space")) != "canonical_global"
        ):
            raise CodegenError(
                "Tensor-descriptor DenseMatMul requires canonical-global RHS storage."
            )
        weight_shape = _static_shape(
            weight_abi,
            "local_capacity_shape" if chip_local_table else "logical_shape",
        )
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
        "local_m_capacity": local_m,
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
        "weight_k_contiguity": (
            _k_major_n8_k16_k_contiguity(
                weight_abi, source_abi,
                local_k_capacity=source_domain["capacity"], block_k=block_k,
            )
            if packed_layout == "k_major_n8_k16" else 1
        ),
        "packed_layout": None if packed_layout is None else str(packed_layout),
        # weight_offset counts scalar elements, also for typed-vector storage.
        "weight_itemsize": (
            int(weight_abi["scalar_itemsize"]) if "scalar_itemsize" in weight_abi else None
        ),
        "accumulate_helper": accumulate_helper,
        "block_k": block_k,
        "tile_n": tile_n,
        "output_type": _triton_dtype(str(result_abi["scalar_dtype"])),
        "host_tensor_descriptor_requests": descriptor_requests,
    }
    if variant == "packed_k_major_threaded_fma":
        dot_lanes = int(raw["parameters"]["dot_lanes"])
        values_per_lane = int(raw["parameters"]["values_per_lane"])
        if (source_abi["scalar_dtype"] != "bfloat16" or weight_abi["scalar_dtype"] != "bfloat16"
                or result_abi["scalar_dtype"] != "float32" or source_domain["capacity"] != global_k
                or dot_lanes < 32 or dot_lanes & (dot_lanes - 1)
                or values_per_lane <= 0 or block_k != dot_lanes * values_per_lane
                or global_k % block_k):
            raise CodegenError("Threaded FMA requires BF16 operands, FP32 output and complete reduction chunks.")
        result.update(dot_lanes=dot_lanes, dot_warps=dot_lanes // 32, values_per_lane=values_per_lane)
    if descriptor_requests:
        descriptor_n_offset = _scalar_last_axis_domain(
            result_abi,
            "dense_local_n_start",
            owner="DenseMatMul descriptor result",
            row_coordinate=row_coordinate,
        )["global_scalar"]
        descriptor_k_offset = _scalar_last_axis_domain(
            source_abi,
            "dense_local_k_start",
            owner="DenseMatMul descriptor lhs",
            row_coordinate=row_coordinate,
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
        or str(residual_abi.get("scalar_dtype")) not in {"bfloat16", "float32"}
        or str(value_abi.get("scalar_dtype")) != str(residual_abi.get("scalar_dtype"))
    ):
        raise CodegenError(
            "Packed MatMulNormStats residual and value must have one identical "
            "BF16 or F32 vector ABI."
        )
    value_shape = _static_shape(value_abi, "logical_shape")
    value_lane_count = int(value_abi.get("scalar_lane_count", 1))
    value_scalar_strides = _static_shape(
        value_abi, "scalar_storage_strides"
    )
    value_rows = prod(value_shape[:-1], start=1)
    if not value_scalar_strides or value_scalar_strides[-1] != value_lane_count:
        raise CodegenError(
            "Packed MatMulNormStats statistics reduction requires a contiguous "
            "result backing."
        )
    # M>1 (prefill) iterates the matmul's own row loop (dense_local_m); each row
    # addresses its residual slice and writes its own statistic. The statistics
    # are one-per-OWNER-LOCAL row, so they must match the matmul's local row
    # count (not the logical M, which can be larger when rows are sharded).
    local_m = int(result.get("local_m_capacity", 1))
    stats_shape = _static_shape(stats_abi, "local_capacity_shape")
    if (
        int(stats_abi.get("scalar_lane_count", 1)) != 1
        or prod(stats_shape, start=1) != local_m
        or str(stats_abi.get("scalar_dtype")) != "float32"
    ):
        raise CodegenError(
            "Packed MatMulNormStats requires one scalar F32 statistic per "
            "owner-local result row."
        )
    row_coordinate = "dense_local_m" if local_m != 1 else None
    residual_domain = _scalar_last_axis_domain(
        residual_abi,
        "dense_local_n_offsets",
        owner="MatMulNormStats residual",
        row_coordinate=row_coordinate,
    )
    stats_coordinates = (
        _unflattened_coordinates(stats_shape, "dense_local_m")
        if local_m != 1
        else ("0",) * len(stats_shape)
    )
    result.update({
        "residual": _pointer(residual),
        # Projection precision belongs to the matmul, not the epilogue ABI.
        # Missing attributes retain the BF16 contract of old checkpoints.
        "projection_type": _triton_dtype(str(attrs.get("output_data_type", "bfloat16"))),
        "addend_cast_types": tuple(_triton_dtype(dtype) for dtype in attrs.get("addend_cast_dtypes", ())),
        "residual_offset": residual_domain["offset"],
        "stats": _pointer(stats),
        "stats_offset": emit_local_scalar_offset(stats_abi, stats_coordinates),
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
    # M==1 is decode; M>1 (prefill) wraps the fused gate/up gemv in a token-row
    # loop (dense_glu_local_m); the gate/up weights are shared across rows.
    source_local_shape = _static_shape(source_abi, "local_capacity_shape")
    local_m = prod(source_local_shape[:-1], start=1)
    row_coordinate = "dense_glu_local_m" if local_m != 1 else None
    source_domain = _scalar_last_axis_domain(
        source_abi, "dense_glu_local_k_offsets", owner="DenseMatMulGlu input",
        row_coordinate=row_coordinate,
    )
    result_domain = _scalar_last_axis_domain(
        result_abi, "dense_glu_local_n_offsets", owner="DenseMatMulGlu result",
        row_coordinate=row_coordinate,
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
        "round_activation": bool(raw.get("semantic_attrs", {}).get("round_activation", True)),
        "gate_clamp_max": raw.get("semantic_attrs", {}).get("gate_clamp_max"),
        "up_clamp": raw.get("semantic_attrs", {}).get("up_clamp"),
        "gate_weight": gate_weight_pointer,
        "up_weight": up_weight_pointer,
        "result": _pointer(result),
        "local_m": local_m,
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
            # Same contract the DenseMatMul descriptor path states: a table
            # names one resident owner component per entry, so it describes the
            # local shape of whatever storage the weight has, while a single map
            # spans the canonical tensor and needs canonical storage.
            chip_local_table = (
                descriptor_kind == "table"
                and str(abi.get("storage_kind"))
                in {"compact_per_chip", "compact_per_owner"}
            )
            if (
                not chip_local_table
                and str(abi.get("coordinate_space")) != "canonical_global"
            ):
                raise CodegenError(
                    "Packed descriptor DenseMatMulGlu requires canonical-global "
                    "weight storage."
                )
            weight_shape = _static_shape(
                abi,
                "local_capacity_shape" if chip_local_table else "logical_shape",
            )
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
    # M==1 is decode; M>1 (prefill) wraps the fused gemv in a token-row loop
    # (qkv_local_m), each row reading its own input slice and writing its q/k/v.
    source_local_shape = _static_shape(source_abi, "local_capacity_shape")
    local_m = prod(source_local_shape[:-1], start=1)
    row_coordinate = "qkv_local_m" if local_m != 1 else None
    source_domain = _scalar_last_axis_domain(
        source_abi, "qkv_local_k_offsets", owner="PackedQKV input",
        row_coordinate=row_coordinate,
    )
    output_domains = tuple(
        _scalar_last_axis_domain(
            output["abi"], "qkv_local_n_offsets", owner=f"PackedQKV output {index}",
            row_coordinate=row_coordinate,
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
        "local_m": local_m,
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
    # `packed_pointer_smem_pipeline` shares this ABI: it reads a strict subset
    # of the fields computed below, taking the weight as a plain pointer rather
    # than a descriptor, which is what its target registration asks for
    # (`descriptor_kind: "pointer"`). It is registered by the BI-V150 target
    # and has a template, so omitting it here rejected a variant the catalog
    # offers.
    if variant not in {
        "packed_mma_smem_pipeline",
        "packed_gemv_smem_pipeline",
        "packed_pointer_smem_pipeline",
    }:
        raise CodegenError(
            f"PackedQKV variant {variant!r} has no local-shard template ABI."
        )

    parameters = raw.get("parameters", {})
    pipeline = raw.get("transfer_pipeline")
    workspaces = raw.get("shared_workspaces")
    direct_lhs = parameters.get("direct_lhs", False) if isinstance(parameters, Mapping) else False
    if type(direct_lhs) is not bool:
        raise CodegenError("PackedQKV direct_lhs must be boolean.")
    if (
        not isinstance(parameters, Mapping)
        or not isinstance(pipeline, Mapping)
        or not isinstance(workspaces, (tuple, list))
        or len(workspaces) != (1 if direct_lhs else 2)
    ):
        raise CodegenError(
            "PackedQKV requires a typed transfer-pipeline, a weight stage and "
            "an LHS stage unless direct_lhs is declared."
        )
    block_n = int(parameters["block_n"])
    block_k = int(parameters["block_k"])
    num_stages = int(parameters["num_stages"])
    consumer_warps = int(parameters["consumer_warps"])
    worker_width = int(parameters["worker_width"])
    local_n = sum(domain["capacity"] for domain in output_domains)
    masked_n_tail = parameters.get("masked_n_tail", False)
    n_tiling = parameters.get("n_tiling", False)
    if type(masked_n_tail) is not bool or type(n_tiling) is not bool:
        raise CodegenError("PackedQKV masked_n_tail and n_tiling must be boolean.")
    n_tile_matches = block_n == local_n or (masked_n_tail and 0 < local_n < block_n) or (n_tiling and local_n > 0)
    if (
        not n_tile_matches
        or block_n <= 0 or block_n & (block_n - 1)
        or block_k <= 0
        or (not direct_lhs and source_domain["capacity"] % block_k)
        or block_k & (block_k - 1)
        or block_n % 8
        or block_k % 16
        or num_stages <= 0
        or consumer_warps <= 0
        or worker_width <= 0
        # The consumer partition must divide, but its literal width is the
        # target's warp geometry, not a constant: BI-V150 warps are 64 lanes
        # and that target registers 16 consumer warps of width 64 where sm90
        # registers 8 of width 32. Pinning 8x32 here rejected a variant the
        # catalog offers, and the sibling dense paths above validate the same
        # partition by divisibility alone.
        or worker_width & (worker_width - 1)
    ):
        raise CodegenError(
            "PackedQKV pipeline requires a full owner-local N tile, declared masked tail or N tiling, complete K16 "
            "tiles (or declared direct LHS tails), and a consumer partition that divides its payload."
        )
    if variant == "packed_pointer_smem_pipeline":
        # Everything below describes the MMA operand path: an (8,2,8) lane
        # shape, an owner-major [owner,K/16,N/8] backing, and a host tensor
        # map whose Shared box reshapes to an MMA [N,K] operand. This variant
        # has no `tl.dot` -- it multiplies elementwise and folds with
        # `tl.sum` -- and takes the weight as a plain pointer, so none of
        # those contracts apply to it. It needs the pipeline fields only.
        result.update({
            "pipeline_contract": dict(pipeline),
            "shared_workspaces": tuple(workspaces),
            "pipeline_channel_field_names": {"weight": ("weight",)},
            "block_n": block_n,
            "block_k": block_k,
            "num_stages": num_stages,
            "direct_lhs": direct_lhs,
            "num_k_tiles": (source_domain["capacity"] + block_k - 1) // block_k,
            "num_n_tiles": (local_n + block_n - 1) // block_n,
            "consumer_warps": consumer_warps,
            "worker_width": worker_width,
            "producer_warps": int(parameters["producer_warps"]),
            "producer_registers": int(parameters["producer_registers"]),
        })
        return result
    if lane_shape != (8, 2, 8):
        raise CodegenError(
            "PackedQKV MMA requires VectorType(N8,KPack2,KVector8) weights."
        )
    expected_weight_shape = (
        int(outputs[0]["abi"].get("owner_count", 0)),
        source_domain["capacity"] // 16,
        local_n // 8,
    )
    # The tensor map describes real storage; only its shared-memory box may
    # extend past N. Inflating these strides to block_n would read other rows.
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
    expected_shared_shapes = ((num_stages, *rendered_block_shape),) + (
        () if direct_lhs else ((1, source_domain["capacity"]),)
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
            f"tl.full((), qkv_n_tile * {block_n // 8}, tl.int32)",
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
            f"tl.full((), qkv_n_tile * {block_n // 8}, tl.int32)",
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
        "direct_lhs": direct_lhs,
        "num_k_tiles": (source_domain["capacity"] + block_k - 1) // block_k,
        "num_n_tiles": (local_n + block_n - 1) // block_n,
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


def _gather_reduce_qkv_statistics_call(raw):
    from .qkv_statistics_renderer import gather_reduce_qkv_statistics_call
    return gather_reduce_qkv_statistics_call(raw)


def _qkv_rope_with_cache_call(raw) -> dict[str, object]:
    """Encode fused Q/K normalization, RoPE, and paged-cache writes.

    The algorithm is expressed entirely in the local-buffer ABI.  Distribution
    affects active bounds and coordinate expressions; it does not select a
    model-, mesh-, or accelerator-specific kernel body. Rotary pairs must be
    owner-local; Q/K statistics are separate, materialized operands.
    """

    q = _buffer(raw, "inputs", "qkv", 0)
    k = _buffer(raw, "inputs", "qkv", 1)
    v = _buffer(raw, "inputs", "qkv", 2)
    return _encode_qkv_rope_with_cache(raw, q, k, v)


def _encode_qkv_rope_with_cache(
    raw,
    q,
    k,
    v,
    *,
    partial_sources=None,
) -> dict[str, object]:
    """Build the apply-only context for materialized QKV and statistics."""

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
        _buffer(raw, "inputs", "q_stats") if partial_sources is None else None,
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
        rotary_dim=attrs.get("rotary_dim"),
        prefix="qkv_q",
        tile=raw["parameters"]["elements_per_program"],
    )
    k_context = _qkv_norm_rope_context(
        k,
        _buffer(raw, "inputs", "k_stats") if partial_sources is None else None,
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
        rotary_dim=attrs.get("rotary_dim"),
        prefix="qkv_k",
        tile=raw["parameters"]["elements_per_program"],
    )
    v_context = _attention_scalar_domain(v["abi"], qkv_layout, "qkv_v_offsets")

    for head in (q_context, k_context):
        head["intermediate_type"] = (head["input_type"] if attrs.get("round_qk_intermediates", True)
                                     else "tl.float32")

    if partial_sources is not None:
        from .qkv_statistics_renderer import partial_qkv_access
        for source, head, prefix in zip(partial_sources, (q_context, k_context), ("qkv_q", "qkv_k")):
            for name, domain in (("partial_reduce", "reduce_domain"), ("partial_input", "apply_domain"),
                                 ("partial_partner", "partner_domain")):
                head[name] = partial_qkv_access(source, head[domain], qkv_layout, f"{prefix}_partial_member")
        v_context["partial_input"] = partial_qkv_access(
            partial_sources[2], v_context, qkv_layout, "qkv_v_partial_member")

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
        or v_context["global_extents"]["dim"] != cache_head_dim
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
    # The query chunk may carry more than one token (prefill). Each token writes
    # its K/V to a distinct cache slot (chunk base + its own sequence position),
    # so the physical block and in-block offset are computed per token in the
    # kernel (see cache_seq). Decode (one token, seq coordinate 0) is unchanged.
    q_context["cache_seq"] = q_context["apply_domain"]["global_by_kind"]["seq"]
    k_context["cache_seq"] = k_context["apply_domain"]["global_by_kind"]["seq"]
    v_context["cache_seq"] = v_context["global_by_kind"]["seq"]
    k_context["cache_offset"] = _paged_cache_scalar_offset(
        cache_abi,
        k_context["apply_domain"],
        cache_index=0,
        layer=layer_expression,
        physical_block="qkv_k_cache_physical_block",
        block_offset="qkv_k_cache_block_offset",
    )
    v_context["cache_offset"] = _paged_cache_scalar_offset(
        cache_abi,
        v_context,
        cache_index=1,
        layer=layer_expression,
        physical_block="qkv_v_cache_physical_block",
        block_offset="qkv_v_cache_block_offset",
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

    return {
        "headwise_partial": (
            partial_sources is not None and q_axis == k_axis == qkv_layout.index("dim") == 2
            and all(head["tile"] == head["reduction_capacity"] == head["head_dim"]
                    for head in (q_context, k_context))
            and v_context["tile"] == v_context["capacity"]
        ),
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
        "num_tokens": k_context["apply_domain"]["global_extents"]["seq"],
    }


def _invert_split_policy(
    coordinate: str,
    parent_extent: str,
    policy,
    hierarchy: tuple[int, ...],
) -> str:
    return _invert_split_axis(coordinate, parent_extent, policy, hierarchy)[0]


def _invert_split_axis(coordinate, parent_extent, policy, hierarchy):
    """Invert staged splits into a local coordinate and mesh coordinates."""
    if not isinstance(policy, Mapping):
        raise CodegenError("Serialized axis policy must be a mapping.")
    if str(policy.get("kind")) != "split":
        return coordinate, ()
    stages = policy.get("stages")
    if not isinstance(stages, (tuple, list)) or not stages:
        raise CodegenError("Serialized split policy requires stages.")
    local = coordinate
    extent = parent_extent
    owners = []
    for stage in stages:
        if not isinstance(stage, Mapping):
            raise CodegenError("Serialized split stage must be a mapping.")
        axes = tuple(int(value) for value in stage.get("hierarchy_axes", ()))
        if (
            not axes
            or any(axis < 0 or axis >= len(hierarchy) for axis in axes)
        ):
            raise CodegenError(
                "Operand split axes must belong to the placement."
            )
        count = prod(hierarchy[axis] for axis in axes)
        distribution = stage.get("distribution")
        if not isinstance(distribution, Mapping):
            raise CodegenError("Serialized split stage has no distribution.")
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
                raise CodegenError("Block-cyclic split requires a block size.")
            cycle = count * block
            owner = f"((({local}) // {block}) % {count})"
            local = f"((({local}) // {cycle}) * {block} + ({local}) % {block})"
            extent = (
                f"((({extent}) // {cycle}) * {block} + "
                f"tl.minimum(tl.maximum(({extent}) % {cycle} - "
                f"({owner}) * {block}, 0), {block}))"
            )
        else:
            raise CodegenError(f"Unsupported split distribution {kind!r}.")
        stage_coordinates = _unflattened_coordinates(
            tuple(hierarchy[axis] for axis in axes), owner
        )
        owners.extend(zip(axes, stage_coordinates, strict=True))
    return local, tuple(owners)


def _compact_source_coordinates(abi, logical_coordinates):
    coordinates, owner_coordinates, hierarchy = _source_owner_coordinates(
        abi, logical_coordinates
    )
    return coordinates, _linear_owner_expression(owner_coordinates, hierarchy)


def _serialized_fixed_dimension(value) -> str:
    if isinstance(value, Mapping) and value.get("kind") == "fixed":
        fixed = int(value.get("value", 0))
    elif isinstance(value, int) and not isinstance(value, bool):
        fixed = value
    else:
        raise CodegenError(
            "Triton contiguous granularity must be statically fixed."
        )
    if fixed <= 0:
        raise CodegenError("Triton split granularity must be positive.")
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
    stats,
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
    rotary_dim=None,
) -> dict[str, object]:
    source_abi = source["abi"]
    scalar_shape = _attention_scalar_shape(source_abi, input_layout)
    outer_shape = scalar_shape[:axis]
    reduction_shape = scalar_shape[axis:]
    outer_coordinates = _unflattened_coordinates(
        outer_shape, f"{prefix}_outer_index"
    )
    apply_coordinates = _unflattened_coordinates(
        reduction_shape, f"{prefix}_element_offsets"
    )
    apply_domain = _attention_scalar_coordinates(
        source_abi,
        input_layout,
        (*outer_coordinates, *apply_coordinates),
    )
    stats_offsets = None if stats is None else _norm_apply_stats_offsets(
        stats["abi"], source_abi, axis,
        apply_domain["physical_local_coordinates"],
        tuple(apply_domain["global_by_kind"][kind] for kind in input_layout),
        use_mean,
    )
    dimension = apply_domain["global_by_kind"]["dim"]
    head_dim = apply_domain["global_extents"]["dim"]
    rotary_dim = head_dim if rotary_dim is None else rotary_dim
    if isinstance(rotary_dim, bool) or not isinstance(rotary_dim, int) or not 0 < rotary_dim <= head_dim or rotary_dim % 2:
        raise CodegenError("QKVRoPEWithCache rotary_dim must be a positive even scalar extent within the head.")
    partner = (
        f"tl.where(({dimension}) >= {rotary_dim}, ({dimension}), "
        f"tl.where(({dimension}) < {rotary_dim} // 2, "
        f"({dimension}) + {rotary_dim} // 2, "
        f"({dimension}) - {rotary_dim} // 2))"
    )
    partner_coordinates = dict(apply_domain["global_by_kind"])
    partner_coordinates["dim"] = partner
    partner_offset = _owner_local_attention_offset(
        source_abi, input_layout, partner_coordinates,
    )
    partner_domain = {**apply_domain, "global_by_kind": partner_coordinates,
                      "offset": partner_offset}
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
        **({"stats": _pointer(stats),
            "stats_sum_offset": stats_offsets[0] if use_mean else None,
            "stats_square_sum_offset": stats_offsets[-1]} if stats is not None else {}),
        "stats_active": " & ".join(
            f"(({coordinate}) < ({emit_active_extent(source_abi, index)}))"
            for index, coordinate in enumerate(outer_coordinates)
        ) or "True",
        "scale": _pointer(scale),
        "bias": _pointer(bias),
        "cosine": _pointer(cosine),
        "sine": _pointer(sine),
        "outer_capacity": prod(outer_shape, start=1),
        "apply_capacity": capacity,
        "reduction_capacity": capacity,
        "normalization_size": prod(
            apply_domain["global_extents"][input_layout[index]]
            for index in range(axis, len(input_layout))
        ),
        "apply_active": apply_domain["active"],
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
        "rotary_dim": rotary_dim,
        "epsilon": repr(epsilon_value),
        "round_before_scale": round_before_scale,
        "input_type": _triton_dtype(str(source_abi["scalar_dtype"])),
        "use_mean": use_mean,
        "apply_domain": apply_domain,
        "partner_domain": partner_domain,
        "tile": _bounded_vector_tile(
            tile,
            capacity,
            name=f"QKVRoPEWithCache {prefix} apply tile",
        ),
    }
    if stats is None:
        reduce_coordinates = _unflattened_coordinates(reduction_shape, f"{prefix}_reduce_offsets")
        reduce_domain = _attention_scalar_coordinates(
            source_abi, input_layout, (*outer_coordinates, *reduce_coordinates))
        context.update({"reduce_domain": reduce_domain, "reduce_active": reduce_domain["active"],
                        "reduce_input_offset": reduce_domain["offset"]})
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


def _owner_local_attention_offset(abi, layout, global_by_kind):
    lanes = int(abi.get("scalar_lane_count", 1))
    coordinates = [str(global_by_kind[kind]) for kind in layout]
    dim_axis = layout.index("dim")
    lane = None
    if lanes != 1:
        lane = f"(({coordinates[dim_axis]}) % {lanes})"
        coordinates[dim_axis] = f"(({coordinates[dim_axis]}) // {lanes})"
    return _owner_local_operand_offset(abi, coordinates, lane)


def _owner_local_operand_offset(abi, coordinates, lane=None):
    """Address a proven owner-local global coordinate in either storage ABI."""
    if str(abi.get("coordinate_space")) == "canonical_global":
        return emit_global_scalar_offset(abi, coordinates, lane_coordinate=lane)
    distributed = abi.get("distributed_type")
    if isinstance(distributed, Mapping):
        hierarchy = tuple(distributed["placement"]["hierarchy"])
        coordinates = tuple(_invert_split_policy(
            str(coordinate), str(extent), policy, hierarchy,
        ) for coordinate, extent, policy in zip(
            coordinates, _static_shape(abi, "logical_shape"),
            distributed["axis_policies"], strict=True,
        ))
    return emit_local_scalar_offset(abi, coordinates, lane_coordinate=lane)


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
        dimension = source_domain["global_by_kind"]["dim"]
        lane_coordinate = f"(({dimension}) % {lanes})"
    return _owner_local_operand_offset(abi, global_coordinates, lane_coordinate)


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
    domain = _scalar_local_domain(cosine_abi, "rotary_offsets")
    lanes = int(cosine_abi.get("scalar_lane_count", 1))
    dimension = domain["logical_coordinates"][-1]
    if lanes != 1:
        dimension = f"(({dimension}) * {lanes} + ({domain['lane_coordinate']}))"
    attrs = raw.get("semantic_attrs", {})
    return {
        "sequence_lengths": _pointer(state[2]),
        "cosine": _pointer(cosine),
        "sine": _pointer(sine),
        "local_capacity": domain["capacity"],
        "active": domain["active"],
        "local_coordinates": domain["local_coordinates"],
        "logical_coordinates": domain["logical_coordinates"],
        "dimension_coordinate": dimension,
        "cosine_offset": emit_local_scalar_offset(
            cosine_abi, domain["local_coordinates"], lane_coordinate=domain["lane_coordinate"],
        ),
        "sine_offset": _access_in_result_domain(
            sine["abi"], cosine_abi, domain, lane_coordinate=domain["lane_coordinate"],
        ),
        "token_coordinate": domain["logical_coordinates"][0],
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
    rotary_dim = raw.get("semantic_attrs", {}).get("rotary_dim")
    if rotary_dim is None:
        rotary_dim = head_dim
    partner_dimension = (
        f"tl.where(({dimension}) < {rotary_dim} // 2, "
        f"({dimension}) + {rotary_dim} // 2, "
        f"({dimension}) - {rotary_dim} // 2)"
    )

    def source_offset(scalar_dimension: str) -> str:
        abi = source["abi"]
        source_lane_count = int(abi.get("scalar_lane_count", 1))
        coordinates = list(logical_coordinates)
        source_lane = None
        if source_lane_count == 1:
            coordinates[-1] = scalar_dimension
        else:
            coordinates[-1] = f"(({scalar_dimension}) // {source_lane_count})"
            source_lane = f"(({scalar_dimension}) % {source_lane_count})"
        return _owner_local_operand_offset(abi, coordinates, source_lane)

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
        "rotary_dim": rotary_dim,
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
    domain = _scalar_local_domain(slots_abi, "cache_offsets")
    shape = _static_shape(slots_abi, "logical_shape")
    cache_shape = _static_shape(state[0]["abi"], "logical_shape")
    attrs = raw.get("semantic_attrs", {})
    layout = tuple(attrs["layout"])
    head_axis, dim_axis = layout.index("head"), layout.index("dim")
    lane_count = int(slots_abi.get("scalar_lane_count", 1))
    head_dim = shape[dim_axis] * lane_count
    cache_head_dim = cache_shape[-1] * int(state[0]["abi"].get("scalar_lane_count", 1))
    pair = raw["family"] == "update_paged_attention_kv_cache_pair"
    if shape[head_axis] != cache_shape[-2] or head_dim != cache_head_dim * (2 if pair else 1):
        raise CodegenError("Cache update slots and paged storage have different scalar head geometry.")
    # Both pointers address scalars, even when the IR element is a vector.
    # Lanes belong to the semantic dim axis, which need not be the last axis.
    dimension = domain["logical_coordinates"][dim_axis]
    if lane_count != 1:
        dimension = f"(({dimension}) * {lane_count} + ({domain['lane_coordinate']}))"
    return {
        "slots": _pointer(slots),
        "kv_cache": _pointer(state[0]),
        "query_start_loc": _pointer(state[1]),
        "sequence_lengths": _pointer(state[2]),
        "slot_mapping": _pointer(state[3]),
        "block_table": _pointer(state[4]),
        "layer_id": _scalar_expression(layer_id),
        "advance_sequence": _scalar_expression(advance),
        "cache_index": f"(({dimension}) // {cache_head_dim})" if pair else (0 if attrs.get("cache_kind") == "key" else 1),
        "num_layers": cache_shape[1],
        "num_kv_heads": cache_shape[-2],
        "head_dim": cache_head_dim,
        "block_size": cache_shape[3],
        "local_capacity": domain["capacity"],
        "active": domain["active"],
        "writer_active": _chip_local_unique_writer_active(slots_abi),
        "metadata_writer_active": f"({_chip_local_owner_group(slots_abi, 1)[0]} == 0)",
        "num_tokens": shape[layout.index("seq")],
        "logical_token": domain["logical_coordinates"][layout.index("seq")],
        "logical_head": domain["logical_coordinates"][head_axis],
        "logical_dimension": f"(({dimension}) % {cache_head_dim})" if pair else dimension,
        "slots_offset": emit_local_scalar_offset(
            slots_abi, domain["local_coordinates"], lane_coordinate=domain["lane_coordinate"]
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
        "query_token": max_domain["logical_coordinates"][layout.index("seq")],
        "query_dim_stride": int(query_abi["scalar_storage_strides"][dim_axis]),
        "query_lanes": int(query_abi.get("scalar_lane_count", 1)),
        "accumulator_dim_stride": int(partial_accumulator["abi"]["scalar_storage_strides"][dim_axis]),
        "kv_cache": _pointer(state[0]),
        "slot_mapping": _pointer(state[3]),
        "block_table": _pointer(state[4]),
        "layer_id": _scalar_expression(layer_id),
        "partial_max": head_pointer(partial_max),
        "partial_sum": head_pointer(partial_sum),
        "partial_accumulator": head_pointer(partial_accumulator),
        # Physical block count: the kernel clamps block_table entries against
        # it, because the host driver refreshes only the leading columns per
        # step and a stale tail entry would otherwise address outside the
        # cache. The template referenced this field but the encoder never
        # provided it, so this kernel failed to render at all.
        "num_blocks": cache_shape[0],
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
    # Local-shard ABI covers every fixed-token-tile decode variant; the tile
    # value itself comes from the selected implementation parameters and only
    # has to be a valid Triton tile extent.
    if variant.startswith("decode_t") and variant[len("decode_t"):].isdecimal():
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
    def state_access(abi, statistics):
        # The state may shard heads differently from the result, so derive its
        # owner and local index from the element's logical coordinates.
        hierarchy = tuple(int(v) for v in abi["distributed_type"]["placement"]["hierarchy"])
        owners = list(_mesh_coordinates(len(hierarchy)))
        owners[partial_axis] = "attention_part"
        local = []
        for kind, extent, policy in zip(
            layout, _static_shape(abi, "logical_shape"),
            abi["distributed_type"]["axis_policies"], strict=True,
        ):
            coordinate = "0" if statistics and kind == "dim" else str(domain["global_by_kind"][kind])
            coordinate, axis_owners = _invert_split_axis(coordinate, str(extent), policy, hierarchy)
            local.append(coordinate)
            for axis, value in axis_owners:
                if axis != partial_axis:
                    owners[axis] = value
        return tuple(local), _linear_owner_expression(owners, hierarchy)

    max_coordinates, owner = state_access(max_abi, True)
    sum_coordinates, _ = state_access(partial_sum["abi"], True)
    accumulator_coordinates, _ = state_access(partial_accumulator["abi"], False)
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


def _paged_attention_gated_combine_call(raw) -> dict[str, object]:
    result = _paged_attention_combine_call(raw)
    gate = _buffer(raw, "inputs", "gate")
    output = _buffer(raw, "outputs", "result")
    layout = tuple(raw["semantic_attrs"]["layout"])
    domain = _attention_scalar_domain(output["abi"], layout, "attention_output_offsets")
    gate_offset, _ = _attention_result_access(gate["abi"], layout, domain)
    return {**result, "gate": _pointer(gate), "gate_offset": gate_offset,
            "output_dtype": _triton_dtype(str(output["abi"]["scalar_dtype"]))}


def _dsa_indexer_call(raw) -> dict[str, object]:
    q = _buffer(raw, "inputs", "q")
    key = _buffer(raw, "inputs", "key")
    gate = _buffer(raw, "inputs", "gate")
    weights = _buffer(raw, "inputs", "weights")
    state = tuple(_buffer(raw, "inputs", "state", index) for index in range(2))
    layer_id = _buffer(raw, "inputs", "layer_id")
    advance = _buffer(raw, "inputs", "advance_sequence")
    norm_weight = _buffer(raw, "inputs", "k_norm_weight")
    norm_bias = _buffer(raw, "inputs", "k_norm_bias")
    compress_ape = _buffer(raw, "inputs", "compress_ape")
    result = _buffer(raw, "outputs", "result")
    attrs = raw.get("semantic_attrs", {})
    q_shape = _static_shape(q["abi"], "logical_shape")
    key_shape = _static_shape(key["abi"], "logical_shape")
    state_shape = _static_shape(state[0]["abi"], "logical_shape")
    state_layer_ids = tuple(attrs.get("state_layer_ids", ()))
    if not state_layer_ids or len(state_layer_ids) != state_shape[0]:
        raise CodegenError(
            "DSA indexer codegen requires explicit state_layer_ids matching its cache; "
            "regenerate the imported IR with the state contract."
        )
    pool = int(attrs["index_kpool"])
    topk = int(attrs["index_topk"])
    output_width = topk + (pool - 1 if attrs.get("always_select_tail", False) else 0)
    output_tile = 1
    while output_tile < output_width:
        output_tile *= 2
    return {
        "q": _pointer(q),
        "key": _pointer(key),
        "gate": _pointer(gate),
        "weights": _pointer(weights),
        "state": _pointer(state[0]),
        "sequence_lengths": _pointer(state[1]),
        "layer_id": _scalar_expression(layer_id),
        "state_layer_ids": state_layer_ids,
        "advance_sequence": _scalar_expression(advance),
        "k_norm_weight": _pointer(norm_weight),
        "k_norm_bias": _pointer(norm_bias),
        "compress_ape": _pointer(compress_ape),
        "result": _pointer(result),
        "q_width": q_shape[-1],
        "index_dim": key_shape[-1],
        "index_heads": int(attrs["index_n_heads"]),
        "index_topk": topk,
        "pool": pool,
        "output_width": output_width,
        "output_tile": output_tile,
        "state_layers": state_shape[0],
        "state_capacity": state_shape[1],
        "state_tile": 1 << (state_shape[1] - 1).bit_length(),
        "pool_tile": 1 << (max(1, (state_shape[1] + pool - 1) // pool) - 1).bit_length(),
        "state_width": state_shape[2],
        "q_stride": int(q["abi"]["scalar_storage_strides"][-1]),
        "key_stride": int(key["abi"]["scalar_storage_strides"][-1]),
        "gate_stride": int(gate["abi"]["scalar_storage_strides"][-1]),
        "weights_stride": int(weights["abi"]["scalar_storage_strides"][-1]),
        "state_layer_stride": int(state[0]["abi"]["scalar_storage_strides"][0]),
        "state_token_stride": int(state[0]["abi"]["scalar_storage_strides"][1]),
        "state_value_stride": int(state[0]["abi"]["scalar_storage_strides"][2]),
        "ape_row_stride": int(compress_ape["abi"]["scalar_storage_strides"][0]),
        "ape_dim_stride": int(compress_ape["abi"]["scalar_storage_strides"][1]),
        "norm_stride": int(norm_weight["abi"]["scalar_storage_strides"][-1]),
        "writer_active": _chip_local_unique_writer_active(q["abi"]),
        "value_type": _triton_dtype(str(key["abi"]["scalar_dtype"])),
        "epsilon": repr(float(attrs["epsilon"])),
        "scale": repr(float(key_shape[-1] ** -0.5)),
    }


def _dsa_paged_attention_call(raw) -> dict[str, object]:
    query = _buffer(raw, "inputs", "q")
    topk = _buffer(raw, "inputs", "topk")
    state = tuple(_buffer(raw, "inputs", "state", index) for index in range(5))
    layer_id = _buffer(raw, "inputs", "layer_id")
    result = _buffer(raw, "outputs", "result")
    attrs = raw.get("semantic_attrs", {})
    query_shape = _static_shape(query["abi"], "logical_shape")
    cache_shape = _static_shape(state[0]["abi"], "logical_shape")
    query_lanes = int(query["abi"].get("scalar_lane_count", 1))
    cache_lanes = int(state[0]["abi"].get("scalar_lane_count", 1))
    query_dim = query_shape[-1] * query_lanes
    cache_dim = cache_shape[-1] * cache_lanes
    if query_dim != cache_dim:
        raise CodegenError("DSA paged attention query/cache scalar dimensions disagree.")
    schedule = str(raw.get("parameters", {}).get("decode_schedule", "simt"))
    if schedule not in {"simt", "bf16_mma"}:
        raise CodegenError(f"Unknown DSA decode_schedule {schedule!r}.")
    index_traversal = attrs.get("index_traversal", "slots")
    if not isinstance(index_traversal, str) or index_traversal not in {"slots", "compact"}:
        raise CodegenError("DSA index_traversal must be slots or compact.")
    if index_traversal == "compact" and schedule != "bf16_mma":
        raise CodegenError("DSA compact index_traversal requires the bf16_mma schedule.")
    if schedule == "bf16_mma":
        if (attrs.get("probability_rounding") != "online_bf16"
                or tuple(attrs["layout"]) != ("seq", "head", "dim")
                or query["abi"]["scalar_dtype"] != "bfloat16"
                or state[0]["abi"]["scalar_dtype"] != "bfloat16"
                or result["abi"]["scalar_dtype"] != "bfloat16"
                or len(query_shape) != 3 or query_shape[0] != 1
                or cache_shape[-2] != 1 or query_dim <= 0
                or query_dim > 512 or query_dim % 16
                or float(attrs["scale"]) <= 0):
            raise CodegenError("DSA bf16_mma requires online_bf16, seq/head/dim layout, one query token, one KV head, "
                               "BF16 buffers and head_dim a positive multiple of 16 up to 512.")
        for value in (query, topk, result):
            if _static_shape(value["abi"], "logical_shape") != _static_shape(value["abi"], "local_capacity_shape"):
                raise CodegenError("DSA bf16_mma requires complete local query/top-k/result domains.")
            distributed = value["abi"].get("distributed_type")
            if isinstance(distributed, Mapping) and (distributed.get("partial") is not None
                    or any(p.get("kind") == "split" for p in distributed.get("axis_policies", ()))):
                raise CodegenError("DSA bf16_mma does not implement partial or split buffers.")
        for value in (query, state[0], result):
            if int(value["abi"]["scalar_storage_strides"][-1]) != int(value["abi"].get("scalar_lane_count", 1)):
                raise CodegenError("DSA bf16_mma requires contiguous final scalar dimensions.")
    return {
        "query": _pointer(query),
        "topk": _pointer(topk),
        "kv_cache": _pointer(state[0]),
        "query_start_loc": _pointer(state[1]),
        "sequence_lengths": _pointer(state[2]),
        "slot_mapping": _pointer(state[3]),
        "block_table": _pointer(state[4]),
        "layer_id": _scalar_expression(layer_id),
        "result": _pointer(result),
        "num_query_heads": query_shape[-2],
        "query_value_type": _triton_dtype(str(query["abi"]["scalar_dtype"])),
        "num_kv_heads": cache_shape[-2],
        "head_dim": query_dim,
        "block_size": cache_shape[3],
        "num_layers": cache_shape[1],
        "topk_width": _static_shape(topk["abi"], "logical_shape")[-1],
        "index_traversal": index_traversal,
        "compact_topk_capacity": 1 << (max(_static_shape(topk["abi"], "logical_shape")[-1], 1) - 1).bit_length(),
        "query_head_stride": int(query["abi"]["scalar_storage_strides"][-2]),
        "query_dim_stride": int(query["abi"]["scalar_storage_strides"][-1]),
        "topk_stride": int(topk["abi"]["scalar_storage_strides"][-1]),
        "result_head_stride": int(result["abi"]["scalar_storage_strides"][-2]),
        "result_dim_stride": int(result["abi"]["scalar_storage_strides"][-1]),
        "cache_strides": tuple(int(value) for value in state[0]["abi"]["scalar_storage_strides"]),
        "scale": repr(float(attrs["scale"])),
        "token_tile": 64,
        "decode_schedule": schedule,
        "padded_head_dim": 1 << (query_dim - 1).bit_length(),
        "scale_log2": repr(float(attrs["scale"]) * 1.4426950408889634),
        "probability_rounding": str(attrs.get("probability_rounding", "normalized")),
        "writer_active": _canonical_writer_active(result["abi"]),
    }


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
    vocab_split_axes = set()
    if isinstance(distributed, Mapping):
        policies = distributed.get("axis_policies", ())
        vocab_policy = policies[1] if len(policies) > 1 else {}
        if vocab_policy.get("kind") == "split":
            for stage in vocab_policy.get("stages", ()):
                vocab_split_axes.update(int(axis) for axis in stage.get("hierarchy_axes", ()))
    levels = (
        str(distributed.get("placement", {}).get("hierarchy_levels", ""))
        if isinstance(distributed, Mapping) else ""
    ) or "b" * len(hierarchy)
    if any(levels[axis] in {"c", "d"} for axis in vocab_split_axes):
        raise CodegenError(
            "Greedy distributed_argmax cannot reduce a logits split across "
            "device/node placement axes without a remote partial reduction."
        )
    partial_owner_index, partial_owner_count = _chip_local_owner_group(
        logits_abi, prod(hierarchy, start=1),
    )
    block_axes = tuple(axis for axis, level in enumerate(levels) if level not in {"c", "d"})
    owner_coordinates = tuple(
        f"((greedy_owners // {prod(hierarchy[inner] for inner in block_axes if inner > axis)}) % {extent})"
        if axis in block_axes else _mesh_coordinate(axis, len(hierarchy))
        for axis, extent in enumerate(hierarchy)
    )
    return {
        "logits": _pointer(logits),
        "result": _pointer(result),
        "partial_max": _pointer(partial_max),
        "partial_index": _pointer(partial_index),
        "partial_owner_count": partial_owner_count,
        "partial_owner_index": partial_owner_index,
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
    row_coordinate: str | None = None,
) -> dict[str, object]:
    """Describe one scalarized row of a possibly-vector physical tensor."""

    shape = _static_shape(abi, "local_capacity_shape")
    if not shape or (row_coordinate is None and prod(shape[:-1], start=1) != 1):
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
    row_coordinates = (tuple("0" for _ in shape[:-1]) if row_coordinate is None else
                       _unflattened_coordinates(shape[:-1], row_coordinate))
    coordinates = (*row_coordinates, physical)
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
        f"({coordinate} < ({active_extents[axis]}))"
        for axis, coordinate in enumerate(row_coordinates)
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
        lane_shape = tuple(int(value) for value in abi.get("scalar_lane_shape", ()))
        if lane_shape:
            if len(lane_shape) != 1:
                raise CodegenError(
                    "TIR DenseMatMul generic vector RHS must have one N-axis lane dimension."
                )
            lane_count = lane_shape[0]
            n_coordinate = f"(({n_coordinate}) // {lane_count})"
            lane_coordinate = f"(({global_n}) % {lane_count})"
        else:
            lane_coordinate = None
        coordinates = (
            (n_coordinate, k_coordinate)
            if transpose_b
            else (k_coordinate, n_coordinate)
        )
        return pointer, emit_offset(abi, coordinates, lane_coordinate=lane_coordinate)

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
    strides = tuple(int(value) for value in abi["scalar_storage_strides"])
    if len(strides) != 4:
        raise CodegenError(
            "Scalar-physical k_major_n8_k16 RHS stride rank disagrees with its "
            "logical shape."
        )
    if strides[2] == payload_width * strides[3]:
        # The two trailing axes only carry one 128-element payload slot, and
        # here they are contiguous, so splitting the slot across them is the
        # identity: `(p // w) * (w * s) + (p % w) * s == p * s` for
        # `0 <= p < 2 * w`. Emit the collapsed form. The split form is
        # value-identical but not code-identical: Triton cannot fold `//`/`%`
        # back through the stride multiply, so the weight load loses its
        # provable contiguity and stops being treated as a contiguous stream.
        # On BI-V150 the split form costs ~5 ms per token, dominated by the
        # GLU and QKV weight traffic.
        offset = emit_offset(
            abi,
            (
                f"({k_coordinate}) // 16",
                f"({n_coordinate}) // 8",
                "0",
                payload,
            ),
        )
        zero_term = f"(0) * {strides[2]} + "
        if zero_term not in offset:
            raise CodegenError(
                "Collapsed k_major_n8_k16 payload addressing did not emit the "
                "expected zero group term."
            )
        return pointer, offset.replace(zero_term, "", 1)
    return pointer, emit_offset(
        abi,
        (
            f"({k_coordinate}) // 16",
            f"({n_coordinate}) // 8",
            f"({payload}) // {payload_width}",
            f"({payload}) % {payload_width}",
        ),
    )


def _k_major_n8_k16_k_contiguity(
    weight_abi: Mapping[str, object],
    source_abi: Mapping[str, object],
    *,
    local_k_capacity: int,
    block_k: int,
) -> int:
    """Return 16 when every 16-aligned K run of a k_major_n8_k16 weight tile is
    provably one contiguous, 16-element-aligned run of storage, else 1.

    The K coordinate used to address canonical storage is checked by
    evaluating the source's local-to-logical mapping for every owner, so a
    template may only assert this to the compiler when it actually holds.
    """

    run = 16
    if local_k_capacity % run or block_k % run:
        return 1
    lanes = tuple(int(value) for value in weight_abi.get("scalar_lane_shape", ()))
    if lanes:
        if lanes != (8, 2, 8):
            return 1
    else:
        shape = _static_shape(weight_abi, "logical_shape")
        strides = tuple(int(value) for value in weight_abi.get("scalar_storage_strides", ()))
        if (len(shape) != 4 or len(strides) != 4 or strides[3] != 1
                or strides[2] != shape[3] or strides[0] % run or strides[1] % run):
            return 1
    coordinate_space = str(weight_abi.get("coordinate_space"))
    if coordinate_space == "local":
        return run
    if coordinate_space != "canonical_global":
        return 1
    if int(source_abi.get("scalar_lane_count", 1)) != 1:
        return 1
    rank = len(_static_shape(source_abi, "local_capacity_shape"))
    distributed = source_abi.get("distributed_type")
    placement = distributed.get("placement") if isinstance(distributed, Mapping) else None
    hierarchy = (
        tuple(int(value) for value in placement.get("hierarchy", ()))
        if isinstance(placement, Mapping) else ()
    )
    shard_names = tuple(f"__shard{axis}" for axis in range(len(hierarchy)))
    try:
        expression = emit_logical_coordinate(
            source_abi, rank - 1, (*("0",) * (rank - 1), "__k"),
            shard_coordinates=shard_names or None,
        )
        code = compile(expression, "<k_major_n8_k16 K coordinate>", "eval")
        for shard in itertools.product(*(range(extent) for extent in hierarchy)):
            names = {"__builtins__": {}, **dict(zip(shard_names, shard))}
            for start in range(0, local_k_capacity, run):
                first = eval(code, names, {"__k": start})
                if first % run:
                    return 1
                for offset in range(1, run):
                    if eval(code, names, {"__k": start + offset}) != first + offset:
                        return 1
    except Exception:
        return 1
    return run


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


def _exclusive_participation_active(raw: Mapping[str, object]) -> str:
    """Return the owner predicate for a call touching an E value.

    Data ownership can restrict ordinary calls. A call containing an internal
    grid rendezvous instead needs every block of each participating chip;
    its buffer domains and writer masks still enforce exclusive data access.
    Chip/node exclusivity remains valid because a grid barrier is chip-local.
    """

    exclusive = None
    for section in ("inputs", "outputs", "workspaces"):
        for parameter in raw.get(section, ()):
            for binding in parameter.get("buffers", ()):
                distributed = binding.get("abi", {}).get("distributed_type")
                if not isinstance(distributed, Mapping):
                    continue
                candidate = distributed.get("exclusive")
                if isinstance(candidate, Mapping):
                    if exclusive is None:
                        exclusive = candidate
                    elif exclusive != candidate:
                        raise CodegenError("One kernel call cannot mix different E owner predicates.")
    if exclusive is None:
        return "True"
    placement = None
    # The serialized E contract is nested in the DistributedType placement.
    for section in ("inputs", "outputs", "workspaces"):
        for parameter in raw.get(section, ()):
            for binding in parameter.get("buffers", ()):
                distributed = binding.get("abi", {}).get("distributed_type")
                if isinstance(distributed, Mapping) and distributed.get("exclusive") == exclusive:
                    placement = distributed.get("placement")
                    break
            if placement is not None:
                break
        if placement is not None:
            break
    if not isinstance(placement, Mapping):
        raise CodegenError("E call has no placement ABI.")
    hierarchy = tuple(int(value) for value in placement.get("hierarchy", ()))
    axes = tuple(int(value) for value in exclusive.get("axes", ()))
    owners = exclusive.get("owner_coordinates")
    if owners is None:
        owners = (0,) * len(axes)
    owners = tuple(int(value) for value in owners)
    if not axes or len(axes) != len(owners) or any(axis < 0 or axis >= len(hierarchy) for axis in axes):
        raise CodegenError("E call has invalid owner axes.")
    selected = tuple(zip(axes, owners, strict=True))
    if raw.get("facts", {}).get("internal_grid_barriers", 0):
        levels = str(placement.get("hierarchy_levels", "")) or "b" * len(hierarchy)
        if len(levels) != len(hierarchy):
            raise CodegenError("E call has inconsistent placement levels.")
        selected = tuple((axis, owner) for axis, owner in selected if levels[axis] in {"c", "d"})
    return " & ".join(
        f"({_mesh_coordinate(axis, len(hierarchy))} == {owner})"
        for axis, owner in selected
    ) or "True"


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
    *,
    preserved_coordinates: tuple[str, ...] | None = None,
    allow_empty: bool = False,
) -> str:
    """Map a dense group member to a placement owner, preserving other axes.

    With ``allow_empty`` the reduction group may span no axes (every placement
    axis preserved): the member is unused and the expression resolves to this
    CTA's own owner, so the degenerate one-element reduction reads its own
    partial.
    """

    distributed = abi.get("distributed_type")
    if not isinstance(distributed, Mapping):
        raise CodegenError("Owner group buffer has no DistributedType ABI.")
    placement = distributed.get("placement")
    if not isinstance(placement, Mapping):
        raise CodegenError("Owner group buffer has no placement ABI.")
    hierarchy = tuple(int(value) for value in placement.get("hierarchy", ()))
    if not axes:
        if not allow_empty:
            raise CodegenError(
                "Owner group axes must be sorted, unique, and inside the placement."
            )
        # No axis is reduced (all preserved): the one-element reduction group
        # reads this CTA's own partial. Broadcast the constant self-owner over
        # the (masked) reduction lane so the gather load keeps a block pointer
        # that matches its block mask -- every lane resolves to the same valid
        # self address and only lane 0 survives the mask.
        self_owner = _linear_owner_expression(
            list(preserved_coordinates or _mesh_coordinates(len(hierarchy))),
            hierarchy,
        )
        return f"({self_owner} + 0 * ({member}))"
    if tuple(sorted(set(axes))) != axes or any(
        axis < 0 or axis >= len(hierarchy) for axis in axes
    ):
        raise CodegenError(
            "Owner group axes must be sorted, unique, and inside the placement."
        )
    if axes == tuple(range(len(hierarchy))):
        return f"({member})"
    if len(hierarchy) == 2 and preserved_coordinates is None:
        if axes == (0,):
            return f"(({member}) * {hierarchy[1]} + shard_x)"
        if axes == (1,):
            return f"(shard_y * {hierarchy[1]} + ({member}))"
    coordinates = list(
        _mesh_coordinates(len(hierarchy))
        if preserved_coordinates is None else preserved_coordinates
    )
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
    return _chip_local_unique_writer_active(abi)


def _reduced_destination_writer_active(source_abi: Mapping[str, object]) -> str:
    """Elect writers for a destination this leaf has already reduced.

    Unlike :func:`_chip_local_unique_writer_active`, which must keep a partial
    chip axis as a writer because a partial is not a replica, this leaf stores
    the reduced value: the chip axes it did not split now hold identical copies
    and each of them owns a private instance of the destination pool, so every
    one of them has to publish into its own instance.
    """

    distributed = source_abi.get("distributed_type")
    if not isinstance(distributed, Mapping):
        return "True"
    placement = distributed.get("placement", {})
    levels = str(placement.get("hierarchy_levels", ""))
    split = {
        int(axis)
        for policy in distributed.get("axis_policies", ())
        if policy.get("kind") == "split"
        for stage in policy.get("stages", ())
        for axis in stage.get("hierarchy_axes", ())
    }
    # Unlike a materialized source, a reduced value carries no partial, so its
    # chip axes are replicas of the stored value no matter whether the
    # reduction or a broadcast made them so.
    replicas = tuple(
        axis
        for axis, level in enumerate(levels)
        if level in {"c", "d"} and axis not in split
    )
    return _distributed_unique_writer_active(
        source_abi, allow_redundant_axes=replicas
    )


def _chip_local_unique_writer_active(abi: Mapping[str, object]) -> str:
    """Elect writers per allocation, preserving distinct logical shards."""
    distributed = abi.get("distributed_type")
    if not isinstance(distributed, Mapping):
        return "True"
    placement = distributed.get("placement", {})
    levels = str(placement.get("hierarchy_levels", ""))
    used = {
        int(axis)
        for policy in distributed.get("axis_policies", ())
        if policy.get("kind") == "split"
        for stage in policy.get("stages", ())
        for axis in stage.get("hierarchy_axes", ())
    }
    # A chip axis carrying a partial is not a replica. It requires a remote
    # reduction rather than independently publishing each rank's component.
    used.update((distributed.get("partial") or {}).get("axes", ()))
    replicas = tuple(axis for axis, level in enumerate(levels) if level in {"c", "d"} and axis not in used)
    return _distributed_unique_writer_active(abi, allow_redundant_axes=replicas)


def _distributed_unique_writer_active(
    abi: Mapping[str, object], *, writer_ordinal: int = 0,
    allow_redundant_axes: tuple[int, ...] = (),
) -> str:
    """Elect one writer among the axes that hold redundant copies of a value.

    ``allow_redundant_axes`` names axes whose owners all publish the same value
    and may therefore write it concurrently.
    """

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
    abi: Mapping[str, object],
    flat_coordinate: str,
    *,
    row_coordinate: str | None = None,
    reduction_axis: int | None = None,
) -> dict[str, object]:
    """Build an owner-local domain whose induction variable counts scalars.

    ``local_capacity_shape`` describes physical vector elements.  Kernels that
    perform scalar arithmetic must additionally iterate the element-type lanes
    and pass the lane coordinate to physical addressing.

    With ``row_coordinate`` the domain splits the leading rows (indexed by the
    caller's ``row_coordinate`` loop variable) from a flattened reduction suffix
    (``capacity`` covers one such suffix). The split falls at ``reduction_axis``
    -- the last axis when unset (single-axis normalization), or an earlier axis
    for suffix normalization, so the whole ``shape[reduction_axis:]`` contributes
    to one row's reduction. ``_unflattened_coordinates`` is the identity on a
    single-axis suffix, so the default stays byte-identical to the last-axis form.
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
    if row_coordinate is not None:
        split = len(shape) - 1 if reduction_axis is None else reduction_axis
        row_coordinates = _unflattened_coordinates(shape[:split], row_coordinate)
        suffix_coordinates = _unflattened_coordinates(shape[split:], physical_flat)
        local_coordinates = (*row_coordinates, *suffix_coordinates)
        active = " & ".join(
            f"(({coordinate}) < ({emit_active_extent(abi, axis)}))"
            for axis, coordinate in enumerate(local_coordinates)
        ) or "True"
        logical_coordinates = tuple(
            emit_logical_coordinate(abi, axis, local_coordinates)
            for axis in range(len(shape))
        )
        return {
            "capacity": prod(shape[split:], start=1) * lane_count,
            "local_coordinates": local_coordinates,
            "logical_coordinates": logical_coordinates,
            "lane_coordinate": lane_coordinate,
            "active": active,
        }
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
        distributed = operand_abi.get("distributed_type")
        if (
            isinstance(distributed, Mapping)
            and distributed.get("kind") == "distributed"
            and tuple(operand_abi["logical_shape"]) == tuple(result_abi["logical_shape"])
            and is_fully_replicated(type_from_data(distributed))
        ):
            # Every owner already contains the complete logical tensor. A
            # consumer's narrower shard therefore indexes this private copy
            # by logical coordinates; it does not require a communication op.
            return emit_local_scalar_offset(
                operand_abi, domain["logical_coordinates"], lane_coordinate=lane_coordinate,
            )
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
    "delta_rule_coefficients": delta_rule_coefficients_call,
    "delta_rule_log_prefix": delta_rule_log_prefix_call,
    "delta_rule_block_update": delta_rule_block_update_call,
    "delta_rule_gates": delta_rule_gates_call,
    "l2_normalization": l2_normalization_call,
    "pack": vector_relayout_call,
    "concat": concat_call,
    "broadcast_to": broadcast_to_call,
    "softmax": softmax_call,
    "reduce_sum": reduce_sum_call,
    "top_k": top_k_call,
    "staged_routing": routing_call,
    "kda_convolution": kda_convolution_call,
    "kda_recurrent_core": kda_recurrent_call,
    "kda_gated_recurrent_core": kda_gated_recurrent_call,
    "mhc_pre": mhc_pre_call,
    "mhc_post": mhc_post_call,
    "sparse_experts_gate_up": sparse_experts_gate_up_call,
    "sparse_experts_down": sparse_experts_down_call,
    "partitioned_sparse_experts_down": sparse_experts_down_call,
    "sparse_experts_dispatch": sparse_experts_routes_call,
    "sparse_experts_weighted_sum": sparse_experts_routes_call,
    "unpack": vector_relayout_call,
    "pad": tensor_transform_call,
    "slice": tensor_transform_call,
    "gather": gather_call,
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
    "gated_norm_apply": _gated_norm_apply_call,
    "matmul_glu": _matmul_glu_call,
    "elementwise": _elementwise_call,
    "norm_stats": _norm_stats_call,
    "dense_matmul": _dense_matmul_call,
    "dense_matmul_glu": _dense_matmul_glu_call,
    "qkv_parallel_linear": _qkv_parallel_linear_call,
    "qkv_rope_with_cache": _qkv_rope_with_cache_call,
    "gather_reduce_qkv_rope_with_cache": _gather_reduce_qkv_statistics_call,
    "rotary_embedding": _rotary_embedding_call,
    "rope": _rope_call,
    "update_paged_attention_kv_cache": _cache_update_call,
    "update_paged_attention_kv_cache_pair": _cache_update_call,
    "paged_attention_partial": _paged_attention_partial_call,
    "paged_attention_combine": _paged_attention_combine_call,
    "paged_attention_gated_combine": _paged_attention_gated_combine_call,
    "dsa_indexer": _dsa_indexer_call,
    "dsa_paged_attention": _dsa_paged_attention_call,
    "mla_projection": mla_projection_call,
    "greedy_sample": _greedy_sample_call,
}

__all__ = ["prepare_kernel_calls"]

# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Renderers for the KDA decode operators (GLM-5.3-Flash).

The templates address scalar elements using ABI strides, including all vector
lanes in packed activation outputs and per-head offsets into backing storage.
"""

from math import prod

from triton.flagmega.codegen.triton.physical_access import emit_active_extent, emit_local_scalar_offset
from triton.flagmega.errors import CodegenError


def _abi_strides(abi):
    return tuple(int(value) for value in abi["scalar_storage_strides"])


def _head_offset(abi):
    """Map the local head through each buffer's ownership and storage ABI."""
    rank = len(abi["local_capacity_shape"])
    return emit_local_scalar_offset(abi, ("0", "_fm_head", *("0" for _ in range(rank - 2))))


def _verify_recurrent_output(raw, out, state):
    lanes = tuple(raw["semantic_attrs"].get("output_lanes", ()))
    abi = out["abi"]
    shape = tuple(int(dim) for dim in state["abi"]["local_capacity_shape"])
    lane_count = prod(lanes)
    if (tuple(abi.get("scalar_lane_shape", ())) != lanes
            or int(abi.get("scalar_lane_count", 1)) != lane_count
            or abi["scalar_dtype"] != "bfloat16"
            or lane_count <= 0 or shape[-1] % lane_count
            or tuple(abi["local_capacity_shape"]) != (1, shape[1], shape[-1] // lane_count)
            or _abi_strides(abi)[-1] != lane_count):
        raise CodegenError("KDA activation output ABI disagrees with output_lanes or recurrent state.")


def kda_convolution_call(raw):
    """Render the part-separate, head-major depthwise convolution.

    The three parts own separate input and output buffers, so the kernel walks
    one ``[1, heads, head_dim]`` domain and unrolls the part axis.  State and
    weight addressing goes through the buffer ABI rather than plain
    local-capacity strides: the conv state can alias an external output whose
    shard view disagrees with them.
    """

    from triton.flagmega.codegen.triton.kernel_call_renderers import (
        _buffer,
        _canonical_writer_active,
        _pointer,
        _scalar_local_domain,
    )

    state = _buffer(raw, "inputs", "conv_state")
    weight = _buffer(raw, "inputs", "conv_weight")
    new_state = _buffer(raw, "outputs", "result_3")
    names = ("query", "key", "value")
    inputs = tuple(_buffer(raw, "inputs", name) for name in names)
    outputs = tuple(_buffer(raw, "outputs", f"result_{index}") for index in range(3))
    domain = _scalar_local_domain(inputs[0]["abi"], "_fm_offsets")
    capacity = domain["capacity"]
    tile = 1 << (max(capacity, 1) - 1).bit_length()
    kernel = int(raw["semantic_attrs"]["kernel_size"])
    head, channel = domain["local_coordinates"][1], domain["local_coordinates"][2]
    parts = tuple({
        "value": _pointer(source),
        "value_offset": emit_local_scalar_offset(source["abi"], domain["local_coordinates"]),
        "result": _pointer(sink),
        "result_offset": emit_local_scalar_offset(sink["abi"], domain["local_coordinates"]),
        "result_writer_active": _canonical_writer_active(sink["abi"]),
        "weight_offsets": [
            emit_local_scalar_offset(weight["abi"], (str(part), head, channel, str(column)))
            for column in range(kernel)
        ],
        "state_offsets": [
            emit_local_scalar_offset(state["abi"], (str(part), head, channel, str(column)))
            for column in range(kernel - 1)
        ],
        "new_state_offsets": [
            emit_local_scalar_offset(new_state["abi"], (str(part), head, channel, str(column)))
            for column in range(kernel - 1)
        ],
    } for part, (source, sink) in enumerate(zip(inputs, outputs, strict=True)))
    return {
        **domain,
        "tile": min(tile, 1024),
        "internal_grid_barriers": 0,
        "kernel": kernel,
        "parts": parts,
        "state": _pointer(state),
        "weight": _pointer(weight),
        "new_state": _pointer(new_state),
        "active": domain["active"],
        "writer_active": _canonical_writer_active(new_state["abi"]),
    }


def kda_recurrent_call(raw):
    from triton.flagmega.codegen.triton.kernel_call_renderers import (
        _buffer,
        _canonical_writer_active,
        _pointer,
    )

    query = _buffer(raw, "inputs", "query")
    key = _buffer(raw, "inputs", "key")
    value = _buffer(raw, "inputs", "value")
    decay = _buffer(raw, "inputs", "decay")
    beta = _buffer(raw, "inputs", "beta")
    state = _buffer(raw, "inputs", "state")
    out = _buffer(raw, "outputs", "result_0")
    new_state = _buffer(raw, "outputs", "result_1")
    _verify_recurrent_output(raw, out, state)
    state_shape = tuple(int(dim) for dim in state["abi"]["local_capacity_shape"])
    return {
        "internal_grid_barriers": 0,
        "heads": emit_active_extent(state["abi"], 1),
        "key_dim": state_shape[2],
        "value_dim": state_shape[3],
        "value_chunks": (state_shape[3] + 31) // 32,
        "epsilon": repr(float(raw["semantic_attrs"]["epsilon"])),
        "l2norm_in_kernel": bool(raw["semantic_attrs"]["l2norm_in_kernel"]),
        "query": _pointer(query),
        "key": _pointer(key),
        "value": _pointer(value),
        "decay": _pointer(decay),
        "beta": _pointer(beta),
        "state": _pointer(state),
        "out": _pointer(out),
        "new_state": _pointer(new_state),
        "query_head_offset": _head_offset(query["abi"]),
        "key_head_offset": _head_offset(key["abi"]),
        "value_head_offset": _head_offset(value["abi"]),
        "decay_head_offset": _head_offset(decay["abi"]),
        "beta_head_offset": _head_offset(beta["abi"]),
        "state_head_offset": _head_offset(state["abi"]),
        "out_head_offset": _head_offset(out["abi"]),
        "new_state_head_offset": _head_offset(new_state["abi"]),
        "out_writer_active": _canonical_writer_active(out["abi"]),
        "state_writer_active": _canonical_writer_active(new_state["abi"]),
    }


def kda_gated_recurrent_call(raw):
    """Like ``kda_recurrent_call``, plus the raw gate/beta operands.

    The forget gate's weight maps (``decay_rate`` = exp(A_log) broadcast,
    ``gate_bias`` = dt_bias in head layout) are ordinary ``[1, H, D]`` buffers,
    so they address through the same per-head stride as ``decay`` did.
    """

    from triton.flagmega.codegen.triton.kernel_call_renderers import (
        _buffer,
        _canonical_writer_active,
        _pointer,
    )

    buffers = {
        name: _buffer(raw, "inputs", name)
        for name in ("query", "key", "value", "gate", "decay_rate", "gate_bias",
                     "beta_logits", "state")
    }
    out = _buffer(raw, "outputs", "result_0")
    new_state = _buffer(raw, "outputs", "result_1")
    _verify_recurrent_output(raw, out, buffers["state"])
    state_shape = tuple(int(dim) for dim in buffers["state"]["abi"]["local_capacity_shape"])
    return {
        "internal_grid_barriers": 0,
        "heads": emit_active_extent(buffers["state"]["abi"], 1),
        "key_dim": state_shape[2],
        "value_dim": state_shape[3],
        "value_chunks": (state_shape[3] + 31) // 32,
        "epsilon": repr(float(raw["semantic_attrs"]["epsilon"])),
        "lower_bound": repr(float(raw["semantic_attrs"]["lower_bound"])),
        "l2norm_in_kernel": bool(raw["semantic_attrs"]["l2norm_in_kernel"]),
        **{name: _pointer(buffer) for name, buffer in buffers.items()},
        **{f"{name}_head_offset": _head_offset(buffer["abi"])
           for name, buffer in buffers.items()},
        "out": _pointer(out),
        "new_state": _pointer(new_state),
        "out_head_offset": _head_offset(out["abi"]),
        "new_state_head_offset": _head_offset(new_state["abi"]),
        "out_writer_active": _canonical_writer_active(out["abi"]),
        "state_writer_active": _canonical_writer_active(new_state["abi"]),
    }

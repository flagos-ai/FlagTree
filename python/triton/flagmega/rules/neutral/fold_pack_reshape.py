# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Move Pack through a reshape using nncase's complete shape map."""

from triton.flagmega.ir import IRModule, Node, get_definition, try_div_exactly
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.tensors.pack import axis_lane_products, normalize_axes
from triton.flagmega.ir.ops.tensors.reshape import (
    _complete_shape_map,
    _reshape_shape_map_matrix,
)
from triton.flagmega.pattern_match import F, MatchResult, wildcard
from triton.flagmega.rules import RewriteResult, RewriteRule

_INPUT = wildcard("input")
_PATTERN = F.tensors.is_pack(
    F.tensors.is_reshape(_INPUT, call_name="reshape"),
    call_name="pack",
)


def _fixed_shape(node: Node) -> tuple[int, ...] | None:
    shape = tensor_of(node.type).shape
    if any(not dimension.is_fixed for dimension in shape):
        return None
    return tuple(dimension.fixed_value for dimension in shape)


def _plan(pack: Node, reshape: Node, source: Node):
    input_shape = _fixed_shape(source)
    output_shape = _fixed_shape(reshape)
    if input_shape is None or output_shape is None:
        return None
    matrix = _reshape_shape_map_matrix(input_shape, output_shape)
    if matrix is None:
        return None
    forward, backward = _complete_shape_map(matrix)
    lanes = tuple(int(value) for value in pack.attrs["lanes"])
    raw_axes = (
        tuple(int(value) for value in pack.attrs["axes"])
        if "axes" in pack.attrs
        else (int(pack.attrs["axis"]),) * len(lanes)
    )
    axes = normalize_axes(raw_axes, len(output_shape))
    mapped_axes: dict[int, int] = {}
    rewritten_shape = list(output_shape)

    # This is the same two-sided complete-map algorithm as nncase's
    # FoldPackReshape.  A lane may move through a split/merge only when it is
    # on the final effective row-major axis of that mapping.
    # Pack divides an axis by the product of all its lanes. Plan that group
    # once: independently processing repeated axes overwrites the new extent
    # and can also skip an input axis that still owns subsequent lane factors.
    for axis, lane in axis_lane_products(lanes, axes).items():
        quotient = try_div_exactly(tensor_of(reshape.type).shape[axis], lane)
        if quotient is None or not quotient.is_fixed:
            return None
        rewritten_shape[axis] = quotient.fixed_value
        for input_axis, output_axes in forward.items():
            if axis not in output_axes:
                continue
            position = output_axes.index(axis)
            if position != len(output_axes) - 1 and any(
                output_shape[value] != 1 for value in output_axes[position + 1:]
            ):
                return None
            mapped_axes[axis] = input_axis
            break

        if axis in mapped_axes:
            continue

        mapped_inputs = backward.get(axis)
        if mapped_inputs is None:
            continue
        for input_axis in reversed(mapped_inputs):
            if input_shape[input_axis] == 1:
                continue
            if input_shape[input_axis] % lane:
                return None
            mapped_axes[axis] = input_axis
            break
        if axis not in mapped_axes:
            return None

    if any(axis not in mapped_axes for axis in axes):
        return None
    return tuple(mapped_axes[axis] for axis in axes), lanes, tuple(rewritten_shape)


def _rewrite(result: MatchResult, _module: IRModule) -> RewriteResult:
    pack = result["pack"]
    reshape = result["reshape"]
    source = result["input"]
    assert all(isinstance(value, Node) for value in (pack, reshape, source))
    plan = _plan(pack, reshape, source)
    if plan is None:
        return RewriteResult(pack)
    input_axes, input_lanes, rewritten_shape = plan

    pack_definition = get_definition("tensors.pack")
    prepared_pack = pack_definition.prepare(
        (source,), {"lanes": input_lanes, "axes": input_axes}
    )
    packed = Node(
        f"{pack.id}.fold_pack_reshape.pack",
        "tensors.pack",
        tuple(value.id for value in prepared_pack.inputs),
        prepared_pack.result_type,
        prepared_pack.effect,
        prepared_pack.attrs,
        {"rewritten_by": "FoldPackReshape"},
    )
    reshape_definition = get_definition("tensors.reshape")
    prepared_reshape = reshape_definition.prepare(
        (packed,), {"shape": rewritten_shape}
    )
    if prepared_reshape.result_type != pack.type:
        return RewriteResult(pack)
    replacement = Node(
        pack.id,
        "tensors.reshape",
        (packed.id,),
        prepared_reshape.result_type,
        prepared_reshape.effect,
        prepared_reshape.attrs,
        {**dict(pack.metadata), "rewritten_by": "FoldPackReshape"},
    )
    return RewriteResult(replacement, (packed,))


def fold_pack_reshape_rule() -> RewriteRule:
    return RewriteRule("FoldPackReshape", pattern=_PATTERN, rewrite=_rewrite)


__all__ = ["fold_pack_reshape_rule"]

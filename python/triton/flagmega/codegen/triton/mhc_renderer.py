# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Renderers for the GLM-5 hyper-connection (mHC) stages.

Both stages are replicated by contract: the pre stage's RMS term and its
coefficient matvec reduce over every ``hc_mult * hidden`` stream element, and
the post stage contracts the stream axis against the mixing matrix.  Neither
has an owner-local form, so the renderers reject a split or partial operand
instead of silently emitting arithmetic over one shard -- which is also what
``MhcPre.infer_type`` and ``MhcPost.infer_type`` already demand of the types.

The coefficient groups are contiguous row ranges of the projection weight
(``pre`` = ``[0, hc)``, ``post`` = ``[hc, 2hc)``, ``comb`` = ``[2hc, 2hc+hc**2)``),
which is what lets one pass over the streams produce all three groups while the
accumulator stays in registers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import prod

from triton.flagmega.codegen.triton.physical_access import (
    emit_active_extent,
    emit_local_scalar_offset,
    emit_triton_scalar_type,
)
from triton.flagmega.errors import CodegenError

# Register budget for one staged tile, in scalars.  The pre stage stages a
# ``[coefficient_lanes, tile]`` weight block and the post stage a
# ``[hc, hc, tile]`` product, so the tile has to shrink as those grow.
_TILE_SCALARS = 4096
_MIN_TILE = 16


def _lanes(extent: int) -> int:
    """Smallest power-of-two ``tl.arange`` width that covers ``extent``."""

    return 1 << max(int(extent) - 1, 0).bit_length()


def _tile(requested: int, rows: int) -> int:
    """Largest power-of-two tile keeping ``rows * tile`` inside the budget.

    ``tl.arange`` only accepts a power-of-two extent, so the budget limit is
    rounded down to one rather than used as a raw division result.
    """

    budget = max(_TILE_SCALARS // max(int(rows), 1), _MIN_TILE)
    limit = 1 << (budget.bit_length() - 1)
    return max(min(_lanes(requested), limit), 1)


def _replicated_abi(binding: Mapping[str, object], name: str, *, scalar: bool = True) -> Mapping[str, object]:
    """Return the buffer ABI, refusing an operand this kernel cannot reduce."""

    abi = binding["abi"]
    distributed = abi.get("distributed_type")
    if isinstance(distributed, Mapping):
        if distributed.get("partial") is not None:
            raise CodegenError(
                f"mHC operand {name!r} is partial; the stage reads every element of it, "
                "so the partial must be reduced by an explicit Boxing first.")
        if any(isinstance(policy, Mapping) and str(policy.get("kind")) == "split"
               for policy in distributed.get("axis_policies", ())):
            raise CodegenError(
                f"mHC operand {name!r} is split; the stage reduces over the stream and "
                "hidden axes, so it requires a replicated operand.")
    if scalar and abi.get("scalar_lane_shape"):
        raise CodegenError(
            f"mHC operand {name!r} carries vector lanes; the stages address scalars.")
    return abi


def _capacity(abi: Mapping[str, object], name: str) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in abi["local_capacity_shape"])
    except (KeyError, TypeError, ValueError) as error:
        raise CodegenError(f"mHC operand {name!r} needs a static local capacity.") from error


def _bind(raw, groups: Mapping[str, Sequence[str]]) -> dict[str, Mapping[str, object]]:
    from triton.flagmega.codegen.triton.kernel_call_renderers import _buffer

    return {formal.split(":")[0]: _buffer(raw, group, formal.split(":")[-1])
            for group, formals in groups.items() for formal in formals}


def _rounding_dtype(attrs, name, default=None):
    value = attrs.get(name)
    if value is None:
        return default
    if not isinstance(value, str) or value not in {"float16", "bfloat16", "float32"}:
        raise CodegenError(f"mHC {name} must be float16, bfloat16 or float32.")
    return value


def mhc_pre_call(raw):
    """Render coefficient generation, Sinkhorn normalization and the collapse.

    Pass one walks the flattened streams once, accumulating the RMS square sum
    and the full ``[coefficients]`` matvec.  The Sinkhorn iteration then runs
    entirely in registers on the ``hc x hc`` matrix -- the reason this is one
    operation instead of two hundred graph nodes.  Pass two applies the ``pre``
    coefficients to collapse the streams into the sublayer input.
    """

    from triton.flagmega.codegen.triton.kernel_call_renderers import (
        _canonical_writer_active,
        _pointer,
    )

    bindings = _bind(raw, {
        "inputs": ("streams", "projection", "base", "scale"),
        "outputs": ("post:result_0", "comb:result_1", "collapsed:result_2"),
    })
    abis = {name: _replicated_abi(binding, name, scalar=name != "collapsed")
            for name, binding in bindings.items()}

    tokens, hc, hidden = _capacity(abis["streams"], "streams")
    lanes = tuple(raw["semantic_attrs"].get("output_lanes", ()))
    lane_count = prod(lanes)
    collapsed_abi = abis["collapsed"]
    collapsed_dtype = _rounding_dtype(raw["semantic_attrs"], "collapsed_dtype", abis["streams"]["scalar_dtype"])
    collapse_input_dtype = _rounding_dtype(raw["semantic_attrs"], "collapse_input_dtype")
    collapse_reduction = raw["semantic_attrs"].get("collapse_reduction", "sum")
    if not isinstance(collapse_reduction, str) or collapse_reduction not in {"sum", "sequential_fma"}:
        raise CodegenError("mHC collapse_reduction must be sum or sequential_fma.")
    if (tuple(collapsed_abi.get("scalar_lane_shape", ())) != lanes
            or int(collapsed_abi.get("scalar_lane_count", 1)) != lane_count
            or collapsed_abi["scalar_dtype"] != collapsed_dtype
            or lane_count <= 0 or hidden % lane_count):
        raise CodegenError("mHC collapsed output ABI disagrees with output_lanes or collapsed dtype.")
    projection_reduction = raw["semantic_attrs"].get("projection_reduction", "parallel")
    splits = raw["semantic_attrs"].get("projection_splits", 1)
    variants = {"parallel": "local", "stream_fma_128": "stream_fma", "contiguous_tf32": "contiguous_tf32"}
    if projection_reduction not in variants:
        raise CodegenError("mHC projection_reduction must be parallel, stream_fma_128 or contiguous_tf32.")
    if type(splits) is not int or splits <= 0 or splits & (splits - 1):
        raise CodegenError("mHC projection_splits must be a positive power of two.")
    variant = variants[projection_reduction]
    if raw.get("variant", "local") != variant:
        raise CodegenError("mHC selected variant does not implement its projection_reduction contract.")
    if (projection_reduction == "parallel" and splits != 1
            or projection_reduction == "stream_fma_128" and hidden % (128 * splits)
            or projection_reduction == "contiguous_tf32" and (not 1 <= hc <= 4 or hc * hidden % (64 * splits))):
        raise CodegenError("mHC projection_splits disagrees with its algorithm or hidden capacity.")
    coefficients = (2 + hc) * hc
    if _capacity(abis["projection"], "projection") != (coefficients, hc * hidden):
        raise CodegenError(
            f"mHC projection must be ({coefficients}, {hc * hidden}); the buffer is "
            f"{_capacity(abis['projection'], 'projection')}.")
    if _capacity(abis["base"], "base") != (coefficients, ):
        raise CodegenError(f"mHC base must carry {coefficients} entries.")
    if _capacity(abis["post"], "post") != (tokens, hc) \
            or _capacity(abis["comb"], "comb") != (tokens, hc, hc) \
            or _capacity(collapsed_abi, "collapsed") != (tokens, hidden // lane_count):
        raise CodegenError("mHC pre outputs disagree with the stream capacity.")

    stream_lanes = _lanes(hc)
    requested = int(raw["parameters"]["elements_per_program"])
    return {
        "internal_grid_barriers": 0,
        "projection_reduction": projection_reduction,
        "projection_splits": splits,
        "projection_coefficient_lanes": _lanes(coefficients),
        "streams_fma_token_offset": emit_local_scalar_offset(abis["streams"], ("_fm_token", "0", "0")),
        "streams_fma_stream_stride": int(abis["streams"]["scalar_storage_strides"][1]),
        "streams_fma_hidden_stride": int(abis["streams"]["scalar_storage_strides"][2]),
        "projection_fma_base_offset": emit_local_scalar_offset(abis["projection"], ("0", "0")),
        "projection_fma_row_stride": int(abis["projection"]["scalar_storage_strides"][0]),
        "projection_fma_hidden_stride": int(abis["projection"]["scalar_storage_strides"][1]),
        "tokens": tokens,
        "hc": hc,
        "stream_lanes": stream_lanes,
        "hidden": hidden,
        "elements": hc * hidden,
        "coefficients": coefficients,
        # Pass one stages the pre, post and comb weight rows of one tile at
        # once: (2 * lanes + lanes ** 2) rows, which is the whole projection.
        "matvec_tile": _tile(requested, 2 * stream_lanes + stream_lanes ** 2),
        "collapse_tile": _tile(requested, stream_lanes),
        "epsilon": repr(float(raw["semantic_attrs"]["epsilon"])),
        "hc_epsilon": repr(float(raw["semantic_attrs"]["sinkhorn_epsilon"])),
        "post_multiplier": repr(float(raw["semantic_attrs"]["post_multiplier"])),
        "sinkhorn_iterations": int(raw["semantic_attrs"]["sinkhorn_iterations"]),
        **{name: _pointer(binding) for name, binding in bindings.items()},
        "token_active": emit_active_extent(abis["streams"], 0),
        "stream_active": emit_active_extent(abis["streams"], 1),
        "hidden_active": emit_active_extent(abis["streams"], 2),
        "streams_flat_offset": emit_local_scalar_offset(
            abis["streams"],
            ("_fm_token", f"(_fm_flat // {hidden})", f"(_fm_flat % {hidden})")),
        "streams_plane_offset": emit_local_scalar_offset(
            abis["streams"], ("_fm_token", "_fm_i[:, None]", "_fm_hidden[None, :]")),
        # Each group indexes its own contiguous row range directly, so the
        # matvec accumulates straight into group-shaped registers instead of
        # producing one flat coefficient vector and gathering out of it.
        "projection_pre_offset": emit_local_scalar_offset(
            abis["projection"], ("_fm_i[:, None]", "_fm_flat[None, :]")),
        "projection_post_offset": emit_local_scalar_offset(
            abis["projection"], (f"(_fm_i[:, None] + {hc})", "_fm_flat[None, :]")),
        "projection_comb_offset": emit_local_scalar_offset(
            abis["projection"],
            (f"({2 * hc} + _fm_i[:, None, None] * {hc} + _fm_i[None, :, None])",
             "_fm_flat[None, None, :]")),
        "base_pre_offset": emit_local_scalar_offset(abis["base"], ("_fm_i", )),
        "base_post_offset": emit_local_scalar_offset(abis["base"], (f"(_fm_i + {hc})", )),
        "base_comb_offset": emit_local_scalar_offset(
            abis["base"], (f"({2 * hc} + _fm_i[:, None] * {hc} + _fm_i[None, :])", )),
        "scale_offsets": [emit_local_scalar_offset(abis["scale"], (str(group), ))
                          for group in range(3)],
        "post_offset": emit_local_scalar_offset(abis["post"], ("_fm_token", "_fm_i")),
        "comb_offset": emit_local_scalar_offset(
            abis["comb"], ("_fm_token", "_fm_i[:, None]", "_fm_i[None, :]")),
        "collapsed_offset": emit_local_scalar_offset(
            collapsed_abi,
            ("_fm_token", f"(_fm_hidden // {lane_count})" if lane_count > 1 else "_fm_hidden"),
            lane_coordinate=f"(_fm_hidden % {lane_count})" if lane_count > 1 else None),
        "collapsed_type": emit_triton_scalar_type(str(abis["collapsed"]["scalar_dtype"])),
        "collapse_input_type": emit_triton_scalar_type(collapse_input_dtype) if collapse_input_dtype is not None else None,
        "collapse_sequential_fma": collapse_reduction == "sequential_fma",
        "post_writer_active": _canonical_writer_active(abis["post"]),
        "comb_writer_active": _canonical_writer_active(abis["comb"]),
        "collapsed_writer_active": _canonical_writer_active(abis["collapsed"]),
    }


def mhc_post_call(raw):
    """Render ``out[t, j] = sum_i comb[t, i, j] * residual[t, i] + post[t, j] * value[t]``.

    The mixing matrix and the per-stream shares are read once per token into
    registers, and the stream contraction then runs over hidden tiles, so the
    ``hc**2`` hidden-sized intermediates the graph form would materialize never
    reach memory.
    """

    from triton.flagmega.codegen.triton.kernel_call_renderers import (
        _canonical_writer_active,
        _pointer,
    )

    bindings = _bind(raw, {
        "inputs": ("value", "residual", "post", "comb"),
        "outputs": ("result", ),
    })
    abis = {name: _replicated_abi(binding, name) for name, binding in bindings.items()}

    tokens, hc, hidden = _capacity(abis["residual"], "residual")
    if _capacity(abis["value"], "value") != (tokens, hidden):
        raise CodegenError("mHC post value must be (tokens, hidden) like the residual streams.")
    if _capacity(abis["post"], "post") != (tokens, hc) \
            or _capacity(abis["comb"], "comb") != (tokens, hc, hc):
        raise CodegenError("mHC post coefficients disagree with the residual capacity.")
    if _capacity(abis["result"], "result") != (tokens, hc, hidden):
        raise CodegenError("mHC post result must match the residual stream capacity.")
    dtype = _rounding_dtype(raw["semantic_attrs"], "output_dtype", abis["residual"]["scalar_dtype"])
    if abis["result"]["scalar_dtype"] != dtype:
        raise CodegenError("mHC post result ABI disagrees with output_dtype or residual dtype.")
    mode = raw["semantic_attrs"].get("accumulation_mode", "residual_then_value")
    if not isinstance(mode, str) or mode not in {"residual_then_value", "value_fma_first"}:
        raise CodegenError("mHC accumulation_mode must be residual_then_value or value_fma_first.")

    stream_lanes = _lanes(hc)
    return {
        "internal_grid_barriers": 0,
        "tokens": tokens,
        "hc": hc,
        "stream_lanes": stream_lanes,
        "hidden": hidden,
        # The product is staged as [stream, stream, tile], so the tile budget
        # divides by the square of the stream lane count.
        "tile": _tile(int(raw["parameters"]["elements_per_program"]), stream_lanes ** 2),
        **{name: _pointer(binding) for name, binding in bindings.items()},
        "token_active": emit_active_extent(abis["residual"], 0),
        "stream_active": emit_active_extent(abis["residual"], 1),
        "hidden_active": emit_active_extent(abis["residual"], 2),
        "value_offset": emit_local_scalar_offset(
            abis["value"], ("_fm_token", "_fm_hidden")),
        "residual_offset": emit_local_scalar_offset(
            abis["residual"], ("_fm_token", "_fm_i[:, None]", "_fm_hidden[None, :]")),
        "post_offset": emit_local_scalar_offset(abis["post"], ("_fm_token", "_fm_j")),
        "comb_offset": emit_local_scalar_offset(
            abis["comb"], ("_fm_token", "_fm_i[:, None]", "_fm_j[None, :]")),
        "result_offset": emit_local_scalar_offset(
            abis["result"], ("_fm_token", "_fm_j[:, None]", "_fm_hidden[None, :]")),
        "result_type": emit_triton_scalar_type(str(abis["result"]["scalar_dtype"])),
        "value_first_fma": mode == "value_fma_first",
        "writer_active": _canonical_writer_active(abis["result"]),
    }


__all__ = ["mhc_post_call", "mhc_pre_call"]

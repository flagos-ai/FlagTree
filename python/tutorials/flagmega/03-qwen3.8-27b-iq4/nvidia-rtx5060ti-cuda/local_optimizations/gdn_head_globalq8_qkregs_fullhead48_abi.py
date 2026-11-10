# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Typed, planner-bound ABI for the explicit approximate 48-CTA full-head leaf.

The scalar Shared statistics and chip-global packed Q8 tensor are not VectorType lanes.  This module validates
encoded compiler data; it neither allocates an arena nor rewrites a selection.
Final validation belongs after both r8 configurators and arena reconciliation.
"""
import ast
from math import prod

from triton.flagmega.errors import CodegenError, IRSchemaError
from triton.flagmega.ir import type_from_data
from triton.flagmega.codegen.triton.candidates.gdn_head_gates import dense_value_geometry

IMPLEMENTATION = "tir.gdn_recurrent.persistent_warp_state_head_globalq8_qkregs_fullhead48_v1"
VARIANT = "persistent_warp_state_head_globalq8_qkregs_fullhead48"
DESIGN_DECISION_SHA256 = "66d7fdc10c02b02d7d5c7925885927570eb942d4c48ab1062d8640f27a60bae0"

_PARAMETERS = {
    "tile_state": (128, 4), "projection_tile": 128, "num_warps": 8,
    "head_statistics_cache": True, "head_cache_slots": 2,
    "projection_order": "q8_residual2_f32scale_roundf",
    "qk_norm_order": "ascending_scalar_C1",
    "cache_contents": ("beta", "decay", "query_norm", "key_norm"),
    "cache_epoch": "per_GDN_invocation",
    "state_reduction_order": "warp32_xor_16_8_4_2_1",
    "state_rows_per_warp": 1, "state_values_per_lane": 4,
    "normalized_qk_math": "f32_division", "fullhead_shared_rms": True,
    "rms_order": "ascending_scalar_v0_to127", "q8_producer_map": "first4warps_first40ctas",
}
_FACTS = {
    "internal_grid_barriers": 1, "internal_cta_barriers": 3,
    "required_codegen_backend": "cpp", "state_reassociation": True,
    "state_bitwise_equivalent": False, "projection_bitwise_equivalent": False,
}
_SHARED = {
    "head_statistics": ("float32", (2, 4), (4, 1), 32),
    "head_core": ("float32", (128,), (1,), 512),
}


def _require(condition, message):
    if not condition:
        raise CodegenError("Head global-Q8/fullhead48: " + message)


def _same(actual, expected):
    if isinstance(expected, tuple):
        return isinstance(actual, (tuple, list)) and len(actual) == len(expected) and all(
            _same(a, b) for a, b in zip(actual, expected))
    return type(actual) is type(expected) and actual == expected


def _identity(value):
    return type(value) is int and value >= 0 or isinstance(value, str) and bool(value.strip())


def _binding(call, group, formal, index=0):
    matches = [p for p in call.get(group, ()) if p.get("formal") == formal]
    _require(len(matches) == 1, f"requires one {group}/{formal} binding")
    buffers = matches[0].get("buffers", ())
    _require(len(buffers) == (2 if formal in ("state", "result_1") else 1),
             f"wrong buffer arity for {formal}")
    buffer = buffers[index]
    _require(isinstance(buffer.get("abi"), dict) and _identity(buffer["abi"].get("physical_buffer"))
             and isinstance(buffer.get("runtime_argument"), str) and bool(buffer["runtime_argument"])
             and buffer.get("runtime_value_kind") == "pointer", f"missing physical pointer ABI for {formal}")
    return buffer


def _shared(call):
    workspaces = call.get("shared_workspaces", ())
    _require(len(workspaces) == 2 and {w.get("name") for w in workspaces} == set(_SHARED),
             "requires exactly the named typed scalar Shared statistics and core workspaces")
    by_name = {}
    for ws in workspaces:
        dtype, shape, strides, nbytes = _SHARED[ws["name"]]
        offset = ws.get("offset_bytes")
        _require(ws.get("dtype") == dtype and _same(ws.get("shape"), shape)
                 and _same(ws.get("strides"), strides) and _same(ws.get("nbytes"), nbytes)
                 and _same(ws.get("alignment_bytes"), 16) and ws.get("matrix_compatible") is False
                 and type(offset) is int and offset >= 0 and offset % 16 == 0
                 and _identity(ws.get("physical_buffer")), "invalid scalar Shared type/span/alignment/ID")
        # Optional extra metadata must not turn scalar extents into vector lanes.
        _require(not ws.get("scalar_lane_shape") and not ws.get("vector_lanes")
                 and ws.get("scalar_lane_count", 1) == 1, "Shared tensor extents are not vector lanes")
        by_name[ws["name"]] = ws
    spans = [(w["offset_bytes"], w["offset_bytes"] + w["nbytes"]) for w in by_name.values()]
    _require(spans[0][1] <= spans[1][0] or spans[1][1] <= spans[0][0], "live Shared spans overlap")
    _require(len({w["physical_buffer"] for w in by_name.values()}) == 2, "Shared buffers need distinct physical IDs")
    return by_name


def _typed(buffer, shape, strides, scalar, lanes=(), logical=None):
    abi = buffer["abi"]
    count = prod(lanes)
    dtype = scalar + ("<" + ",".join(map(str, lanes)) + ">" if lanes else "")
    itemsize = {"bfloat16": 2, "uint8": 1, "float32": 4}[scalar]
    _require(abi.get("schema") == "flagmega.local-buffer-abi/v2"
             and abi.get("dtype") == dtype and abi.get("scalar_dtype") == scalar
             and _same(abi.get("scalar_itemsize"), itemsize)
             and _same(abi.get("scalar_lane_shape"), lanes)
             and _same(abi.get("scalar_lane_count"), count)
             and _same(abi.get("local_capacity_shape"), shape)
             and _same(abi.get("logical_shape"), logical or shape)
             and _same(abi.get("scalar_storage_strides"), strides),
             f"wrong dtype/lanes/shape/stride for {buffer.get('formal')}")
    alignment, offset = abi.get("alignment_bytes"), abi.get("pool_byte_offset", 0)
    # External scalar pointers carry alignment=1 in the normal descriptor.
    # Their runtime typed allocation still owes natural alignment; do not invent
    # float4/BF16x8 transactions or demand a 4B IQ4 pointer from this metadata.
    _require(type(alignment) is int and alignment > 0 and not alignment & (alignment - 1)
             and type(offset) is int and offset >= 0 and offset % itemsize == 0,
             "invalid typed pointer alignment/offset")
    for field in ("distributed_type", "distributed_backing_type"):
        data = abi.get(field)
        if data is None:
            continue
        try:
            ty = type_from_data(data)
            _require(ty.placement.hierarchy == (4, 12) and ty.placement.hierarchy_levels == "bb"
                     and ty.partial is None and ty.exclusive is None
                     and ty.tensor.dtype.value == dtype
                     and tuple(d.fixed_value for d in ty.tensor.shape) == (logical or shape),
                     "wrong actual distributed dtype/shape/mesh/partial/exclusive")
        except (IRSchemaError, ValueError, KeyError, TypeError, AttributeError) as error:
            raise CodegenError("Head global-Q8/fullhead48: malformed distributed type") from error
    return abi


def _interval(abi):
    # Canonical output is locally128 values but different CTAs address the full
    # 6144-element span.  Alias checking must use that full footprint.
    shape = abi["logical_shape"] if abi.get("coordinate_space") == "canonical_global" else abi["local_capacity_shape"]
    count = abi["scalar_lane_count"] + sum((n - 1) * s for n, s in zip(shape, abi["scalar_storage_strides"]))
    offset = abi.get("pool_byte_offset", 0)
    return offset, offset + count * abi["scalar_itemsize"]


def _disjoint(a, b):
    aa, ba = a["abi"], b["abi"]
    _require(aa["physical_buffer"] != ba["physical_buffer"], "mutable/readonly or mutable/mutable physical alias")
    if a["runtime_argument"] != b["runtime_argument"]:
        return
    _require(not aa.get("view_byte_offset") and not ba.get("view_byte_offset"), "unproved symbolic pointer alias")
    a0, a1 = _interval(aa)
    b0, b1 = _interval(ba)
    _require(a1 <= b0 or b1 <= a0, "overlapping pooled operand spans")


def _global_q8(call):
    q8 = _binding(call, "workspaces", "head_projection_q8_global")
    abi = _typed(q8, (2, 160, 36), (5760, 36, 1), "uint8")
    _require(abi.get("storage") == "workspace"
             and abi.get("coordinate_space") == "canonical_global"
             and abi.get("memory_sharing_scope") == "chip"
             and abi.get("alignment_bytes", 0) >= 16
             and abi.get("pool_byte_offset", 0) % 16 == 0,
             "global Q8 requires chip-visible canonical workspace with16B alignment")
    return q8


def _operands(call):
    read_specs = (
        ("qkv", (1, 10240), (10240, 1), "bfloat16", None),
        ("z", (1, 128), (128, 1), "bfloat16", (1, 6144)),
        ("projection_input", (1, 5120), (5120, 1), "bfloat16", None),
        ("b_weight", (48, 2720), (2720, 1), "uint8", None),
        ("a_weight", (48, 2720), (2720, 1), "uint8", None),
        ("a_log", (48,), (1,), "float32", None),
        ("dt_bias", (48,), (1,), "float32", None),
        ("norm_weight", (128,), (1,), "float32", None),
    )
    reads = []
    for name, shape, strides, dtype, logical in read_specs:
        b = _binding(call, "inputs", name)
        _typed(b, shape, strides, dtype, logical=logical)
        reads.append(b)
    state = _binding(call, "inputs", "state", 1)
    conv = _binding(call, "inputs", "state", 0)
    _typed(state, (1, 48, 128, 32), (786432, 16384, 128, 4), "float32", (4,))
    _typed(conv, (1, 1280, 3), (30720, 24, 8), "bfloat16", (8,))
    for i, src in enumerate((conv, state)):
        dst = _binding(call, "outputs", "result_1", i)
        _require(dst["abi"] == src["abi"] and dst["runtime_argument"] == src["runtime_argument"],
                 "returned state must alias exactly the original state field")
    result, core = _binding(call, "outputs", "result_0"), _binding(call, "workspaces", "core_scratch")
    _typed(result, (1, 128), (6144, 1), "bfloat16", logical=(1, 6144))
    _typed(core, (48, 128), (128, 1), "float32")
    _require(all(b["abi"].get("coordinate_space") == "canonical_global" for b in (state, result, core)),
             "state/core/output need canonical global addresses")
    _require(reads[1]["abi"].get("memory_sharing_scope") == "block"
             and reads[1]["abi"].get("coordinate_space") == "parent_shard_local", "Z must retain its private owner-local view")
    q8 = _global_q8(call)
    for other in (*reads, conv, state, result, core):
        _disjoint(q8, other)
    writes = (state, result, core)
    for i, write in enumerate(writes):
        for other in (*reads, conv, *writes[i + 1:]):
            _disjoint(write, other)
    return result


def _formal_types(call):
    """Bind the selected distributed formal types to the actual physical ABI."""
    distribution = call.get("parameters", {}).get("distribution", {})
    inputs = distribution.get("input_types", ())
    output = distribution.get("output_type", {})
    _require(len(inputs) == 9 and output.get("kind") == "tuple" and len(output.get("fields", ())) == 2,
             "missing selected input/result type lineage")
    ref = inputs[0]
    _require(ref.get("kind") == "ref" and output["fields"][1] == ref
             and [f.get("name") for f in ref.get("fields", ())] == ["convolution", "recurrent"],
             "selected state tuple lineage changed")
    for field, shape, scalar, lanes in zip(ref["fields"], ((1, 1280, 3), (1, 48, 128, 32)),
                                           ("bfloat16", "float32"), ((8,), (4,))):
        data = field.get("type", {})
        _require(data.get("kind") == "tensor" and data.get("dtype") == {
            "kind": "vector", "elem_type": scalar, "lanes": list(lanes)}
                 and tuple(d.get("value") for d in data.get("shape", ()) if d.get("kind") == "fixed") == shape,
                 "state formal TensorType lanes/shape differ from physical ABI")
    names = ("qkv", "z", "projection_input", "b_weight", "a_weight", "a_log", "dt_bias", "norm_weight")
    for data, name in zip(inputs[1:], names):
        _require(data == _binding(call, "inputs", name)["abi"].get("distributed_type"),
                 f"selected formal and physical distributed type disagree for {name}")
        if name != "z":
            _require(data.get("axis_policies") == [{"kind": "broadcast"}] * len(data["tensor"]["shape"]),
                     f"{name} must retain complete replicated source rows")
    _require(output["fields"][0] == _binding(call, "outputs", "result_0")["abi"].get("distributed_type"),
             "selected result formal and physical distributed type disagree")
    _require(inputs[2] == output["fields"][0], "Z and output must retain the same value ownership type")


def prepare_fields(raw, call):
    """Return convenience fields from planned raw spans; never mutate either input."""
    if raw.get("implementation") != IMPLEMENTATION:
        return {}
    _require(raw.get("parameters", {}).get("head_statistics_cache") is True, "missing selected cache parameter")
    ws = _shared(raw)
    _operands(raw)
    _formal_types(raw)
    if call.get("implementation") is not None:
        _require(call["implementation"] == IMPLEMENTATION, "raw/encoded implementation mismatch")
    if "shared_workspaces" in call:
        _require(call["shared_workspaces"] == raw["shared_workspaces"], "raw/encoded Shared binding mismatch")
    stats, core, q8 = ws["head_statistics"], ws["head_core"], _global_q8(raw)
    from triton.flagmega.codegen.triton.kernel_call_renderers import _pointer
    return {
        "head_statistics_cache": True,
        "head_statistics_offset_bytes": stats["offset_bytes"],
        "head_statistics_physical_buffer": stats["physical_buffer"],
        "head_core_offset_bytes": core["offset_bytes"],
        "head_core_physical_buffer": core["physical_buffer"],
        "head_projection_q8_global": _pointer(q8),
        "head_projection_q8_global_physical_buffer": q8["abi"]["physical_buffer"],
        "head_projection_arena_end_bytes": max(w["offset_bytes"] + w["nbytes"] for w in ws.values()),
    }


def _expression(expression):
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, TypeError) as error:
        raise CodegenError("Head global-Q8/fullhead48: malformed coordinate expression") from error
    allowed = (ast.Expression, ast.BinOp, ast.Add, ast.Mult, ast.Name, ast.Load, ast.Constant)
    names = {"shard_index", "shard_coord0", "shard_coord1", "shard_x", "shard_y", "local_values", "recurrent_start"}
    _require(all(isinstance(n, allowed) and (not isinstance(n, ast.Name) or n.id in names)
                 and (not isinstance(n, ast.Constant) or type(n.value) is int) for n in ast.walk(tree)),
             "requires a statically proven dense affine owner map")
    return compile(tree, "<head-dualq8-owner>", "eval")


def _owners(call):
    result = _binding(call, "outputs", "result_0")["abi"]
    try:
        distributed = type_from_data(result["distributed_type"])
        geometry = dense_value_geometry(distributed, 48, 128)
        _require(geometry is not None and geometry[0] == 128, "output type has wrong dense owner geometry")
    except (IRSchemaError, ValueError, TypeError, KeyError, AttributeError) as error:
        raise CodegenError("Head global-Q8/fullhead48: invalid output owner type") from error
    expressions = {k: _expression(call.get(k)) for k in (
        "global_value", "head_statistics_first_value", "global_projection_value", "result_offset", "z_offset")}
    for c in range(48):
        env = {"shard_index": c, "shard_x": c % 12, "shard_y": c // 12,
               "shard_coord0": c // 12, "shard_coord1": c % 12}
        try:
            _require(eval(expressions["head_statistics_first_value"], {"__builtins__": {}}, env) == 128 * c,
                     "head first value differs from Distribution owner map")
            for value in range(128):
                env.update(local_values=value, recurrent_start=value)
                for key, expected in (("global_value", 128 * c + value), ("global_projection_value", 128 * c + value),
                                      ("result_offset", 128 * c + value), ("z_offset", value)):
                    _require(eval(expressions[key], {"__builtins__": {}}, env) == expected, f"wrong {key} owner map")
        except (NameError, TypeError) as error:
            raise CodegenError("Head global-Q8/fullhead48: incomplete coordinate map") from error


def validate_call(call, context):
    """Validate an actual CPP encoded call; old identities are left to their guards."""
    if call.get("implementation") != IMPLEMENTATION:
        return
    _require(context.get("schema") == "flagmega.cpp-package-descriptor/v1"
             and context.get("platform") == "nvidia" and context.get("target_architecture") == "sm120"
             and _same(context.get("mesh_hierarchy"), (4, 12)) and _same(context.get("grid"), (48, 1, 1))
             and _same(context.get("block_threads"), 256) and _same(context.get("num_warps"), 8)
             and _same(context.get("warp_size"), 32) and context.get("pipeline_schedule") is None,
             "requires actual CPP SM120/4x12/grid48/block256/warp32 descriptor")
    _require(call.get("family") == "gdn_recurrent" and call.get("variant") == VARIANT
             and call.get("semantic_op") == "nn.gdn_recurrent_core"
             and call.get("transfer_pipeline") is None and not call.get("pipeline_channels")
             and str(call.get("participation_active")) == "True" and str(call.get("writer_active")) == "True",
             "wrong variant, semantic operation, pipeline or participation")
    for source, expected in ((call.get("parameters", {}), _PARAMETERS), (call.get("facts", {}), _FACTS)):
        _require(all(_same(source.get(k), v) for k, v in expected.items()), "selected numeric/publication contract differs")
    expected = {"b_weight_encoding": "IQ4_XS", "a_weight_encoding": "IQ4_XS", "value_heads": 48,
                "key_heads": 16, "value_dim": 128, "key_dim": 128, "projection_dim": 5120,
                "tiled_value_heads": True, "round_beta": True, "round_core": False,
                "round_normalized_qk": False, "qk_norm_add": False, "gate_math_mode": "accurate",
                "accurate_norm_silu": False, "local_capacity": 128, "internal_grid_barriers": 1}
    _require(all(_same(call.get(k), v) for k, v in expected.items()) and str(call.get("active_values")) == "128",
             "unsupported K/head/batch/tail/numeric mode")
    try:
        _require(float(call.get("qk_norm_epsilon")) == 1e-12
                 and float(call.get("epsilon")) == 9.999999974752427e-7
                 and float(call.get("query_scale")) == 1.0 / (128 ** 0.5), "altered norm epsilon or query scale")
    except (ValueError, TypeError) as error:
        raise CodegenError("Head global-Q8/fullhead48: invalid numerical scalar") from error
    expected_fields = prepare_fields(call, call)
    _require(all(_same(call.get(k), v) for k, v in expected_fields.items()), "fields differ from planned Shared bindings")
    # Rebind emitted pointer/semantic convenience fields through the real core
    # encoder. A correct raw ABI does not authorize a different emitted pointer.
    # This identity never enters the legacy scalar-head special branch.
    from triton.flagmega.codegen.triton.kernel_call_renderers import _gdn_recurrent_call
    encoded = _gdn_recurrent_call(call)
    for key in ("state", "qkv", "z", "projection_input", "b_weight", "a_weight", "a_log", "dt_bias",
                "norm_weight", "result", "core_scratch", "value_heads", "key_heads", "key_dim", "value_dim",
                "projection_dim", "epsilon", "query_scale", "qk_norm_add", "qk_norm_epsilon",
                "round_normalized_qk", "round_beta", "round_core", "gate_math_mode", "accurate_norm_silu",
                "writer_active", "b_weight_encoding", "a_weight_encoding", "tiled_value_heads"):
        _require(_same(call.get(key), encoded[key]), f"emitted {key} differs from actual encoded ABI/semantics")
    cap = context.get("launch_resource_limits", {}).get("max_shared_memory_bytes")
    _require(type(cap) is int and 0 < expected_fields["head_projection_arena_end_bytes"] <= min(49152, cap),
             "Shared spans exceed static/target limit")
    _owners(call)


def program_contexts(descriptor):
    """Bind per-entry guard contexts to C0's compact native entry table.

    This is a local guard adapter, not a second program planner/serializer.
    Only C0's normal renderer may supply these full contexts.
    """
    programs = descriptor.get("cpp_programs")
    contexts = descriptor.get("native_program_contexts")
    if programs is None and contexts is None:
        return None
    _require(isinstance(programs, dict) and programs.get("schema") == "flagmega.cpp-programs/v1"
             and isinstance(contexts, (list, tuple)) and bool(contexts)
             and all(isinstance(c, dict) for c in contexts),
             "separate programs require actual native entry contexts")
    entries = programs.get("entries", ())
    _require(isinstance(entries, (list, tuple)) and len(entries) == len(contexts)
             and all(isinstance(e, dict) for e in entries),
             "native entry/context count differs")
    ids = [e.get("entry_id") for e in entries]
    _require(all(_identity(i) for i in ids) and len(set(ids)) == len(ids), "duplicate/missing native entry identity")
    by_id = {e["entry_id"]: e for e in entries}
    seen = set()
    for context in contexts:
        entry_id = context.get("entry_id")
        _require(_identity(entry_id) and entry_id in by_id and entry_id not in seen and not context.get("cpp_programs")
                 and not context.get("native_program_contexts"), "stale or recursive native entry context")
        seen.add(entry_id)
        entry = by_id[entry_id]
        _require(context.get("launch_kind") == entry.get("launch_kind")
                 and context.get("launch_kind") in ("ordinary", "cooperative")
                 and isinstance(entry.get("grid"), (list, tuple))
                 and _same(context.get("grid"), tuple(entry.get("grid", ()))),
                 "native context launch/grid differs from immutable table")
        if any(c.get("implementation") == IMPLEMENTATION for c in context.get("render_calls", ())):
            _require(context.get("launch_kind") == "cooperative"
                     and context.get("requires_grid_wide_sync") is True,
                     "GDN Shared/private ABI belongs to its fixed cooperative entry")
    return contexts


def validate_selection(descriptor):
    """Validate final combined arena and static schedule, returning every GDN site."""
    contexts = program_contexts(descriptor)
    if contexts is not None:
        manifest = []
        for context in contexts:
            if any(c.get("family") == "gdn_recurrent" for c in context.get("render_calls", ())):
                manifest.extend({**site, "entry_id": context["entry_id"]} for site in validate_selection(context))
        _require(bool(manifest), "program selection lost all GDN sites")
        return manifest
    calls = [c for c in descriptor.get("render_calls", ()) if c.get("family") == "gdn_recurrent"]
    _require(bool(calls) and all(c.get("implementation") == IMPLEMENTATION for c in calls),
             "all selected GDN sites must use the requested implementation")
    recipe = descriptor.get("local_iq4_recipe", {})
    _require(recipe.get("gdn_head_projection") == "gdn_head_globalq8_qkregs_fullhead48" and recipe.get("gdn_warp_state") is True
             and not recipe.get("head_gate_cache", False) and not recipe.get("head_gate_chunk", 0),
             "wrong/legacy head selectors")
    arena = descriptor.get("local_iq4_shared_bytes")
    _require(type(arena) is int and arena > 0 and recipe.get("scratch", {}).get("nbytes") == arena,
             "combined arena and recipe scratch size disagree")
    for other in descriptor.get("render_calls", ()):
        if other.get("local_iq4_async"):
            spans = other.get("shared_workspaces", ())
            _require(bool(spans) and all(type(w.get("offset_bytes")) is int and type(w.get("nbytes")) is int
                                       and 0 <= w["offset_bytes"] and 0 < w["nbytes"]
                                       and w["offset_bytes"] + w["nbytes"] <= arena for w in spans),
                     "combined arena cannot cover an existing async matrix span")
    schedules = [descriptor.get("entry_events", ())]
    event_owners = {id(e): descriptor.get("function_schedule", {}).get("function") for e in schedules[0]}
    for function in descriptor.get("device_functions", ()):
        schedule = function.get("schedule", {})
        _require(schedule.get("schema") == "flagmega.triton-pipeline-source-schedule/v3"
                 and not any(schedule.get(k) for k in ("stages", "handoffs", "producer_events", "auxiliary_events",
                                                       "trip_scoped_drains", "trip_scoped_closes")),
                 "arena reuse requires sequential function schedules")
        schedules.append(schedule.get("consumer_events", ()))
        event_owners.update({id(e): function.get("function") for e in schedules[-1]})
    events = [e for schedule in schedules for e in schedule]
    _require(all(e.get("kind") in ("barrier", "function_call", "tir.kernel_call")
                 and not e.get("pipeline_channels") and e.get("participation_active", "True") == "True" for e in events),
             "arena reuse forbids conditional/nonsequential events")
    manifest = []
    ids, symbols = set(), set()
    for call in calls:
        validate_call(call, descriptor)
        _require(call["head_projection_arena_end_bytes"] <= arena <= min(49152, descriptor["launch_resource_limits"]["max_shared_memory_bytes"]),
                 "final arena cannot cover the selected spans")
        symbol = call.get("symbol")
        matches = [e for e in events if e.get("kind") == "tir.kernel_call" and e.get("symbol") == symbol]
        _require(isinstance(symbol, str) and symbol not in symbols and len(matches) == 1
                 and matches[0].get("call") == call.get("call") and matches[0].get("variant") == VARIANT,
                 "selected call is not bound to exactly one matching static event")
        caller = event_owners[id(matches[0])]
        _require(isinstance(caller, str) and bool(caller), "selected event needs an actual caller function identity")
        symbols.add(symbol)
        for index, ws in enumerate(call["shared_workspaces"]):
            identity = ws["physical_buffer"]
            # Final execution materialization gives each call its own identity;
            # offsets can be shared across sequential calls, identities cannot.
            expected_id = f"execution:{caller}:{call['call']}:shared:{call['callee']}:{index}:{ws['name']}"
            _require(identity == expected_id and identity not in ids, "stale or foreign execution Shared ID")
            ids.add(identity)
        manifest.append({"symbol": symbol, "call": call["call"], "implementation": IMPLEMENTATION,
                         "shared_workspaces": call["shared_workspaces"],
                         **prepare_fields(call, call), "arena_name": "_local_iq4_q8",
                         "statistics_lifetime": "current call: prepare through final norm read; slot4 broadcasts RMS",
                         "head_core_lifetime": "current call: each completed core store through final norm read",
                         "q8_lifetime": "entire GDN invocation; globally published before A/B reads and disjoint from core scratch",
                         "reuse": "sequential calls only; preceding async groups drained and CTA reuse complete",
                         "design_decision_sha256": DESIGN_DECISION_SHA256,
                         "projection_bitwise_equivalent": False})
    return manifest

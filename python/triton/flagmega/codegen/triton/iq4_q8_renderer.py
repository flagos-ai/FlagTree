# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Checked r003 Q8 ABI and calls into the unchanged accepted JIT source.

This module owns operand encoding, not graph admission or scheduling. A fixed
canonical pool offset includes the buffer subspan once. Dynamic subspans and
distributed partial owners require a separately qualified interface.
"""
from __future__ import annotations

from collections.abc import Mapping
from math import prod

from triton.flagmega.errors import CodegenError

SCHEMA = "flagmega.iq4-q8-native/r003-v2"
GRAPH_SCHEMA = "flagmega.iq4-q8-graph/r003-v2"
PLAN_SHA256 = "eaa88d099e7d4c9cb33b24793abbaa6e1b42159900c032c40b958ffd1d8bebc4"
NUMERICAL_SUBJECT_SHA256 = "2d0483edc3787d3dab325f3665e205bec952216a146937152300d3fc4a26ccb1"
NATIVE_INTERFACE = "K4L32-Q8_1-DP4A/r003-v2"
Q8_ABI = "Q8_1-bytes36-le/v1"
RESIDENCY_ENCODING = {"size_per_thread": [1, 1], "threads_per_warp": [1, 32],
                      "warps_per_cta": [4, 1], "order": [1, 0]}
FAMILIES = ("q8_1_prepare", "iq4_q8_projection", "iq4_q8_glu", "iq4_q8_interleaved")
VARIANT = "q8_r003_v2"
JIT_MODULE = "triton.flagmega.codegen.triton.iq4_q8_kernels"
JIT_SOURCE_SHA256 = "5f9dd5a6370273f62d8b4906754b252f634d7e63083912e89ba455d4931885eb"


def multirow_column_tile(family, k, n=0):
    # R039: two columns leave room for the full instruction window without
    # spilling. This also qualifies output K6144 and interleaved Q.
    if family == 'iq4_q8_projection' and (k == 17408 or n >= 131072):
        return 4
    return 2 if family in ('iq4_q8_projection', 'iq4_q8_interleaved') else 1


def implementation_parameters(family):
    """Fresh JSON values for the catalog and graph's parameter verifier."""
    if family not in FAMILIES:
        raise CodegenError(f"Unknown r003 Q8 family {family!r}.")
    return dict(graph_schema=GRAPH_SCHEMA, native_schema=SCHEMA,
                plan_hash=PLAN_SHA256, plan_sha256=PLAN_SHA256,
                numerical_profile="H003-N-v2", numerical_profile_hash=NUMERICAL_SUBJECT_SHA256,
                native_interface=NATIVE_INTERFACE, q8_abi=Q8_ABI,
                residency_encoding={k: tuple(v) for k, v in RESIDENCY_ENCODING.items()},
                row_tile=1, tile_n=1, num_warps=4, threads=128, num_ctas=1,
                program_kind="ordinary", num_stages=3, enable_fp_fusion=False,
                maxnreg=104 if family == FAMILIES[2] else 96 if family == FAMILIES[3] else 80,
                enable_static_shared=family != FAMILIES[0], dynamic_shared_bytes=0,
                ptx_options="--minnctapersm=1")


def _require(ok, message):
    if not ok:
        raise CodegenError("r003 Q8: " + message)


def _canonical(value):
    # TIR immutable attrs freeze nested lists to tuples; exported raw JSON
    # restores lists. Both carry the identical versioned encoding identity.
    if isinstance(value, Mapping):
        return {k: _canonical(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_canonical(v) for v in value)
    return value


def _integer(value, name, *, minimum=0):
    _require(type(value) is int and value >= minimum, f"invalid {name}")
    return value


def _shape(abi, key):
    value = abi.get(key)
    _require(isinstance(value, (tuple, list)) and value
             and all(type(v) is int and v > 0 for v in value), f"missing/static {key}")
    return tuple(value)


def _bindings(raw, side, formal, count=1):
    args = raw.get(side)
    _require(isinstance(args, (tuple, list)), f"missing {side}")
    matched = [a for a in args if isinstance(a, Mapping) and a.get("formal") == formal]
    _require(len(matched) == 1, f"requires one {side} formal {formal}")
    buffers = matched[0].get("buffers")
    _require(isinstance(buffers, (tuple, list)) and len(buffers) == count,
             f"{formal} requires {count} buffer(s)")
    for b in buffers:
        _require(isinstance(b, Mapping) and isinstance(b.get("abi"), Mapping)
                 and isinstance(b.get("actual"), str) and bool(b["actual"])
                 and isinstance(b.get("runtime_argument"), str)
                 and b["runtime_argument"].isidentifier(), f"bad {formal} binding/identity")
    return tuple(buffers)


def parameter_alignments(family, attrs):
    """Graph's TIR materializer applies these kernel-owned ABI requirements."""
    _require(family in FAMILIES, "unknown alignment family")
    if family == FAMILIES[0]:
        _require(attrs.get("source_dtype") in {"bfloat16", "float32"}, "unknown source dtype")
        return {"value": 2 if attrs["source_dtype"] == "bfloat16" else 4, "result": 4}
    output = 4 if attrs.get("output_data_type") == "float32" else 2
    weights = ("gate_weight", "up_weight") if family == FAMILIES[2] else ("weight",)
    result = {"q8": 4, **{name: 4 for name in weights}, **{name + "_scale": 4 for name in weights}}
    result.update({"result_0": 2, "result_1": 2} if family == FAMILIES[3] else {"result": output})
    return result


def _results(raw):
    if raw["family"] == FAMILIES[3]:
        _require(tuple(a.get("formal") for a in raw.get("outputs", ())) == ("result_0", "result_1"),
                 "requires two ordered interleaved output formals")
        return (_bindings(raw, "outputs", "result_0")[0], _bindings(raw, "outputs", "result_1")[0])
    _require(tuple(a.get("formal") for a in raw.get("outputs", ())) == ("result",), "wrong output formal")
    return _bindings(raw, "outputs", "result")


def _placeholder(raw, formal, shape):
    args = [a for a in raw["inputs"] if a.get("formal") == formal]
    _require(len(args) == 1, "missing scale placeholder")
    a = args[0]
    if a.get("buffers"):
        b = _bindings(raw, "inputs", formal)[0]
        _abi(b, dtype="float32", shape=shape)
        return {"formal": formal, "actual": b["actual"], "device_read": False}
    # MemoryEffect.NONE is represented by a typed metadata binding, not an
    # allocated device argument. Retain identity/type without inventing a read.
    from triton.flagmega.ir import type_from_data
    from triton.flagmega.ir.distributed_inference import tensor_of
    _require(a.get("role") == "metadata" and isinstance(a.get("actual_value"), str)
             and bool(a["actual_value"]) and isinstance(a.get("type"), Mapping),
             "missing scale placeholder identity/type")
    try:
        t = tensor_of(type_from_data(a["type"]))
        valid = t.dtype.value == "float32" and tuple(d.fixed_value for d in t.shape) == tuple(shape)
    except Exception as error:
        raise CodegenError("r003 Q8: invalid metadata scale type") from error
    _require(valid, "wrong metadata scale dtype/shape")
    return {"formal": formal, "actual": a["actual_value"], "type": dict(a["type"]), "device_read": False}


def _abi(binding, *, dtype, shape, lanes=(), strides=None, alignment=None):
    abi = binding["abi"]
    itemsize = {"uint8": 1, "bfloat16": 2, "float32": 4}[dtype]
    _require(abi.get("scalar_dtype") == dtype and abi.get("scalar_itemsize") == itemsize,
             f"wrong dtype/itemsize for {binding['actual']}")
    _require(tuple(abi.get("scalar_lane_shape", ())) == tuple(lanes)
             and abi.get("scalar_lane_count") == prod(lanes), "wrong scalar lanes")
    _require(_shape(abi, "logical_shape") == tuple(shape)
             and _shape(abi, "local_capacity_shape") == tuple(shape), "incomplete owner/shape")
    _require(tuple(str(v) for v in abi.get("active_shape_expressions", ()))
             == tuple(str(v) for v in shape), "incomplete active owner")
    if strides is None:
        strides = tuple(prod(shape[i+1:]) * prod(lanes) for i in range(len(shape)))
    _require(tuple(abi.get("scalar_storage_strides", ())) == tuple(strides),
             "bad scalar/record stride")
    _require(abi.get("coordinate_space") in {"local", "canonical_global", "parent_shard_local"},
             "missing local coordinate space")
    _require(abi.get("owner_count", 1) == 1, "multiple owners/K sharding unsupported")
    # The canonical serializer omits the scope stride for a single CHIP pool.
    _require(abi.get("pool_scope_stride_bytes", 0) == 0
             and abi.get("memory_sharing_scope", "chip") == "chip",
             "requires CHIP span, not per-CTA storage")
    _require(abi.get("component_stride_scalar_elements", 0) == 0,
             "component-table storage requires a separate qualified ABI")
    _require("view_byte_offset" not in abi and not binding.get("address_arguments"),
             "dynamic subspan requires a qualified versioned proof")
    align = alignment or itemsize
    effective = _integer(abi.get("alignment_bytes"), "alignment_bytes", minimum=1)
    offset = _integer(abi.get("pool_byte_offset"), "pool_byte_offset")
    _require(effective % align == 0 and offset % align == 0, "misaligned span/pool offset")
    # canonical pool_byte_offset already includes a fixed mem_span subspan.
    # An aligned effective pointer and an aligned total offset prove the base
    # alignment independently of the record stride. Optional detailed proofs
    # are checked as well; their offsets are never added a second time.
    if "allocation_alignment_bytes" in abi:
        _require(_integer(abi["allocation_alignment_bytes"], "allocation alignment", minimum=1)
                 % align == 0, "misaligned allocation base")
    if "subspan_byte_offset" in abi:
        _require(_integer(abi["subspan_byte_offset"], "subspan offset") % align == 0,
                 "misaligned subspan")
    payload = prod(shape) * prod(lanes) * itemsize
    if "span_capacity_bytes" in abi:
        _require(abi["span_capacity_bytes"] == payload, "wrong span capacity")
    _require(isinstance(abi.get("physical_buffer"), str) and bool(abi["physical_buffer"]),
             "missing physical allocation identity")
    return payload


def _disjoint(bindings):
    """Prove physical byte intervals; actual/formal names are not allocations."""
    intervals = []
    for b in bindings:
        abi = b["abi"]
        start = abi["pool_byte_offset"]
        size = prod(abi["local_capacity_shape"]) * abi["scalar_lane_count"] * abi["scalar_itemsize"]
        interval = (abi["physical_buffer"], start, start + size)
        for allocation, low, high in intervals:
            _require(interval[0] != allocation or interval[2] <= low or high <= interval[1],
                     "overlapping write/read or Q field spans")
        intervals.append(interval)


def validate_raw(raw):
    """Reject missing semantic/physical/compile metadata before encoding."""
    _require(isinstance(raw, Mapping), "raw must be a mapping")
    family = raw.get("family")
    _require(family in FAMILIES and raw.get("variant") == VARIANT, "wrong family/variant")
    p, a, facts = (raw.get(k) for k in ("parameters", "semantic_attrs", "facts"))
    _require(all(isinstance(v, Mapping) for v in (p, a, facts)), "missing parameters/attrs/facts")
    for key, expected in implementation_parameters(family).items():
        # bool == int in Python must not admit malformed compiler options.
        _require(key in p and type(p[key]) is type(expected) and _canonical(p[key]) == _canonical(expected),
                 f"missing/wrong parameter {key}")
    _require(type(facts.get("internal_grid_barriers")) is int
             and facts["internal_grid_barriers"] == 0
             and facts.get("required_codegen_backend") == "triton", "wrong backend/grid barrier facts")
    _require(raw.get("execution_kind") == "local_shard", "requires local_shard owner relationship")
    identity = dict(graph_schema=GRAPH_SCHEMA, plan_hash=PLAN_SHA256,
                    numerical_profile="H003-N-v2", numerical_profile_hash=NUMERICAL_SUBJECT_SHA256,
                    native_interface=NATIVE_INTERFACE, q8_abi=Q8_ABI)
    for key, value in identity.items():
        _require(a.get(key) == value, f"missing/wrong attr {key}")
    k = _integer(a.get("logical_k"), "logical_k", minimum=1)
    kp, g = 32 * ((k + 31) // 32), (k + 31) // 32
    for key, expected in dict(padded_k=kp, group_size=32, group_count=g, record_bytes=36).items():
        _require(type(a.get(key)) is int and a[key] == expected, f"wrong K/Kp/G or {key}")
    _require(a.get("source_dtype") in {"bfloat16", "float32"}
             and isinstance(a.get("source_version_key"), str) and bool(a["source_version_key"]),
             "missing source dtype/version")
    inputs = ("value",) if family == FAMILIES[0] else (
        ("q8", "gate_weight", "gate_weight_scale", "up_weight", "up_weight_scale")
        if family == FAMILIES[2] else ("q8", "weight", "weight_scale"))
    _require(tuple(arg.get("formal") for arg in raw.get("inputs", ())) == inputs,
             "wrong formal inputs/gate-up order")
    results = _results(raw)
    qb = results[0] if family == FAMILIES[0] else _bindings(raw, "inputs", "q8")[0]
    # A static token bucket widens only the leading record axis. Every other
    # Q8 contract stays frozen, so the single-row call is unchanged.
    rows = _shape(qb["abi"], "logical_shape")[0]
    _require(rows >= 1, "Q8 record needs a positive token extent")
    _abi(qb, dtype="uint8", shape=(rows, g, 36), alignment=4)
    _require(qb["abi"].get("storage") == "workspace", "Q8 must be dynamic CHIP temporary")
    if family == FAMILIES[0]:
        x = _bindings(raw, "inputs", "value")[0]
        _abi(x, dtype=a["source_dtype"], shape=(rows, k))
        _disjoint((x, qb))
        return
    _require(k in {5120, 6144, 17408} and kp == k and k % 256 == 0,
             "consumer requires model full K256 (5120/6144/17408)")
    n = _integer(a.get("logical_n"), "logical_n", minimum=1)
    _require(p.get("row_origin", 0) == 0 and p.get("active_rows", n) == n
             and p.get("local_n_capacity", n) == n, "incomplete row origin/active owner")
    bn = _integer(a.get("weight_block_n"), "weight_block_n", minimum=1)
    bk = _integer(a.get("weight_block_k"), "weight_block_k", minimum=1)
    weight_formals = ("gate_weight", "up_weight") if family == FAMILIES[2] else ("weight",)
    weights = []
    for formal in weight_formals:
        _require(a.get(formal + "_encoding") == "IQ4_XS", "wrong weight encoding")
        w = _bindings(raw, "inputs", formal)[0]
        _abi(w, dtype="uint8", shape=(n, 136 * (k // 256)), alignment=4)
        _placeholder(raw, formal + "_scale", ((n + bn - 1) // bn, (k + bk - 1) // bk))
        weights.append(w)
    dtype = a.get("output_data_type")
    lanes = a.get("output_lanes")
    if family == FAMILIES[3]:
        _require(dtype == "bfloat16" and n == 12288 and tuple(a.get("feature_shape", ())) == (24, 256)
                 and a.get("field_count") == 2 and lanes in (((8,), (8,)), [[8], [8]]),
                 "wrong interleaved fields/lanes/whole-head shape")
        for b in results:
            _abi(b, dtype="bfloat16", shape=(rows, 24, 32), lanes=(8,), strides=(6144, 256, 8))
    else:
        _require(isinstance(lanes, (list, tuple)) and all(type(v) is int and v > 0 for v in lanes),
                 "missing output lanes")
        lane_count = prod(lanes)
        # r034 union: r033 relaxed the F32-result store so the MTP eh_proj halves
        # (n == 5120) can use the same MODE=0 accumulator store the 248320-row
        # head already used. The r031 target never takes this branch with
        # n != 248320, so the r031 subject is unaffected.
        _require(n % lane_count == 0 and (dtype == "bfloat16" or
                 family == FAMILIES[1] and dtype == "float32" and not lanes),
                 "wrong output dtype/lanes")
        _abi(results[0], dtype=dtype, shape=(rows, n // lane_count), lanes=lanes)
    if family == FAMILIES[2]:
        _require(a.get("activation") == "silu" and a.get("gate_clamp_max") is None
                 and a.get("up_clamp") is None, "GLU requires frozen SiLU gate/up without clamps")
    # Outputs may share an allocation only with disjoint scalar byte ranges.
    _disjoint(results)
    for output in results:
        for read in (qb, *weights):
            _disjoint((read, output))


def _pointer(binding):
    abi = binding["abi"]
    dtype = {"uint8": "tl.uint8", "bfloat16": "tl.bfloat16", "float32": "tl.float32"}[abi["scalar_dtype"]]
    pointer = f"{binding['runtime_argument']}.to(tl.pointer_type({dtype}))"
    # Constants are evaluated as I64 pointer displacements by Triton; never
    # count Q8 bytes as U32 words or apply a fixed subspan again.
    offset = abi["pool_byte_offset"] // abi["scalar_itemsize"]
    return f"({pointer} + tl.full((), {offset}, tl.int64))" if offset else pointer


def materialize(raw):
    """Graph-facing, JSON-safe native lowering description of a checked call."""
    validate_raw(raw)
    f, a = raw["family"], raw["semantic_attrs"]
    results = _results(raw)
    k = a["logical_k"]
    qb = results[0] if f == FAMILIES[0] else _bindings(raw, "inputs", "q8")[0]
    rows = _shape(qb["abi"], "logical_shape")[0]
    if f == FAMILIES[0]:
        pointers = [_pointer(_bindings(raw, "inputs", "value")[0]), _pointer(results[0])]
        constants = {"K": k, "ROWS": rows}
        entry, grid = "q8_prepare", [(a["group_count"] + 3) // 4, 1, 1]
        placeholders = []
    else:
        gate = "gate_weight" if f == FAMILIES[2] else "weight"
        w = _pointer(_bindings(raw, "inputs", gate)[0])
        u = _pointer(_bindings(raw, "inputs", "up_weight")[0]) if f == FAMILIES[2] else w
        y = _pointer(results[0])
        z = _pointer(results[1]) if f == FAMILIES[3] else y
        pointers = [_pointer(_bindings(raw, "inputs", "q8")[0]), w, u, y, z]
        constants = {"N": a["logical_n"], "K": k, "ROWS": rows,
                     "MODE": 1 if f == FAMILIES[2] else 2 if f == FAMILIES[3] else 0}
        entry, grid = "dp4a_m1", [a["logical_n"], 1, 1]
        formals = ("gate_weight_scale", "up_weight_scale") if f == FAMILIES[2] else ("weight_scale",)
        shape = ((a["logical_n"] + a["weight_block_n"] - 1) // a["weight_block_n"],
                 (k + a["weight_block_k"] - 1) // a["weight_block_k"])
        placeholders = [_placeholder(raw, formal, shape) for formal in formals]
    params = implementation_parameters(f)
    options = {key: params[key] for key in ("num_warps", "num_ctas", "num_stages", "maxnreg",
               "enable_fp_fusion", "enable_static_shared", "ptx_options")}
    # The multirow consumer shares weight packets across token accumulators.
    # Keep the single-row subject's compile options exactly as qualified.
    # Long-K down projections benefit from the larger instruction window;
    # short-K projections benefit from more resident CTAs instead.
    if 1 < rows <= 5 and f != FAMILIES[0]:
        options["maxnreg"] = 80 if f == FAMILIES[2] else 80 if k == 17408 else 56
        columns = multirow_column_tile(f, k, a['logical_n'])
        if columns in (2, 4):
            # Concurrent independent columns increase in-flight weight bytes
            # without changing the original K256 lane partition or reduction.
            constants['COLS'] = columns
            entry = 'dp4a_two_columns' if columns == 2 else 'dp4a_columns'
            grid = [(a['logical_n'] + columns - 1) // columns, 1, 1]
            options['maxnreg'] = 80 if columns == 4 else 56
    jit_module = ('triton.flagmega.codegen.triton.iq4_q8_multi_column'
                  if 'COLS' in constants else JIT_MODULE)
    return dict(schema=SCHEMA, jit_module=jit_module, jit_entry=entry,
                jit_source_sha256=JIT_SOURCE_SHA256, pointers=pointers,
                constexprs=constants, grid=grid, compile_options=options,
                program_kind="ordinary", placeholder_bindings=placeholders,
                logical_static_shared_bytes=768 if f == FAMILIES[2] else 0 if f == FAMILIES[0] else 384,
                dynamic_shared_bytes=0)


def prepare(raw):
    _require(raw.get("family") == FAMILIES[0], "wrong prepare encoder family")
    return materialize(raw)


def projection(raw):
    _require(raw.get("family") == FAMILIES[1], "wrong projection encoder family")
    return materialize(raw)


def glu(raw):
    _require(raw.get("family") == FAMILIES[2], "wrong GLU encoder family")
    return materialize(raw)


def interleaved(raw):
    _require(raw.get("family") == FAMILIES[3], "wrong interleaved encoder family")
    return materialize(raw)


def register():
    """Called by graph's normal bootstrap; identical registration is harmless."""
    from .kernel_call_renderers import register_family_encoder
    for family, encoder in zip(FAMILIES, (prepare, projection, glu, interleaved), strict=True):
        register_family_encoder(family, _dispatch, schema=SCHEMA, validate_raw=_dispatch_validate)


# Transport version dispatch is local to the isolated r006 experiment.
def _dispatch_validate(raw):
    if raw.get('variant') == 'r050_projection_consumer_prefetch_v1':
        from .iq4_q8_projection_consumer_renderer import validate_raw as check
        return check(raw)
    if raw.get('variant') == 'r049_glu_consumer_prefetch_v1':
        from .iq4_q8_consumer_prefetch_renderer import validate_raw as check
        return check(raw)
    if raw.get('variant') == 'r048_glu_larger_tiles_v1':
        from .iq4_q8_larger_tiles_renderer import validate_raw as check
        return check(raw)
    if raw.get('variant') == 'r041_glu_transport_t3_compact':
        from .iq4_q8_static_t3_compact_renderer import validate_raw as check
        return check(raw)
    if raw.get('variant') == 'r041_glu_transport_t3':
        from .iq4_q8_static_t3_renderer import validate_raw as check
        return check(raw)
    if raw.get('variant') == 'r040_down_pair_columns_v1':
        from .iq4_q8_pair_renderer import validate_raw as check
        return check(raw)
    if raw.get('variant') == 'r040_q8_prepare_rows_v1':
        from .iq4_q8_prepare_renderer import validate_raw as check
        return check(raw)
    if raw.get('variant') == 'r047_glu_occupancy_v1':
        from .iq4_q8_occupancy_renderer import validate_raw as check
        return check(raw)
    if raw.get('variant') == 'r046_glu_staging_v1':
        from .iq4_q8_staging_renderer import validate_raw as check
        return check(raw)
    if raw.get('variant') == 'r038_glu_global_prefetch_v1':
        from .iq4_q8_global_prefetch_renderer import validate_raw as check
        return check(raw)
    if raw.get('variant') == 'r006_glu_transport_t3':
        from .iq4_q8_r031_transport_renderer import validate_raw as check
        return check(raw)
    if str(raw.get('variant', '')).startswith('r006_glu_transport_t'):
        from .iq4_q8_transport_renderer import validate_raw as check
        return check(raw)
    if raw.get('variant') == 'r006_glu_persistent_v1':
        from .iq4_q8_persistent_renderer import validate_raw as encode
        return encode(raw)
    if raw.get('variant') == 'q8_r006_consumer_barrier_v3':
        from .iq4_q8_pipeline_renderer import validate_raw as check
        return check(raw)
    return validate_raw(raw)


def _dispatch(raw):
    if raw.get('variant') == 'r050_projection_consumer_prefetch_v1':
        from .iq4_q8_projection_consumer_renderer import materialize as encode
        result = encode(raw)
        return {**result, 'schema': SCHEMA, 'transfer_schema': result['schema']}
    if raw.get('variant') == 'r049_glu_consumer_prefetch_v1':
        from .iq4_q8_consumer_prefetch_renderer import materialize as encode
        result = encode(raw)
        return {**result, 'schema': SCHEMA, 'transfer_schema': result['schema']}
    if raw.get('variant') == 'r048_glu_larger_tiles_v1':
        from .iq4_q8_larger_tiles_renderer import materialize as encode
        result = encode(raw)
        return {**result, 'schema': SCHEMA, 'transfer_schema': result['schema']}
    if raw.get('variant') == 'r047_glu_occupancy_v1':
        from .iq4_q8_occupancy_renderer import materialize as encode
        result = encode(raw)
        return {**result, 'schema': SCHEMA, 'transfer_schema': result['schema']}
    if raw.get('variant') == 'r046_glu_staging_v1':
        from .iq4_q8_staging_renderer import materialize as encode
        result = encode(raw)
        return {**result, 'schema': SCHEMA, 'transfer_schema': result['schema']}
    if raw.get('variant') == 'r041_glu_transport_t3_compact':
        from .iq4_q8_static_t3_compact_renderer import materialize as encode
        return {**encode(raw), 'schema': SCHEMA}
    if raw.get('variant') == 'r041_glu_transport_t3':
        from .iq4_q8_static_t3_renderer import materialize as encode
        return {**encode(raw), 'schema': SCHEMA}
    if raw.get('variant') == 'r040_down_pair_columns_v1':
        from .iq4_q8_pair_renderer import materialize as encode
        return encode(raw)
    if raw.get('variant') == 'r040_q8_prepare_rows_v1':
        from .iq4_q8_prepare_renderer import materialize as encode
        return encode(raw)
    if raw.get('variant') == 'r038_glu_global_prefetch_v1':
        from .iq4_q8_global_prefetch_renderer import materialize as encode
        result = encode(raw)
        return {**result, 'schema': SCHEMA, 'transfer_schema': result['schema']}
    if raw.get('variant') == 'r006_glu_transport_t3':
        from .iq4_q8_r031_transport_renderer import materialize as encode
        result = encode(raw)
        return {**result, 'schema': SCHEMA, 'transfer_schema': result['schema']}
    if str(raw.get('variant', '')).startswith('r006_glu_transport_t'):
        from .iq4_q8_transport_renderer import materialize as encode
        result = encode(raw)
        return {**result, 'schema': SCHEMA, 'transfer_schema': result['schema']}
    if raw.get('variant') == 'r006_glu_persistent_v1':
        from .iq4_q8_persistent_renderer import materialize as encode
        result = encode(raw)
        return {**result, 'schema': SCHEMA, 'transfer_schema': result['schema']}
    if raw.get('variant') == 'q8_r006_consumer_barrier_v3':
        from .iq4_q8_pipeline_renderer import materialize as encode
        result = encode(raw)
        # The family dispatcher carries the original registry schema; the
        # independently versioned transfer identity remains in the result.
        return {**result, 'schema': SCHEMA, 'transfer_schema': result['schema']}
    return materialize(raw)

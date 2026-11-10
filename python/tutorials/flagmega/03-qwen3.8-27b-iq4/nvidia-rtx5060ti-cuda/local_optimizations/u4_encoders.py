# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Local U4 family encoders through C0's checked registration hook.

The compiler owns signatures, lifetimes, program descriptors and launches.
V0 owns the nested primitive ABI; D0 owns the serialized task policy.
"""
from collections.abc import Mapping
from math import prod
from triton.flagmega.codegen.triton.physical_access import emit_storage_pointer, emit_buffer_pointer
from .u4_q8_abi import Q8Layout, exact, require, validate_primitive_fields
from .u4_views import TensorLayout
from .u4_mmvq import FAMILIES, implementation

ENCODER_SCHEMA = "flagmega.u4-cpp-fields/v1"
TRACE_SITES = {
    "encode": ((2, "encode_begin"), (3, "encode_stores_issued")),
    "row": ((4, "group_load_compute_begin"), (5, "group_load_compute_end"),
            (6, "partial_stores_issued"), (7, "partial_publication_complete"),
            (8, "combine_begin"), (9, "combine_end"), (10, "output_store_issued")),
    "glu_row4": ((4, "row_private_traversal_begin"), (5, "row_private_traversal_and_chunk_merge_end"),
                 (6, "private_sums_ready"), (7, "no_cta_synchronization"),
                 (8, "warp_combine_begin"), (9, "warp_combine_end"), (10, "row_output_store_issued")),
    "gather": ((11, "z_gather_begin"), (12, "z_gather_private_publication")),
}


def _binding(raw, group, formal):
    entries = [v for v in raw.get(group, ()) if v.get("formal") == formal]
    require(len(entries) == 1 and len(entries[0].get("buffers", ())) == 1,
            f"missing/ambiguous {group} binding {formal}")
    value = entries[0]["buffers"][0]
    require(isinstance(value.get("abi"), Mapping) and isinstance(value.get("runtime_argument"), str),
            "local encoder requires actual typed runtime buffer bindings")
    return value


def _typed(binding, layout, *, canonical=True, alignment=None, local_shape=None, strides=None):
    a = binding["abi"]
    expected = {"schema": "flagmega.local-buffer-abi/v2", "dtype": layout.dtype,
                "scalar_dtype": layout.scalar_dtype, "scalar_itemsize": layout.itemsize,
                "scalar_lane_shape": list(layout.lanes), "scalar_lane_count": layout.lane_count,
                "logical_shape": list(layout.shape), "local_capacity_shape": list(local_shape or layout.shape),
                "scalar_storage_strides": list(strides or layout.strides)}
    require(all(exact(a.get(k), v) for k, v in expected.items()),
            "actual dtype/lanes/shape/stride differs: " + str({k: (a.get(k), v) for k, v in expected.items() if not exact(a.get(k), v)}))
    require(isinstance(a.get("physical_buffer"), str) and a["physical_buffer"], "missing physical root identity")
    require(type(a.get("pool_byte_offset", 0)) is int and a.get("pool_byte_offset", 0) >= 0
            and a.get("pool_byte_offset", 0) % layout.itemsize == 0
            and not a.get("view_byte_offset"), "invalid or unproved pointer offset")
    require(type(a.get("alignment_bytes")) is int and a["alignment_bytes"] >= (alignment or layout.itemsize),
            "required pointer alignment is not proven")
    if canonical:
        require(a.get("coordinate_space") == "canonical_global"
                and not a.get("pool_scope_stride_bytes"), "ordinary U4 buffer must have canonical unscoped addressing")
    return a


def _no_alias(reads, writes):
    for i, output in enumerate(writes):
        for other in (*reads, *writes[:i]):
            a, b = output["abi"], other["abi"]
            require(a["physical_buffer"] != b["physical_buffer"], "mutable U4 root aliases another live value")
            if output["runtime_argument"] != other["runtime_argument"]:
                continue
            def span(v):
                shape = v["logical_shape"] if v.get("coordinate_space") == "canonical_global" else v["local_capacity_shape"]
                count = v["scalar_lane_count"] + sum((n - 1) * s for n, s in zip(shape, v["scalar_storage_strides"]))
                base = v.get("pool_byte_offset", 0)
                scope_stride = v.get("pool_scope_stride_bytes", 0)
                require(type(scope_stride) is int and scope_stride >= 0, "invalid scope stride")
                return base, base + count * v["scalar_itemsize"] + (95 * scope_stride if scope_stride else 0)
            aa, bb = span(a), span(b)
            require(aa[1] <= bb[0] or bb[1] <= aa[0], "live U4 pool spans overlap")


def _shared(raw, fields, components, partial):
    ws = raw.get("shared_workspaces", ())
    if not partial:
        require(not ws, "encode/gather has no declared Shared workspace")
        return
    require(len(ws) == 1, "row kernel requires one typed partial workspace")
    w = ws[0]
    expected = {"name": "row_partials", "dtype": "float32", "shape": [3, fields, components, 32],
                "strides": [fields * components * 32, components * 32, 32, 1],
                "nbytes": 3 * fields * components * 32 * 4, "alignment_bytes": 16}
    require(all(exact(w.get(k), v) for k, v in expected.items())
            and isinstance(w.get("physical_buffer"), str) and w["physical_buffer"]
            and type(w.get("offset_bytes")) is int and w["offset_bytes"] >= 0
            and w["offset_bytes"] == 0 and not w.get("matrix_compatible", False),
            "Shared partial type/span/alignment/identity differs")


def _checked(raw):
    require(isinstance(raw, Mapping), "missing raw compiler call")
    kinds = {v: k for k, v in FAMILIES.items()}
    require(raw.get("family") in kinds and raw.get("variant") == "simt", "unknown local family/variant")
    kind = kinds[raw["family"]]
    params = raw.get("parameters", {})
    if kind == "gather":
        return _gather(raw)
    fields = params.get("u4_primitive")
    layout = validate_primitive_fields(fields)
    require(fields["kind"] == kind, "selected family differs from typed primitive")
    impl = implementation(kind, layout.k, layout.components)
    require(raw.get("implementation") == impl.id and all(exact(params.get(k), v) for k, v in impl.parameters.items()),
            "selected U4 implementation identity/parameters differ")
    require(all(exact(raw.get("facts", {}).get(k), v) for k, v in impl.facts.items())
            and raw.get("transfer_pipeline") is None, "U4 facts or unsupported pipeline differ")
    attrs = raw.get("semantic_attrs", {})
    require(all(exact(attrs.get(k), fields["q8"][k]) for k in ("k", "components", "format_id", "numerical_id"))
            and attrs.get("source_epoch") == fields["source_epoch"]
            and exact(attrs.get("q8_schema"), fields["q8"])
            and attrs.get("encoding_id") == "f32-amax127-roundf-residual-rn/v1", "semantic Q8 identity differs")
    from .u4_execution_plan import q8_task_policy, row_task_policy
    policy = q8_task_policy(layout.k, layout.components) if kind == "encode" else row_task_policy(
        layout.k, fields["native_rows"], components=layout.components, fields=2 if kind == "glu" else 1)
    require(exact(params.get("u4_task_policy"), policy.to_data())
            and exact(params.get("task_mapping"), policy.recipe.to_data()), "task owner/launch policy differs")
    pointers, reads, writes = {}, [], []
    if kind == "encode":
        source, q8 = _binding(raw, "inputs", "value"), _binding(raw, "outputs", "result")
        _typed(source, TensorLayout("bfloat16", (1, layout.k)))
        _typed(q8, TensorLayout("uint8", layout.shape), alignment=16)
        pointers.update(value=source, q8=q8)
        reads, writes = [source], [q8]
    else:
        q8 = _binding(raw, "inputs", "q8")
        _typed(q8, TensorLayout("uint8", layout.shape), alignment=16)
        pointers["q8"] = q8
        reads.append(q8)
        for role in ("gate_weight", "up_weight") if kind == "glu" else ("weight",):
            weight = _binding(raw, "inputs", role)
            abi = _typed(weight, TensorLayout("uint8", (fields["native_rows"], layout.native_row_bytes)), alignment=1)
            require(abi.get("storage") in ("rdata", "weight", "weights", "readonly", "parameter"),
                    "native IQ4 weight is not an external readonly bank")
            pointers[role] = weight
            reads.append(weight)
        # Scale placeholders are real formals; native IQ4 scales remain in bytes.
        for role in ("gate_scale", "up_scale") if kind == "glu" else ("weight_scale",):
            scale = _binding(raw, "inputs", role)
            bn, bk = attrs.get("weight_block_n"), attrs.get("weight_block_k")
            require(type(bn) is int and bn > 0 and type(bk) is int and bk > 0, "missing native block shape")
            _typed(scale, TensorLayout("float32", ((fields["native_rows"] + bn - 1) // bn,
                                                   (layout.k + bk - 1) // bk)), alignment=1)
            reads.append(scale)
        require(attrs.get("weight_encoding") == "IQ4_XS", "native encoding differs")
        for i, (role, data) in enumerate(zip(fields["result_roles"], fields["output_layouts"])):
            output = _binding(raw, "outputs", "result_" + str(i) if kind == "interleaved" else "result")
            _typed(output, TensorLayout.from_data(data))
            pointers[role] = output
            writes.append(output)
    if kind == "glu":
        require(attrs.get("activation") == "silu", "GLU activation differs from frozen epilogue")
        require(all(isinstance(raw.get(name, ()), (list, tuple)) and not raw.get(name, ())
                    for name in ("shared_workspaces", "workspaces")),
                "GLU row4 requires empty shared/global workspace sequences")
    if kind == "matmul":
        result_layout = TensorLayout.from_data(fields["output_layouts"][0])
        require(attrs.get("output_dtype") == result_layout.scalar_dtype
                and exact(attrs.get("output_lanes"), list(result_layout.lanes)), "matrix output storage contract differs")
    if kind == "interleaved":
        require(exact(attrs.get("feature_shape"), [24, 256]) and exact(attrs.get("output_lanes"), [[8], [8]]),
                "interleaved tuple coordinate contract differs")
    require(len(raw.get("inputs", ())) == len(fields["input_roles"])
            and len(raw.get("outputs", ())) == len(fields["result_roles"]), "unexpected U4 call operands")
    _no_alias(reads, writes)
    f = 2 if kind == "glu" else 1
    _shared(raw, f, layout.components, kind in ("matmul", "interleaved"))
    return dict(schema=ENCODER_SCHEMA, u4_kind=kind, u4_k=layout.k, u4_components=layout.components,
        u4_fields=f, u4_rows=fields["native_rows"], u4_row_bytes=layout.native_row_bytes,
        u4_primitive=fields, cpp_task_mapping="independent-cta", task_launch_recipe=policy.recipe.to_data(),
        cpp_trace_sites=[dict(id=i, name=n) for i, n in TRACE_SITES["encode" if kind == "encode" else "glu_row4" if kind == "glu" else "row"]],
        u4_pointers={k: emit_storage_pointer(v["abi"], v["runtime_argument"]) for k, v in pointers.items()},
        shared_workspaces=list(raw.get("shared_workspaces", ())))


# The callback ABI contains a callee-relative pool pointer. Only the surrounding
# normal module/caller plan can prove its enclosing allocation. ContextVar keeps
# proof state private to one render operation and restores it on errors/nesting.
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy

_GATHER_CONTEXT = ContextVar("u4_gather_context", default=None)


def _gather_key(raw):
    return (raw.get("call"), _binding(raw, "inputs", "value").get("actual"),
            _binding(raw, "outputs", "result").get("actual"))


@contextmanager
def register_gather_context(module):
    """Prove actual nested caller frames before normal raw family encoding.

    I0 wraps the ordinary ``describe_cpp_package(module)`` call with this
    context. No raw count/stride, BufferPlan, pointer, or descriptor is changed.
    Unsupported deeper/loop call chains fail rather than inventing a frame.
    The yielded receipt is diagnostic data, never a token accepted by encoders.
    """
    from triton.flagmega.ir import verify_module, verify_buffer_plan
    from triton.flagmega.codegen.triton.runtime_binding import describe_function_runtime_binding
    from triton.flagmega.ir.program_execution import ProgramExecution
    from .u4_execution_plan import flatten_execution
    from .u4_views import GDNZCallScope, bind_gdn_z_gather, bind_gdn_z_reader, validate_call_view

    verify_module(module)
    plan = verify_buffer_plan(module)
    bindings = {name: describe_function_runtime_binding(module, function_name=name)
                for name in plan.function_map}
    gathered_calls = [(name, raw) for name, binding in bindings.items()
                      for raw in binding["call_abi"]["kernel_calls"]
                      if raw.get("family") == FAMILIES["gather"]]
    scoped = [(name, raw) for name, raw in gathered_calls
              if _binding(raw, "outputs", "result")["abi"].get("pool_scope_local") is True]
    proofs = {_gather_key(raw): deepcopy(raw) for _, raw in gathered_calls if
              _binding(raw, "outputs", "result")["abi"].get("pool_scope_local") is not True}
    records = []
    if scoped:
        execution = ProgramExecution.from_data(module.metadata["kernel_program_execution"])
        trace, invocations = flatten_execution(execution)
        require(execution.entry == module.entry, "gather execution entry differs from actual module")
        partitions = module.metadata["kernel_program_partition"]
        owners = {(p["function"], call): p["symbol"] for p in partitions for call in p["calls"]}
        for function, raw in scoped:
            key = _gather_key(raw)
            require(key not in proofs, "ambiguous scoped gather definition")
            src, dst = _binding(raw, "inputs", "value"), _binding(raw, "outputs", "result")
            source, destination = plan.buffer_map[src["actual"]], plan.buffer_map[dst["actual"]]
            callee_pool = next(p for p in bindings[function]["pools"] if p["storage"] == "block_local_data")
            readers = []
            for call in bindings[function]["call_abi"]["kernel_calls"]:
                if call.get("family") != "gdn_recurrent":
                    continue
                z = _binding(call, "inputs", "z")
                if z["abi"].get("physical_buffer") == destination.physical_id:
                    readers.append((call, z))
            require(len(readers) == 1, "scoped gather requires its actual retained GDN reader")
            gdn, reader = readers[0]
            relevant = [path for path, callee in invocations if callee == function]
            require(bool(relevant), "scoped gather has no actual reachable caller")
            for path in relevant:
                require(len(path) == 1, "nested gather proof supports one actual caller frame only")
                call = plan.call_map[path[0]]
                require(call.caller == module.entry and call.callee == function, "gather caller/callee binding differs")
                parent_pool = next(p for p in bindings[call.caller]["pools"] if p["storage"] == "block_local_data")
                scope = GDNZCallScope(plan, call, parent_pool, module.semantic_hash + "/" + call.call)
                _, gathered = bind_gdn_z_gather(source, destination, plan,
                    source_abi=src["abi"], destination_abi=dst["abi"], runtime_pool=callee_pool,
                    frame=scope.frame, epoch=scope.epoch, call_scope=scope)
                view = bind_gdn_z_reader(gathered, plan.buffer_map[reader["actual"]], reader["abi"])
                validate_call_view(gathered, view, plan.buffer_map[reader["actual"]])
                indices = [next(i for i, (symbol, frame) in enumerate(trace)
                                if frame == path and symbol == owners[(function, name)])
                           for name in (src["actual"], raw["call"], gdn["call"])]
                require(indices[0] < indices[1] == indices[2],
                        "gather must follow completed source and share the retained GDN entry")
                resolved = scope.resolve(destination)
                records.append(dict(call=call.call, gather=raw["call"], reader=reader["actual"],
                    frame=scope.frame, source_entry=indices[0], gather_gdn_entry=indices[1],
                    physical_root=resolved.physical_id, byte_offset=resolved.offset,
                    physical_scope_count=gathered.scope_count, physical_scope_pitch=gathered.scope_stride_bytes,
                    raw_scope_count=dst["abi"]["pool_scope_count"], raw_scope_local=dst["abi"]["pool_scope_local"]))
            proofs[key] = deepcopy(raw)
    token = _GATHER_CONTEXT.set(proofs)
    try:
        yield dict(schema="flagmega.u4-gather-context/v1", module=module.semantic_hash,
                   definitions=len(proofs), invocations=records)
    finally:
        _GATHER_CONTEXT.reset(token)


def _require_scoped_gather_proof(raw):
    proofs = _GATHER_CONTEXT.get()
    require(proofs is not None and _gather_key(raw) in proofs
            and exact(raw, proofs[_gather_key(raw)]),
            "scope-local gather requires matching actual module/caller/root/lifetime proof")


def _gather(raw):
    params, attrs = raw.get("parameters", {}), raw.get("semantic_attrs", {})
    fullhead48 = raw.get("implementation") == "tir.local_u4.gdn_z_gather.fullhead48.v1"
    if fullhead48:
        return _gather_fullhead48(raw)
    impl = implementation("gather")
    require(raw.get("implementation") == impl.id and params.get("u4_fuse_into_fixed96") is True
            and params.get("u4_source_epoch") == attrs.get("source_epoch")
            and isinstance(attrs.get("source_epoch"), str) and attrs["source_epoch"]
            and attrs.get("mapping_id") == "gdn-z-c64-private/v1", "Z gather identity/fusion/epoch differs")
    require(all(exact(params.get(k), v) for k, v in impl.parameters.items())
            and all(exact(raw.get("facts", {}).get(k), v) for k, v in impl.facts.items()), "Z gather selected facts differ")
    source, dest = _binding(raw, "inputs", "value"), _binding(raw, "outputs", "result")
    _typed(source, TensorLayout("bfloat16", (1, 6144)))
    abi = _typed(dest, TensorLayout("bfloat16", (1, 6144)), canonical=False,
                 local_shape=(1, 64), strides=(64, 1))
    from triton.flagmega.ir import type_from_data
    from .u4_transitions import gdn_z_reader_type
    from triton.flagmega.ir import tensor_type
    require(abi.get("coordinate_space") in ("local", "parent_shard_local")
            and abi.get("memory_sharing_scope") == "block" and abi.get("memory_space") == "block_local_data"
            and abi.get("storage_kind") == "compact_local"
            and type_from_data(abi.get("distributed_type")) == gdn_z_reader_type(tensor_type("bfloat16", (1, 6144))),
            "Z gather must retain exact96 private64 roots/owners; GDN reader view remains separate")
    if _GATHER_CONTEXT.get() is not None:
        _require_scoped_gather_proof(raw)
    if abi.get("pool_scope_local") is True:
        require(abi.get("pool_scope_count") == 1 and not abi.get("pool_scope_stride_bytes"),
                "scope-local raw pointer must retain one relative frame without owner offset")
        _require_scoped_gather_proof(raw)
    else:
        require(abi.get("pool_scope_count") == 96 and abi.get("pool_scope_stride_bytes", 0) >= 128,
                "full-entry gather requires all96 explicit physical scopes")
    require(len(raw.get("inputs", ())) == len(raw.get("outputs", ())) == 1
            and not raw.get("workspaces") and raw.get("transfer_pipeline") is None,
            "Z gather has unexpected buffers/pipeline")
    _no_alias((source,), (dest,))
    _shared(raw, 1, 1, False)
    return dict(schema=ENCODER_SCHEMA, u4_kind="gather", u4_fuse_into_fixed96=True,
        u4_pointers={"value": emit_storage_pointer(source["abi"], source["runtime_argument"]),
                     "result": emit_buffer_pointer(abi, dest["runtime_argument"])},
        cpp_trace_sites=[dict(id=i, name=n) for i, n in TRACE_SITES["gather"]])


def _gather_fullhead48(raw):
    params, attrs = raw.get("parameters", {}), raw.get("semantic_attrs", {})
    impl = implementation("gather", fullhead48=True)
    require(raw.get("implementation") == impl.id and params.get("u4_fuse_into_fullhead48") is True
            and params.get("u4_source_epoch") == attrs.get("source_epoch")
            and isinstance(attrs.get("source_epoch"), str) and attrs["source_epoch"]
            and attrs.get("mapping_id") == "gdn-z-c128-private/v1", "Z gather identity/fusion/epoch differs")
    require(all(exact(params.get(k), v) for k, v in impl.parameters.items())
            and all(exact(raw.get("facts", {}).get(k), v) for k, v in impl.facts.items()), "Z gather selected facts differ")
    source, dest = _binding(raw, "inputs", "value"), _binding(raw, "outputs", "result")
    _typed(source, TensorLayout("bfloat16", (1, 6144)))
    abi = _typed(dest, TensorLayout("bfloat16", (1, 6144)), canonical=False,
                 local_shape=(1, 128), strides=(128, 1))
    from triton.flagmega.ir import type_from_data
    from .u4_transitions import gdn_z_reader_type
    from triton.flagmega.ir import tensor_type
    require(abi.get("coordinate_space") in ("local", "parent_shard_local")
            and abi.get("memory_sharing_scope") == "block" and abi.get("memory_space") == "block_local_data"
            and abi.get("storage_kind") == "compact_local"
            and type_from_data(abi.get("distributed_type")) == gdn_z_reader_type(tensor_type("bfloat16", (1, 6144)), fullhead48=True),
            "Z gather must retain exact48 private128 roots/owners; GDN reader view remains separate")
    if _GATHER_CONTEXT.get() is not None:
        _require_scoped_gather_proof(raw)
    if abi.get("pool_scope_local") is True:
        require(abi.get("pool_scope_count") == 1 and not abi.get("pool_scope_stride_bytes"),
                "scope-local raw pointer must retain one relative frame without owner offset")
        _require_scoped_gather_proof(raw)
    else:
        require(abi.get("pool_scope_count") == 48 and abi.get("pool_scope_stride_bytes", 0) >= 256,
                "full-entry gather requires all48 explicit physical scopes")
    require(len(raw.get("inputs", ())) == len(raw.get("outputs", ())) == 1
            and not raw.get("workspaces") and raw.get("transfer_pipeline") is None,
            "Z gather has unexpected buffers/pipeline")
    _no_alias((source,), (dest,))
    _shared(raw, 1, 1, False)
    return dict(schema=ENCODER_SCHEMA, u4_kind="gather", u4_fuse_into_fullhead48=True, u4_private_values_per_cta=128,
        u4_pointers={"value": emit_storage_pointer(source["abi"], source["runtime_argument"]),
                     "result": emit_buffer_pointer(abi, dest["runtime_argument"])},
        cpp_trace_sites=[dict(id=i, name=n) for i, n in TRACE_SITES["gather"]])


def validate_raw(raw):
    _checked(raw)


def encode_call(raw):
    return _checked(raw)


def register_encoders():
    from triton.flagmega.codegen.triton.kernel_call_renderers import register_family_encoder
    for family in FAMILIES.values():
        register_family_encoder(family, encode_call, schema=ENCODER_SCHEMA, validate_raw=validate_raw)

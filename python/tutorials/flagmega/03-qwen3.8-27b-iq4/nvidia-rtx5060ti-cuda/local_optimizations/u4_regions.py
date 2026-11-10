"""Complete stats_combined U4-L rewrite, preserving original caller/state roots."""
from collections import Counter
from dataclasses import replace
from math import prod

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import Node, PURE, verify_module, VectorType
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.execution_mesh import ExecutionMesh, with_execution_mesh
from triton.flagmega.ir.ops.core import get_definition
from triton.flagmega.ir.model import contains_distributed
from .u4_ops import (
    Q8Encode, IQ4Matmul, IQ4Glu, IQ4Interleaved, GDNZGather, encoding_attrs,
    FORMAT_ID, NUMERICAL_ID,
)

# Exact post-packing site identities from the frozen t9 complete envelope.
# No suffix or shape-only matching can accidentally select GDN's internal work.
SITES = {
    "decode_linear_0_in_proj_qkv": ("linear_qkv", 5120, 1, 10240),
    "decode_linear_0_in_proj_z": ("linear_z", 5120, 1, 6144),
    "decode_linear_0_out_proj": ("linear_out", 6144, 1, 5120),
    "decode_linear_0_mlp_gate_up": ("linear_glu", 5120, 2, 17408),
    "decode_linear_0_mlp_down": ("linear_down", 17408, 1, 5120),
    "decode_attention_1_k.packed": ("attention_k", 5120, 1, 1024),
    "decode_attention_1_v.packed": ("attention_v", 5120, 1, 1024),
    "decode_attention_1_q.packed": ("attention_q_gate", 5120, 2, 12288),
    "decode_attention_1_attention_output": ("attention_out", 6144, 1, 5120),
    "decode_attention_1_mlp_gate_up": ("attention_glu", 5120, 2, 17408),
    "decode_attention_1_mlp_down": ("attention_down", 17408, 1, 5120),
    "logits": ("logits", 5120, 1, 248320),
}
MOTIF = frozenset(k for k, (site, *_) in SITES.items() if site.endswith(("_glu", "_down")))
MATRIX_OPS = frozenset(("math.block_scaled_matmul", "nn.matmul_glu", "nn.interleaved_projection"))
U4_MATRIX_OPS = frozenset(("u4.iq4_matmul", "u4.iq4_glu", "u4.iq4_interleaved"))


def function_body(module, function):
    """Dataflow closure, stopping at formal roots; never enter a callee body."""
    pending = list(function.outputs)
    seen = set(function.parameters)
    while pending:
        node_id = pending.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        if node_id not in module.node_map:
            raise IRSchemaError(f"U4 function {function.name} references missing SSA {node_id}.")
        pending.extend(module.node_map[node_id].inputs)
    return seen


def _inventory(module):
    if module.entry != "main" or set(module.function_map) != {"main", "decode_linear_0", "decode_attention_1"}:
        raise IRSchemaError("U4 requires the complete main + two reusable layer functions.")
    bodies = {f.name: function_body(module, f) for f in module.functions}
    calls = [n for n in module.nodes if n.op == "builtin.call" and n.id in bodies["main"]]
    count = Counter(n.attrs["callee"] for n in calls)
    if count != Counter({"decode_linear_0": 48, "decode_attention_1": 16}):
        raise IRSchemaError("U4 requires all64 original layer calls (48 linear/16 attention).")
    if any(n.op == "control.for" for n in module.nodes):
        raise IRSchemaError("U4 frozen caller envelope contains no loops.")
    owners = {}
    for node_id in SITES:
        names = [name for name, body in bodies.items() if node_id in body]
        if len(names) != 1:
            raise IRSchemaError(f"U4 site {node_id} must belong to exactly one complete function.")
        owners[node_id] = names[0]
    return bodies, calls, owners


def _make(definition, node_id, inputs, attrs, metadata=None):
    prepared = definition.prepare(inputs, attrs)
    return Node(node_id, definition.op_name, tuple(n.id for n in inputs),
                prepared.result_type, prepared.effect, prepared.attrs, metadata or {})


def _fresh(name, occupied):
    index, result = 0, name
    while result in occupied:
        index += 1
        result = f"{name}.{index}"
    occupied.add(result)
    return result


def _validate_site(node, specification, module):
    site, k, c, rows = specification
    expected_op = ("nn.matmul_glu" if site.endswith("_glu") else
                   "nn.interleaved_projection" if site == "attention_q_gate" else
                   "math.block_scaled_matmul")
    if node.op != expected_op or node.effect != PURE:
        raise IRSchemaError(f"U4 site {node.id} changed operation/effect.")
    inputs = [module.node_map[x] for x in node.inputs]
    if tensor_of(inputs[0].type).dtype.value != "bfloat16" or tuple(
            x.fixed_value for x in tensor_of(inputs[0].type).shape) != (1, k):
        raise IRSchemaError(f"U4 site {node.id} lacks its complete BF16 input boundary.")
    if tensor_of(inputs[1].type).shape[0].fixed_value != rows:
        raise IRSchemaError(f"U4 site {node.id} native row count changed.")
    encoding_keys = ("gate_weight_encoding", "up_weight_encoding") if expected_op == "nn.matmul_glu" else ("weight_encoding",)
    if any(node.attrs.get(key) != "IQ4_XS" for key in encoding_keys):
        raise IRSchemaError(f"U4 site {node.id} is not native IQ4_XS.")
    if node.attrs.get("weight_dequant_dtype") is not None or node.attrs.get("pre_ops") or node.attrs.get("post_ops"):
        raise IRSchemaError(f"U4 site {node.id} has an unapproved precision/fusion contract.")
    if expected_op == "nn.matmul_glu" and (node.attrs.get("activation") != "silu"
            or node.attrs.get("gate_clamp_max") is not None or node.attrs.get("up_clamp") is not None):
        raise IRSchemaError("U4 GLU must preserve the frozen unclamped complete-row SiLU.")
    return inputs


def rewrite_u4_regions(module, *, scope="motif", input_norm_q8=False):
    """Rewrite only a complete pre-distribution module; returns the same stage.

    Apply directly or via register_u4_stage(). Post-packing scalar BF16 views
    remain intact. Recompute every subsequent distribution/physical plan.
    """
    if input_norm_q8 and scope != "full12":
        raise IRSchemaError("A49 input NormApply Q8 requires full12 scope.")
    if scope not in ("motif", "full12"):
        raise IRSchemaError("U4 scope must be motif or full12.")
    if module.stage != "stats_combined" or any((module.prim_functions, module.execution_functions,
                                               module.kernel_definitions)):
        raise IRSchemaError("U4 replay requires clean complete stats_combined before constant freeze/distribution.")
    # Completed upstream representation choices are history carried by normal
    # stats_combined dumps. Preserve them verbatim; downstream choices are
    # invalidated by this rewrite and cannot be reused under a forged stage.
    upstream = {point.id for point in module.selection_points
                if point.kind in {"vectorization", "packing"}}
    if len(upstream) != len(module.selection_points) or upstream != {r.point_id for r in module.selections}:
        raise IRSchemaError("U4 replay permits only resolved upstream vectorization/packing selections.")
    from triton.flagmega.passes.constants import constant_phase, ConstantPhase
    if constant_phase(module) != ConstantPhase.OPEN or any(
            contains_distributed(n.type) or n.op.startswith(("distributed.", "tir.")) for n in module.nodes):
        raise IRSchemaError("U4 replay requires open logical stats_combined before constant freeze/distribution.")
    verify_module(module)
    if "u4_rewrite_scope" in module.metadata:
        if module.metadata["u4_rewrite_scope"] != scope:
            raise IRSchemaError("Change U4 scope by replaying the original stats_combined module.")
        verify_u4_contracts(module)
        if input_norm_q8:
            from .u4_norm_q8_regions import rewrite_input_norm_q8
            return rewrite_input_norm_q8(module)
        return module
    bodies, calls, owners = _inventory(module)
    if set(n.id for n in module.nodes if n.op in MATRIX_OPS) != set(SITES):
        raise IRSchemaError("U4 requires exactly the frozen12 matrix sites; unexpected/unreachable matrices reject.")
    # Validate the full envelope even when only the motif is selected.
    for node_id, specification in SITES.items():
        _validate_site(module.node_map[node_id], specification, module)
    selected = MOTIF if scope == "motif" else frozenset(SITES)
    occupied = set(module.node_map)
    nodes, rebuilt, encodings = [], {}, {}
    records = []
    from .u4_q8_abi import primitive_fields
    from .u4_views import output_layouts
    for old in module.nodes:
        if old.id not in selected:
            nodes.append(old)
            rebuilt[old.id] = old
            continue
        site, k, c, rows = SITES[old.id]
        args = tuple(rebuilt.get(x, module.node_map[x]) for x in old.inputs)
        source = args[0]
        epoch = f"{owners[old.id]}:invocation:{source.id}"
        attrs = encoding_attrs(k=k, components=c, source_epoch=epoch)
        key = (owners[old.id], source.id, epoch, k, c, FORMAT_ID, NUMERICAL_ID)
        q8 = encodings.get(key)
        if q8 is None:
            q8_id = _fresh(f"{source.id}.u4_q8_c{c}", occupied)
            meta = {"canonical_global": True, "u4_opaque": True, "u4_q8_contract": attrs,
                    "u4_source_ssa": source.id, "kernel_program_kind": "ordinary",
                    "u4_primitive": primitive_fields(kind="encode", k=k, components=c,
                        source_ssa=source.id, source_epoch=epoch)}
            q8 = with_execution_mesh(_make(Q8Encode, q8_id, (source,), attrs, meta), ExecutionMesh.dynamic())
            encodings[key] = q8
            nodes.append(q8)
            rebuilt[q8.id] = q8
        matrix_attrs = {**attrs, "weight_block_n": old.attrs["weight_block_n"],
                        "weight_block_k": old.attrs["weight_block_k"], "weight_encoding": "IQ4_XS"}
        if old.op == "nn.matmul_glu":
            definition, kind = IQ4Glu, "glu"
        elif old.op == "nn.interleaved_projection":
            definition, kind = IQ4Interleaved, "interleaved"
            matrix_attrs.update(feature_shape=old.attrs["feature_shape"], output_lanes=old.attrs["output_lanes"])
        else:
            definition, kind = IQ4Matmul, "matmul"
            dtype = tensor_of(old.type).dtype
            matrix_attrs.update(output_dtype=(dtype.elem_type if isinstance(dtype, VectorType) else dtype).value,
                                output_lanes=dtype.lanes if isinstance(dtype, VectorType) else ())
        meta = {**old.metadata, "canonical_global": True, "kernel_program_kind": "ordinary",
                "u4_site": site, "u4_source_ssa": source.id,
                "u4_primitive": primitive_fields(kind=kind, k=k, components=c,
                    source_ssa=source.id, source_epoch=epoch, output_layouts=output_layouts(site))}
        # D0 materializes the preferred distributed.sharded_view from actual
        # solver reader types, retaining this canonical full-span requirement.
        if site.endswith(("_down", "_out")):
            meta["u4_boundary"] = "canonical-to-fixed96-cyclic32"
        elif site == "attention_q_gate":
            meta["u4_boundary"] = "unique-canonical-q-gate-distinct-readonly-roots"
        replacement_id = _fresh(old.id + ".u4_canonical", occupied) if site == "linear_z" else old.id
        new = with_execution_mesh(_make(definition, replacement_id, (q8, *args[1:]),
                                       matrix_attrs, meta), ExecutionMesh.dynamic())
        if new.type != old.type:
            raise IRSchemaError(f"U4 changed original result/tuple type at {old.id}.")
        nodes.append(new)
        rebuilt[new.id] = new
        if site == "linear_z":
            gather = _make(GDNZGather, old.id, (new,), {"source_epoch": epoch},
                           {"u4_site": "linear_z_gather", "u4_boundary": "gdn-parent-shard-local",
                            "u4_fuse_into_fixed96": True, "u4_private_values_per_cta": 64})
            gather = with_execution_mesh(gather, ExecutionMesh.fixed((4, 24), axes=("y", "x")))
            nodes.append(gather)
            rebuilt[old.id] = gather
        else:
            rebuilt[old.id] = new
        records.append({"site": site, "node": new.id, "original": old.id,
                        "q8": q8.id, "source": source.id, "source_epoch": epoch,
                        "function": owners[old.id]})
    result = replace(module, nodes=tuple(nodes), metadata={
        **module.metadata, "u4_rewrite_scope": scope, "u4_site_records": records,
        "u4_numerical_id": NUMERICAL_ID,
        "u4_original_call_ids": tuple(n.id for n in calls),
        "u4_original_call_callees": tuple(n.attrs["callee"] for n in calls),
    })
    # Function signatures, call operands/effects, all non-matrix operations
    # (including every state root, logits consumer and host-token boundary)
    # are structurally unchanged.
    assert result.functions == module.functions
    assert all(result.node_map[n.id] == n for n in module.nodes if n.id not in selected)
    verify_module(result)
    verify_u4_contracts(result)
    if input_norm_q8:
        from .u4_norm_q8_regions import rewrite_input_norm_q8
        return rewrite_input_norm_q8(result)
    return result


def verify_u4_contracts(module):
    """Fail closed on schema drift, Q8 packing/escape, envelope loss and CSE."""
    from .u4_ops import _q8, verify_q8_provenance
    bodies, calls, _ = _inventory(module)
    scope = module.metadata.get("u4_rewrite_scope")
    expected_sites = MOTIF if scope == "motif" else frozenset(SITES) if scope == "full12" else None
    if expected_sites is None:
        raise IRSchemaError("Missing U4 scope.")
    records = module.metadata["u4_site_records"]
    if {r["original"] for r in records} != expected_sites:
        raise IRSchemaError("U4 selected site coverage changed.")
    if tuple(n.id for n in calls) != tuple(module.metadata["u4_original_call_ids"]):
        raise IRSchemaError("U4 original caller ordering/identity changed.")
    if tuple(n.attrs["callee"] for n in calls) != tuple(module.metadata["u4_original_call_callees"]):
        raise IRSchemaError("U4 original caller callee ordering changed.")
    fused_records = module.metadata.get("u4_input_norm_q8_records", ())
    fused_q8 = {r[key]: r for r in fused_records for key in ("q8", "q8_c2") if key in r}
    q8s = {n.id: n for n in module.nodes if n.op == "u4.q8_encode" or n.id in fused_q8}
    records_by_node = {r["node"]: r for r in records}
    for node in module.nodes:
        if node.op in U4_MATRIX_OPS:
            if node.id not in records_by_node:
                raise IRSchemaError("Unregistered U4 matrix site.")
            verify_q8_provenance(module, module.node_map[node.inputs[0]], node.attrs)
        for argument in node.inputs:
            if argument in q8s and node.op not in U4_MATRIX_OPS:
                if (module.metadata.get("u4_input_norm_q8_c1c2") is not None
                        and node.op == "u4.gdn_recurrent_core_q8_c2"
                        and node.inputs[9] == argument and argument not in node.inputs[:9]):
                    verify_q8_provenance(module, module.node_map[argument], node.attrs)
                    continue
                raise IRSchemaError("Opaque Q8 may only feed a typed encoded consumer; packing/escape is forbidden.")
    for q8 in q8s.values():
        owners = [f for f, body in bodies.items() if q8.id in body]
        source = fused_q8[q8.id]["scalar_view"] if q8.id in fused_q8 else q8.inputs[0]
        contract = q8.metadata["u4_q8_contract"] if q8.id in fused_q8 else q8.attrs
        if len(owners) != 1 or contract["source_epoch"] != owners[0] + ":invocation:" + source:
            raise IRSchemaError("U4 Q8 cannot be hoisted across function/layer invocation epochs.")
        get_definition(q8.op).verify(q8, module)
        if source not in bodies[owners[0]]:
            # A56 attention no longer reads normalized BF16 after both encoders
            # fuse. Its actual tuple field0 remains the source value even when
            # the scalar projection has no readers. Follow that real field
            # dependency to the same reachable producer; metadata alone fails.
            r = fused_q8.get(q8.id)
            scalar = module.node_map.get(source)
            normalized = module.node_map.get(r["normalized"]) if r else None
            if not (module.metadata.get("u4_input_norm_q8_c1c2") is not None and r
                    and r["fused"] in bodies[owners[0]] and scalar is not None and normalized is not None
                    and scalar.op == "tensors.bitcast" and scalar.inputs == (normalized.id,)
                    and scalar.attrs == {"dtype": "bfloat16"}
                    and normalized.op == "builtin.get_item" and normalized.attrs == {"index": 0}
                    and normalized.inputs == (r["fused"],)):
                raise IRSchemaError("U4 source escaped its function.")
    multiplicity = {"main": 1, "decode_linear_0": 48, "decode_attention_1": 16}
    matrix_calls = sum(multiplicity[r["function"]] for r in records)
    encoded_calls = sum(multiplicity[next(f for f, body in bodies.items() if q.id in body)] for q in q8s.values())
    expected = (4, 4, 128, 128) if scope == "motif" else (12, 10, 337, 273)
    if scope == "full12" and module.metadata.get("u4_input_norm_q8_c1c2") is not None:
        expected = (12, 11, 337, 321)
    actual = (len(records), len(q8s), matrix_calls, encoded_calls)
    if actual != expected:
        raise IRSchemaError(f"U4 rewrite counts/sharing {actual}, expected {expected}.")
    fusion_report = {}
    if module.metadata.get("u4_input_norm_q8_c1c2") is not None:
        from .u4_norm_q8_c1_c2_regions import verify_input_norm_common_q8
        fusion_report = verify_input_norm_common_q8(module)
    elif module.metadata.get("u4_input_norm_q8") is not None:
        from .u4_norm_q8_regions import verify_input_norm_q8
        fusion_report = verify_input_norm_q8(module)
    return {**fusion_report, "scope": scope, "matrix_sites": len(records), "q8_sites": len(q8s),
            "matrix_calls": matrix_calls, "q8_calls": encoded_calls,
            "layer_calls": len(calls), "z_gather_calls": 48 if scope == "full12" else 0}


def make_canonical_reader_view(source, reader_type, *, name):
    """D0 hook after owner inference; creates a real verified read-only view.

    Caller must replace the reader edge and regenerate buffer/alias plans.
    This cannot be called with an old compact-local pointer substitution.
    """
    from triton.flagmega.ir.ops.distributed.sharded_view import ShardedView
    if not source.metadata.get("canonical_global"):
        raise IRSchemaError("U4 reader view requires a canonical producer.")
    return _make(ShardedView, name, (source,), {"new_type": reader_type},
                 {"u4_canonical_root": source.id, "canonical_global": True})


def register_u4_stage(*, scope="motif", input_norm_q8=False):
    """Return/register a normal Stage; I0 inserts it immediately at stats_combined."""
    if scope not in ("motif", "full12"):
        raise IRSchemaError("U4 scope must be motif or full12.")
    from triton.flagmega.stages import Stage, register_stage, get_stage
    if input_norm_q8 and scope != "full12":
        raise IRSchemaError("A49 input NormApply Q8 stage requires full12.")
    name = f"form-u4-{scope}" + ("-input-norm-q8-c1" if input_norm_q8 else "")
    try:
        return get_stage(name)
    except Exception as error:
        from triton.flagmega.errors import StageError
        if not isinstance(error, StageError):
            raise
    stage = Stage(name, "stats_combined", "stats_combined",
                  lambda module, target: rewrite_u4_regions(module, scope=scope, input_norm_q8=input_norm_q8))
    register_stage(stage)
    return stage



def rewrite_input_norm_q8(module):
    from .u4_norm_q8_regions import rewrite_input_norm_q8 as rewrite
    return rewrite(module)

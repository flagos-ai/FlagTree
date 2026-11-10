"""Exact two input-region rewrite after full12 formation; no physical edits."""
from dataclasses import replace
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import PURE, verify_module
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.execution_mesh import ExecutionMesh, with_execution_mesh
from triton.flagmega.ir.ops.core import get_definition
from .u4_norm_q8 import (
    CONTRACT_ID, NORM_ATTRS, NormApplyQ8C1, norm_q8_attrs, validate_bound_stats,
    HIDDEN_TYPE, STATS_TYPE, Q8_TYPE,
)
from .u4_ops import q8_projection_metadata, verify_q8_provenance, Q8_CONTRACT_KEYS

REGIONS = ("decode_linear_0", "decode_attention_1")
MULTIPLICITY = dict(decode_linear_0=48, decode_attention_1=16)


def _require(value, message):
    if not value:
        raise IRSchemaError("A49 " + message)


def _users(module, node_id):
    return tuple(n.id for n in module.nodes if node_id in n.inputs)


def verify_hidden_epochs(module):
    """Prove the stats parameter follows hidden through all64 sequential calls."""
    from .u4_regions import _inventory
    _, calls, _ = _inventory(module)
    for name in REGIONS:
        fn = module.function_map[name]
        _require(len(fn.outputs) == 3, "callee output ABI must remain hidden/state/stats.")
        output, stats = module.node_map[fn.outputs[0]], module.node_map[fn.outputs[2]]
        _require(stats.op == "nn.norm_stats" and stats.inputs == (output.id,)
                 and stats.attrs.get("axis") in (-1, 1)
                 and stats.attrs.get("use_mean") is False,
                 "callee returned stats must describe its exact returned hidden.")
    previous = None
    for i, call in enumerate(calls):
        fn = module.function_map[call.attrs["callee"]]
        _require(call.id == f"layer_{i}_call", "full original sequential call identity changed.")
        hidden_index = fn.parameters.index(fn.name + "_hidden")
        stats_index = fn.parameters.index(fn.name + "_hidden.norm_stats")
        hidden, stats = (module.node_map[call.inputs[j]] for j in (hidden_index, stats_index))
        if previous is None:
            _require(stats.op == "nn.norm_stats" and stats.inputs == (hidden.id,)
                     and stats.attrs.get("axis") in (-1, 1)
                     and stats.attrs.get("use_mean") is False,
                     "first layer stats must describe the exact entry hidden.")
        else:
            _require(hidden.op == stats.op == "builtin.get_item"
                     and hidden.inputs == stats.inputs == (previous.id,)
                     and hidden.attrs.get("index") == 0 and stats.attrs.get("index") == 2,
                     f"layer {i} hidden/stats must share the preceding invocation epoch.")
        previous = call
    return len(calls)


def _matched(module, fn):
    norm = module.node_map[fn + "_input_norm.vectorized.compute"]
    view = module.node_map[fn + "_input_norm"]
    q8 = module.node_map[view.id + ".u4_q8_c1"]
    hidden = module.node_map[fn + "_hidden"]
    stats = module.node_map[norm.inputs[1]]
    residual = module.node_map[fn + "_attention_residual.vectorized.compute"]
    hidden_view = module.node_map[fn + "_hidden.boundary_restore"]
    _require(norm.op == "nn.norm_apply" and norm.effect == PURE, "input NormApply changed.")
    _require(norm.inputs[0] == hidden.id, "input norm does not read original hidden formal.")
    _require(dict(get_definition(norm.op).normalize_attrs(norm.attrs)) ==
             dict(get_definition(norm.op).normalize_attrs(NORM_ATTRS)), "fixed norm arithmetic changed.")
    validate_bound_stats(hidden, stats)
    _require(stats.inputs[1] == fn + "_hidden.norm_stats", "bound stats must be the original stats formal.")
    _require(view.op == "tensors.bitcast" and view.inputs == (norm.id,)
             and view.attrs == {"dtype": "bfloat16"}, "normalized scalar alias changed.")
    _require(tensor_of(norm.type) == HIDDEN_TYPE and tensor_of(stats.type) == STATS_TYPE,
             "norm/stats type changed.")
    _require(q8.op == "u4.q8_encode" and q8.inputs == (view.id,) and q8.effect == PURE
             and q8.attrs["k"] == 5120 and q8.attrs["components"] == 1
             and q8.attrs["source_epoch"] == fn + ":invocation:" + view.id, "input C1 encoding changed.")
    _require(hidden_view.op == "tensors.bitcast" and hidden_view.inputs == (hidden.id,)
             and residual.op == "math.vectorized_binary"
             and residual.attrs.get("binary_op") == "add"
             and residual.inputs[0] == hidden_view.id, "original hidden residual path changed.")
    expected_users = ({"decode_linear_0_in_proj_qkv", "decode_linear_0_in_proj_z.u4_canonical"}
                      if fn == "decode_linear_0"
                      else {"decode_attention_1_k.packed", "decode_attention_1_v.packed"})
    _require(set(_users(module, q8.id)) == expected_users, "input C1 shared consumers changed.")
    return norm, view, q8, hidden, stats, residual


def rewrite_input_norm_q8(module):
    from .u4_regions import verify_u4_contracts, _make
    _require(module.stage == "stats_combined"
             and module.metadata.get("u4_rewrite_scope") == "full12",
             "input fusion must follow full12 at stats_combined.")
    _require(not (module.prim_functions or module.execution_functions or module.kernel_definitions),
             "fusion requires a clean logical module.")
    if module.metadata.get("u4_input_norm_q8") is not None:
        verify_u4_contracts(module)
        return module
    verify_u4_contracts(module)
    verify_hidden_epochs(module)
    # Validate both regions before returning any transformed module.
    replacements, insertions, records = {}, {}, []
    for fn in REGIONS:
        norm, view, q8, hidden, stats, residual = _matched(module, fn)
        attrs = norm_q8_attrs(source_epoch=q8.attrs["source_epoch"],
                              hidden_epoch=fn + ":invocation:" + hidden.id)
        fused_id = view.id + ".u4_norm_q8_c1"
        _require(fused_id not in module.node_map, "fused SSA already exists.")
        metadata = {"canonical_global": True, "kernel_program_kind": "ordinary",
                    "u4_input_norm_q8": CONTRACT_ID, "u4_source_ssa": view.id,
                    "u4_hidden_ssa": hidden.id, "u4_stats_ssa": stats.id,
                    "u4_original_norm": norm.id, "u4_original_q8": q8.id,
                    "u4_no_output_alias_inputs": True}
        inputs = tuple(module.node_map[x] for x in norm.inputs)
        fused = with_execution_mesh(_make(NormApplyQ8C1, fused_id, inputs, attrs, metadata),
                                    ExecutionMesh.dynamic())
        normalized = _make(get_definition("builtin.get_item"), norm.id, (fused,), {"index": 0},
                           {"canonical_global": True, "u4_input_norm_q8": CONTRACT_ID,
                            "u4_normalized_producer": fused.id, "u4_normalized_field": 0})
        encoded = _make(get_definition("builtin.get_item"), q8.id, (fused,), {"index": 1},
                        {**q8_projection_metadata(fused), "u4_source_ssa": view.id})
        _require(normalized.type == norm.type and encoded.type == q8.type, "tuple result ABI changed.")
        replacements[norm.id], replacements[q8.id] = normalized, encoded
        insertions[norm.id] = fused
        records.append(dict(function=fn, fused=fused.id, normalized=norm.id, scalar_view=view.id,
                            q8=q8.id, hidden=hidden.id, stats=stats.id,
                            scale=norm.inputs[2], bias=norm.inputs[3], residual=residual.id,
                            normalized_users=_users(module, norm.id), scalar_users=_users(module, view.id),
                            q8_users=_users(module, q8.id), source_epoch=attrs["source_epoch"],
                            hidden_epoch=attrs["hidden_epoch"]))
    nodes = []
    for node in module.nodes:
        if node.id in insertions:
            nodes.append(insertions[node.id])
        nodes.append(replacements.get(node.id, node))
    result = replace(module, nodes=tuple(nodes), metadata={
        **module.metadata, "u4_input_norm_q8": CONTRACT_ID, "u4_input_norm_q8_records": records})
    _require(result.functions == module.functions, "function/call ABI changed.")
    _require(all(result.node_map[n.id] == n for n in module.nodes if n.id not in replacements),
             "non-region source/state/output changed.")
    verify_module(result)
    verify_u4_contracts(result)
    return result


def verify_input_norm_q8(module):
    """Graph checkpoint audit. Distributed/native checks remain owner supplied."""
    _require(module.metadata.get("u4_input_norm_q8") == CONTRACT_ID, "unknown fusion contract.")
    records = module.metadata.get("u4_input_norm_q8_records", ())
    _require(len(records) == 2 and {r["function"] for r in records} == set(REGIONS),
             "must contain exactly two reused input fusions.")
    _require({n.id for n in module.nodes if n.op == NormApplyQ8C1.op_name} ==
             {r["fused"] for r in records}, "unregistered or missing fused producer.")
    verify_hidden_epochs(module)
    for r in records:
        fused, normalized, q8, scalar, hidden, stats = (
            module.node_map[r[k]] for k in ("fused", "normalized", "q8", "scalar_view", "hidden", "stats"))
        _require(fused.inputs == (hidden.id, stats.id, r["scale"], r["bias"]), "fused operands changed.")
        validate_bound_stats(hidden, stats)
        _require(stats.inputs[1] == r["function"] + "_hidden.norm_stats", "bound formal epoch changed.")
        _require(fused.attrs == NormApplyQ8C1.normalize_attrs(norm_q8_attrs(source_epoch=r["function"] + ":invocation:" + scalar.id,
                  hidden_epoch=r["function"] + ":invocation:" + hidden.id)), "fused epoch/attributes changed.")
        get_definition(fused.op).verify(fused, module)
        _require(normalized.op == "builtin.get_item" and normalized.inputs == (fused.id,)
                 and normalized.attrs == {"index": 0} and tensor_of(normalized.type) == HIDDEN_TYPE,
                 "normalized tuple field changed.")
        _require(q8.op == "builtin.get_item" and q8.inputs == (fused.id,)
                 and q8.attrs == {"index": 1} and tensor_of(q8.type) == Q8_TYPE,
                 "Q8 tuple field changed.")
        verify_q8_provenance(module, q8, fused.attrs)
        _require(scalar.op == "tensors.bitcast" and scalar.inputs == (normalized.id,)
                 and scalar.attrs == {"dtype": "bfloat16"}, "normalized scalar alias lost.")
        for key, node in (("normalized_users", normalized), ("scalar_users", scalar), ("q8_users", q8)):
            # The old C1 source read is removed; it consumes the fused register
            # result instead. Every other old edge is preserved exactly.
            wanted = tuple(n for n in r[key] if not (key == "scalar_users" and n == q8.id))
            _require(_users(module, node.id) == wanted, f"{key} changed or escaped.")
        residual = module.node_map[r["residual"]]
        hidden_view = module.node_map[residual.inputs[0]]
        _require(hidden_view.op == "tensors.bitcast" and hidden_view.inputs == (hidden.id,),
                 "residual lost the same-epoch original hidden.")
        _require(set(_users(module, fused.id)) == {normalized.id, q8.id}, "tuple escaped registered fields.")
        _require(not NormApplyQ8C1.inplace_input_parameters and not NormApplyQ8C1.inplace_output_parameters,
                 "fused results may not alias live inputs.")
    return {"fused_sites": 2, "fused_calls": 64, "standalone_q8_sites": 8,
            "standalone_q8_calls": 209, "logical_q8_calls": 273,
            "hidden_stats_epochs": 64}

"""A56 complete two-region rewrite; normal physical planning must follow."""
from dataclasses import replace

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import verify_module
from triton.flagmega.ir.execution_mesh import ExecutionMesh, with_execution_mesh
from triton.flagmega.ir.ops.core import get_definition
from .u4_ops import encoding_attrs, _common, q8_tuple_contract, q8_projection_metadata, verify_q8_provenance, Q8_CONTRACT_KEYS
from .u4_norm_q8 import NormApplyQ8C1
from .u4_norm_q8_regions import verify_hidden_epochs, _users, REGIONS
from .u4_norm_q8_c1_c2 import (
    CONTRACT_ID, NormApplyQ8C1C2, norm_q8_attrs, HIDDEN_TYPE, Q8_C1_TYPE, Q8_C2_TYPE,
)
from .u4_gdn_q8_c2 import GDNRecurrentCoreQ8C2


def require(value, message):
    if not value:
        raise IRSchemaError("A56 " + message)


def rewrite_input_norm_common_q8(module):
    from .u4_regions import _make, _inventory, verify_u4_contracts
    require(module.stage == "stats_combined" and module.metadata.get("u4_rewrite_scope") == "full12",
            "requires the complete full12 logical graph at stats_combined.")
    require(not (module.prim_functions or module.execution_functions or module.kernel_definitions),
            "requires clean logical IR before distribution and selection.")
    if module.metadata.get("u4_input_norm_q8_c1c2") is not None:
        verify_u4_contracts(module)
        return module
    verify_u4_contracts(module)
    verify_hidden_epochs(module)
    bodies, _, _ = _inventory(module)
    old_records = module.metadata.get("u4_input_norm_q8_records", ())
    require(len(old_records) == 2, "requires the existing two A49 input-norm regions.")
    replacements, insertions, records = {}, {}, []
    for old in old_records:
        r = dict(old)
        fn = r["function"]
        original = module.node_map[r["fused"]]
        require(original.op == NormApplyQ8C1.op_name, "unexpected parent NormApply producer.")
        attrs = norm_q8_attrs(source_epoch=original.attrs["source_epoch"],
                              hidden_epoch=original.attrs["hidden_epoch"])
        fused = with_execution_mesh(_make(NormApplyQ8C1C2, original.id,
            tuple(module.node_map[x] for x in original.inputs), attrs,
            {**original.metadata, "u4_input_norm_q8": CONTRACT_ID,
             "u4_no_output_alias_inputs": True, "u4_disjoint_result_fields": (0, 1, 2)}),
            ExecutionMesh.dynamic())
        replacements[original.id] = fused
        for field, node_id in [(0, r["normalized"]), (1, r["q8"])]:
            previous = module.node_map[node_id]
            metadata = ({**previous.metadata, "u4_input_norm_q8": CONTRACT_ID} if field == 0
                        else {**q8_projection_metadata(fused, field=1), "u4_source_ssa": r["scalar_view"]})
            node = _make(get_definition("builtin.get_item"), node_id, (fused,), {"index": field}, metadata)
            require(node.type == previous.type, "existing normalized/C1 field type changed.")
            replacements[node_id] = node
        c2_id = r["scalar_view"] + ".u4_q8_c2"
        c2 = _make(get_definition("builtin.get_item"), c2_id, (fused,), {"index": 2},
                   {**q8_projection_metadata(fused, field=2), "u4_source_ssa": r["scalar_view"]})
        r["q8_c2"] = c2_id
        if fn == "decode_attention_1":
            previous = module.node_map[c2_id]
            require(previous.op == "u4.q8_encode" and previous.inputs == (r["scalar_view"],)
                    and previous.attrs == _common(encoding_attrs(k=5120, components=2, source_epoch=attrs["source_epoch"])),
                    "attention C2 original source/contract changed.")
            require(set(_users(module, c2_id)) == {"decode_attention_1_q.packed"}, "attention C2 readers changed.")
            require(previous.type == c2.type, "attention C2 ABI changed.")
            replacements[c2_id] = c2
            r["q8_c2_users"] = _users(module, c2_id)
        else:
            require(fn == "decode_linear_0" and c2_id not in module.node_map, "linear C2 SSA collision.")
            candidates = [n for n in module.nodes if n.id in bodies[fn] and n.op == "nn.gdn_recurrent_core"]
            require(len(candidates) == 1, "requires the original single recurrent core site.")
            gdn = candidates[0]
            require(len(gdn.inputs) == 9 and gdn.inputs[3] == r["scalar_view"], "GDN projection source changed.")
            inputs = tuple(module.node_map[x] for x in gdn.inputs) + (c2,)
            changed = _make(GDNRecurrentCoreQ8C2, gdn.id, inputs,
                {**gdn.attrs, **encoding_attrs(k=5120, components=2, source_epoch=attrs["source_epoch"])},
                {**gdn.metadata, "u4_external_q8_c2": CONTRACT_ID, "u4_source_ssa": r["scalar_view"]})
            require(changed.type == gdn.type and changed.effect == gdn.effect, "GDN state/result semantics changed.")
            replacements[gdn.id] = changed
            insertions[r["q8"]] = c2
            r.update(gdn=gdn.id, original_gdn_inputs=gdn.inputs, original_gdn_attrs=dict(gdn.attrs),
                     q8_c2_users=(gdn.id,))
        records.append(r)
    nodes = []
    for old in module.nodes:
        nodes.append(replacements.get(old.id, old))
        if old.id in insertions:
            nodes.append(insertions[old.id])
    retired_owners = set(replacements)
    retired = {p.id for p in module.selection_points if p.owner in retired_owners}
    result = replace(module, nodes=tuple(nodes),
        selection_points=tuple(p for p in module.selection_points if p.id not in retired),
        selections=tuple(s for s in module.selections if s.point_id not in retired),
        metadata={**module.metadata, "u4_input_norm_q8": CONTRACT_ID,
                  "u4_input_norm_q8_c1c2": CONTRACT_ID, "u4_input_norm_q8_records": records,
                  "u4_input_norm_q8_c1c2_retired_selection_points": tuple(sorted(retired))})
    require(result.functions == module.functions, "caller/formal/result ABI changed.")
    require(all(result.node_map[n.id] == n for n in module.nodes if n.id not in replacements),
            "unrelated graph/state/constant nodes changed.")
    verify_module(result)
    verify_u4_contracts(result)
    return result


def verify_input_norm_common_q8(module):
    require(module.metadata.get("u4_input_norm_q8_c1c2") == CONTRACT_ID
            and module.metadata.get("u4_input_norm_q8") == CONTRACT_ID, "unknown common-C0 contract.")
    records = module.metadata.get("u4_input_norm_q8_records", ())
    require(len(records) == 2 and {r["function"] for r in records} == set(REGIONS), "two original regions required.")
    require({n.id for n in module.nodes if n.op == NormApplyQ8C1C2.op_name} == {r["fused"] for r in records},
            "unregistered or missing producer.")
    require({n.id for n in module.nodes if n.op == GDNRecurrentCoreQ8C2.op_name} ==
            {r["gdn"] for r in records if "gdn" in r}, "unregistered or missing GDN consumer.")
    verify_hidden_epochs(module)
    for r in records:
        fused = module.node_map[r["fused"]]
        require(fused.inputs == (r["hidden"], r["stats"], r["scale"], r["bias"]), "producer operands changed.")
        require(fused.attrs == NormApplyQ8C1C2.normalize_attrs(norm_q8_attrs(
            source_epoch=r["function"] + ":invocation:" + r["scalar_view"],
            hidden_epoch=r["function"] + ":invocation:" + r["hidden"])), "producer epoch/attributes changed.")
        get_definition(fused.op).verify(fused, module)
        stats = module.node_map[r["stats"]]
        require(stats.inputs[1] == r["function"] + "_hidden.norm_stats", "bound stats formal changed.")
        for field, key, typ in [(0, "normalized", HIDDEN_TYPE), (1, "q8", Q8_C1_TYPE), (2, "q8_c2", Q8_C2_TYPE)]:
            node = module.node_map[r[key]]
            require(node.op == "builtin.get_item" and node.inputs == (fused.id,)
                    and node.attrs == {"index": field} and node.type == typ, "tuple field changed.")
            if field:
                verify_q8_provenance(module, node, q8_tuple_contract(fused, field))
        require(set(_users(module, fused.id)) == {r["normalized"], r["q8"], r["q8_c2"]}, "tuple escaped fields.")
        scalar = module.node_map[r["scalar_view"]]
        require(scalar.op == "tensors.bitcast" and scalar.inputs == (r["normalized"],)
                and scalar.attrs == {"dtype": "bfloat16"}, "normalized scalar alias changed.")
        for key, nodekey in [("normalized_users", "normalized"), ("q8_users", "q8"), ("q8_c2_users", "q8_c2")]:
            require(_users(module, r[nodekey]) == tuple(r[key]), key + " changed.")
        expected_scalar = tuple(n for n in r["scalar_users"] if n not in (r["q8"], r["q8_c2"]))
        require(_users(module, scalar.id) == expected_scalar, "normalized readers changed.")
        residual = module.node_map[r["residual"]]
        hidden_view = module.node_map[residual.inputs[0]]
        require(hidden_view.op == "tensors.bitcast" and hidden_view.inputs == (r["hidden"],), "residual epoch changed.")
        if "gdn" in r:
            gdn = module.node_map[r["gdn"]]
            require(gdn.inputs == tuple(r["original_gdn_inputs"]) + (r["q8_c2"],), "GDN original operands changed.")
            require({k: v for k, v in gdn.attrs.items() if k not in Q8_CONTRACT_KEYS} == r["original_gdn_attrs"],
                    "GDN original arithmetic attributes changed.")
            get_definition(gdn.op).verify(gdn, module)
    return dict(fused_sites=2, fused_calls=64, standalone_q8_sites=7, standalone_q8_calls=193,
                logical_q8_calls=321, hidden_stats_epochs=64, external_GDN_C2_calls=48,
                removed_attention_encoder_calls=16, physical_planning_verified=False)


def register_input_norm_common_q8_stage():
    from triton.flagmega.stages import Stage, register_stage, get_stage
    from triton.flagmega.errors import StageError
    from .u4_regions import rewrite_u4_regions
    name = "form-u4-full12-input-norm-common-c0-c1-c2"
    try:
        return get_stage(name)
    except StageError:
        stage = Stage(name, "stats_combined", "stats_combined", lambda module, target:
                      rewrite_input_norm_common_q8(rewrite_u4_regions(module, scope="full12", input_norm_q8=True)))
        register_stage(stage)
        return stage

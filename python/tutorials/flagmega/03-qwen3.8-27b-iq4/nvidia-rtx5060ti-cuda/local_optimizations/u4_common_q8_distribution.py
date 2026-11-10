# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A56 ownership requests and checks of authoritative normal compiler plans.

This module never allocates, injects pointers, substitutes a workspace for an
operand, or claims that logical metadata establishes physical publication.
The Kernel owner supplies the new implementation, encoder and templates.
"""
from dataclasses import replace
from itertools import combinations

from triton.flagmega.passes.auto_distributed.providers import GdnRecurrentCandidateProvider
from .u4_q8_abi import require
from .u4_norm_q8_c1_c2 import CONTRACT_ID
from .u4_norm_q8_distribution import _canonical, _disjoint, audit_norm_q8_callers

FUNCTIONS = ("decode_linear_0", "decode_attention_1")
RESULT_BYTES = (10240, 5760, 11520)
GDN_OP = "u4.gdn_recurrent_core_q8_c2"


def validate_fused_node(node, module=None):
    from triton.flagmega.ir import TupleType
    from triton.flagmega.ir.distributed_inference import tensor_of
    from .u4_norm_q8_c1_c2 import HIDDEN_TYPE, NormApplyQ8C1C2
    from .u4_norm_q8_c1_c2_vector import field_q8_contract, validate_norm_types
    require(node.op == NormApplyQ8C1C2.op_name and len(node.inputs) == 4,
            "A56 requires its actual four-operand operation")
    require(isinstance(node.type, TupleType) and len(node.type.fields) == 3
            and tensor_of(node.type.fields[0]) == HIDDEN_TYPE,
            "A56 requires normalized BF16x8 and two independent Q8 results")
    NormApplyQ8C1C2.normalize_attrs(node.attrs)
    for field in (1, 2):
        field_q8_contract(node, field)
    require(node.metadata.get("u4_no_output_alias_inputs") is True,
            "A56 results cannot alias any input")
    if module is not None:
        validate_norm_types(tuple(module.node_map[x].type for x in node.inputs), node.type)


def prepare_common_q8_ownership(module):
    """Attach canonical storage requirements before normal distribution."""
    from triton.flagmega.ir.execution_mesh import ExecutionMesh
    from .u4_norm_q8_c1_c2_regions import verify_input_norm_common_q8
    from .u4_norm_q8_regions import verify_hidden_epochs
    require(module.stage == "stats_combined"
            and module.metadata.get("u4_input_norm_q8_c1c2") == CONTRACT_ID,
            "A56 preparation requires its open logical stage")
    verify_input_norm_common_q8(module)
    verify_hidden_epochs(module)
    records = module.metadata["u4_input_norm_q8_records"]
    require(len(records) == 2 and {r["function"] for r in records} == set(FUNCTIONS),
            "both reusable layer definitions must participate")
    nodes = dict(module.node_map)
    for r in records:
        fused, hidden, stats, residual = (nodes[r[k]] for k in ("fused", "hidden", "stats", "residual"))
        validate_fused_node(fused, module)
        hidden_view = nodes[residual.inputs[0]]
        fn = module.function_map[r["function"]]
        require(hidden_view.op == "tensors.bitcast" and hidden_view.inputs == (hidden.id,),
                "residual must retain the original hidden scalar view")
        require(hidden.id in fn.parameters and stats.inputs == (hidden.id, fn.name + "_hidden.norm_stats"),
                "bound hidden and statistics must be the same invocation formals")
        for name in (hidden.id, hidden_view.id, r["normalized"], r["q8"], r["q8_c2"], r["scalar_view"], fused.id):
            n = nodes[name]
            nodes[name] = replace(n, metadata={**n.metadata, "canonical_global": True,
                "u4_norm_q8_ownership": CONTRACT_ID})
        nodes[fused.id] = replace(nodes[fused.id], metadata={**nodes[fused.id].metadata,
            "u4_retain_all_result_stores": True, "u4_result_nbytes": RESULT_BYTES,
            "u4_no_result_aliases": True})
        nodes[hidden.id] = replace(nodes[hidden.id], metadata={**nodes[hidden.id].metadata,
            "u4_hidden_epoch": r["hidden_epoch"], "u4_hidden_last_reader": residual.id})
        nodes[hidden_view.id] = replace(nodes[hidden_view.id], metadata={**nodes[hidden_view.id].metadata,
            "execution_mesh": ExecutionMesh.dynamic().to_data()})
        nodes[residual.id] = replace(residual, metadata={**residual.metadata,
            "u4_cyclic32_reader": True, "u4_canonical_hidden_source": hidden.id,
            "u4_canonical_hidden_view": hidden_view.id, "u4_hidden_epoch": r["hidden_epoch"]})
        nodes[r["q8_c2"]] = replace(nodes[r["q8_c2"]], metadata={**nodes[r["q8_c2"]].metadata,
            "u4_complete_value_readers": tuple(r["q8_c2_users"]),
            "u4_publication": "same-stream-kernel-completion", "u4_read_only_consumers": True})
        if "gdn" in r:
            gdn = nodes[r["gdn"]]
            require(gdn.op == GDN_OP and len(gdn.inputs) == 10 and gdn.inputs[9] == r["q8_c2"]
                    and gdn.inputs[3] == r["scalar_view"]
                    and gdn.inputs[:9] == tuple(r["original_gdn_inputs"]),
                    "external C2 must be the real tenth operand with unchanged original nine")
            nodes[gdn.id] = replace(gdn, metadata={**gdn.metadata,
                "u4_external_q8_input": 9, "u4_external_q8_source": r["q8_c2"],
                "u4_external_q8_producer": fused.id, "u4_external_q8_epoch": r["source_epoch"]})
        hindex = fn.parameters.index(hidden.id)
        for call in module.nodes:
            if call.op == "builtin.call" and call.attrs.get("callee") == fn.name:
                actual = nodes[call.inputs[hindex]]
                nodes[actual.id] = replace(actual, metadata={**actual.metadata,
                    "canonical_global": True, "u4_norm_q8_ownership": CONTRACT_ID})
    return replace(module, nodes=tuple(nodes[n.id] for n in module.nodes))


def audit_common_q8_callers(module):
    """Check all64 logical SSA epochs, not actual BufferPlan call resolution."""
    from .u4_norm_q8_c1_c2_regions import verify_input_norm_common_q8
    verify_input_norm_common_q8(module)
    rows = audit_norm_q8_callers(module)
    records = {r["function"]: r for r in module.metadata["u4_input_norm_q8_records"]}
    for row in rows:
        r = records[row["function"]]
        row.update(producer=r["fused"], source_epoch=r["source_epoch"],
                   q8_c2=r["q8_c2"], q8_c2_readers=list(r["q8_c2_users"]),
                   actual_call_epoch=row["call"] + ":" + r["source_epoch"],
                   physical_binding_verified=False)
    return rows


class GDNCommonQ8DistributionProvider(GdnRecurrentCandidateProvider):
    """New operation only: inherit and constrain the normal fullhead relation."""
    op_names = frozenset({GDN_OP})
    allows_partial_inputs = False

    def _enumerate_candidates(self, context):
        from triton.flagmega.ir import Placement, TupleType
        from triton.flagmega.ir.distributed_type import SBPBroadCast
        from triton.flagmega.ir.execution_mesh import effective_execution_mesh
        from .u4_transitions import gdn_z_reader_type
        from .u4_gdn_q8_c2 import GDNRecurrentCoreQ8C2
        from .u4_norm_q8_c1_c2_vector import validate_gdn_q8_type
        node = context.source_call
        mesh = effective_execution_mesh(context.module, node)
        require(node.op == GDN_OP and len(node.inputs) == 10
                and context.placement == Placement((4, 12), "yx", "bb")
                and mesh is not None and mesh.kind.value == "fixed" and mesh.shape == (4, 12)
                and node.metadata.get("kernel_num_warps") == 8
                and node.metadata.get("u4_gdn_fullhead48") is True,
                "external C2 GDN requires the exact48 CTA/8-warp owner request")
        GDNRecurrentCoreQ8C2.verify(node, context.module)
        wanted = gdn_z_reader_type(node.type.fields[0], fullhead48=True)
        values = []
        for candidate in super()._enumerate_candidates(context):
            if candidate.reason != "gdn-recurrent-head-sbp":
                continue
            # The generic provider leaves contiguous grain implicit. Fix the
            # proven6144/48=128 equal split on both Z and output explicitly;
            # normal forward inference must accept this exact candidate.
            candidate = replace(candidate,
                input_types=(*candidate.input_types[:2], wanted, *candidate.input_types[3:]),
                return_type=TupleType((wanted, candidate.return_type.fields[1])))
            if any(getattr(t, "partial", None) is not None or getattr(t, "exclusive", None) is not None
                   for t in candidate.input_types):
                continue
            q8, projection = candidate.input_types[9], candidate.input_types[3]
            validate_gdn_q8_type(q8)
            require(q8.placement == projection.placement == context.placement
                    and all(isinstance(p, SBPBroadCast) for t in (q8, projection) for p in t.axis_policies),
                    "input9 C2 and input3 projection must expose the same complete broadcast placement")
            values.append(replace(candidate, reason="u4-gdn-fullhead48-external-c2-complete-read"))
        require(len(values) == 1, "external C2 GDN must have exactly one complete fullhead48 candidate")
        return tuple(values)


def _overlap(a, b):
    """Physical arena overlap; two named parameters defer to caller binding."""
    ar, br = a.mem_span.buffer, b.mem_span.buffer
    same_root = a.physical_id == b.physical_id
    same_arena = (ar.memory_space == br.memory_space and ar.function == br.function
                  and a.storage != "parameter" and b.storage != "parameter")
    return (same_root or same_arena) and max(a.mem_span.offset, b.mem_span.offset) < min(
        a.mem_span.offset + a.mem_span.nbytes, b.mem_span.offset + b.mem_span.nbytes)


def _require_live(root, start, end):
    physical = root.mem_span.buffer
    require(type(physical.live_start) is int and type(physical.live_end) is int
            and physical.live_start <= start <= end <= physical.live_end,
            "C2 allocation must cover producer through final consumer inclusive")


def _q8_alignment(root):
    require(root.mem_span.buffer.alignment >= 16 and root.mem_span.offset % 16 == 0,
            "Q8 roots must retain16-byte alignment after caller resolution")


def _alias_reaches(descriptor, root, plan):
    seen = set()
    while descriptor.id != root.id:
        if descriptor.id in seen or descriptor.alias_of not in plan.buffer_map:
            return False
        seen.add(descriptor.id)
        descriptor = plan.buffer_map[descriptor.alias_of]
    return True


def _live_intersections(root, plan, function, start, end):
    """Check every simultaneous pool descriptor, not just named QKV/state."""
    _require_live(root, start, end)
    checked = 0
    for other in plan.buffers:
        if other.function != function or other.id == root.id:
            continue
        if _alias_reaches(other, root, plan):
            require(other.mem_span == root.mem_span, "C2 view must retain complete canonical bytes")
            continue
        physical = other.mem_span.buffer
        lo = physical.live_start
        hi = physical.live_end
        # Parameters/state/constants with no local ordinal live for this call.
        concurrent = lo is None or hi is None or max(start, lo) <= min(end, hi)
        if concurrent:
            checked += 1
            require(not _overlap(root, other), "live C2 overlaps independent allocation: " + other.id)
    return checked


def validate_common_q8_physical_contract(module, plan, *, records=None):
    """Audit actual authoritative plan after normal Bufferize.

    Missing renamed bindings, erased result0, synthetic input9/workspaces or
    stale lifetimes fail closed. Records identify semantic values, while core
    function_nodes supplies the planner's actual ordinal space.
    """
    from triton.flagmega.ir import verify_buffer_plan, Placement
    from triton.flagmega.ir.bufferization import BufferPlan
    from triton.flagmega.passes.functions.graph import function_nodes
    from .u4_execution_plan import cyclic32_suffix_type
    require(isinstance(plan, BufferPlan) and verify_buffer_plan(module) == plan,
            "A56 requires the authoritative regenerated BufferPlan")
    records = records or module.metadata.get("u4_input_norm_q8_records", ())
    require(len(records) == 2 and {r["function"] for r in records} == set(FUNCTIONS),
            "physical audit needs both semantic region records")
    functions, regions, calls = {}, [], []
    for r in records:
        fn = plan.function_map[r["function"]]
        bindings = dict((*fn.parameters, *fn.values, *fn.outputs))
        def one(name):
            require(name in bindings and len(bindings[name]) == 1, "missing one-leaf binding: " + name)
            return plan.buffer_map[bindings[name][0]]
        require(r["fused"] in bindings and len(bindings[r["fused"]]) == 3,
                "producer must retain three actual output roots, including unused attention field0")
        outputs = tuple(plan.buffer_map[x] for x in bindings[r["fused"]])
        hidden, stats = one(r["hidden"]), one(r["function"] + "_hidden.norm_stats")
        _canonical(hidden, plan, 10240, "original hidden")
        _canonical(stats, plan, 4, "stats")
        for field, (root, size) in enumerate(zip(outputs, RESULT_BYTES)):
            _canonical(root, plan, size, "result" + str(field))
            if field:
                _q8_alignment(root)
            _disjoint(root, hidden)
        for left, right in combinations(outputs, 2):
            _disjoint(left, right)
        for field, key in enumerate(("normalized", "q8", "q8_c2")):
            if field == 0 and r[key] not in bindings:
                require(r["function"] == FUNCTIONS[1], "only unused attention projection can be trimmed")
            else:
                require(one(r[key]).mem_span == outputs[field].mem_span,
                        "tuple projection disagrees with actual output field")
        residual = module.node_map[r["residual"]]
        left, right = (one(n) for n in residual.inputs)
        for d in (left, right):
            _canonical(d, plan, 10240, "residual source")
            require(d.distributed_type == cyclic32_suffix_type(d.distributed_type, Placement((4, 24), "yx", "bb")),
                    "residual original cyclic32 owner map changed")
        require(left.mem_span == hidden.mem_span, "residual lost original hidden root")
        if hidden.mem_span.buffer.live_end is not None:
            require(left.live_end is not None and hidden.mem_span.buffer.live_end >= left.live_end,
                    "original hidden allocation dies before residual completion")
        require(not any(n.metadata.get("boundary_layout") == "raw_parameter_restore"
                        and n.inputs == (r["hidden"],) and n.op not in ("distributed.sharded_view", "tir.buffer_view")
                        for n in module.nodes), "forbidden hidden restore materialized")
        ordered = function_nodes(module, module.function_map[r["function"]])
        ordinal = {n.id: i for i, n in enumerate(ordered)}
        require(r["fused"] in ordinal and all(x in ordinal for x in r["q8_c2_users"]),
                "actual producer/consumer ordinal identities must be preserved")
        start = ordinal[r["fused"]]
        end = max(ordinal[n] for n in r["q8_c2_users"])
        require(start < end, "C2 consumers must follow complete Norm producer")
        c2 = outputs[2]
        readers = []
        for name in r["q8_c2_users"]:
            reader = module.node_map[name]
            matching = [one(x) for x in reader.inputs if x in bindings and len(bindings[x]) == 1
                        and one(x).mem_span == c2.mem_span]
            require(len(matching) == 1 and _alias_reaches(matching[0], c2, plan),
                    "actual C2 consumer needs one normal complete operand, never a copied/workspace pointer")
            if name == r.get("gdn"):
                require(len(reader.inputs) == 10 and one(reader.inputs[9]).mem_span == c2.mem_span,
                        "new GDN input9 must bind the complete C2 root")
                require(one(reader.inputs[3]).mem_span == outputs[0].mem_span,
                        "GDN input3 must retain this Norm's normalized BF16 root")
                record = plan.kernel_call_map.get(name)
                require(record is not None and all("head_projection_q8_global" not in formal for formal, _ in record.workspaces),
                        "new GDN cannot retain its old writable Q8 workspace")
            readers.append(dict(node=name, ordinal=ordinal[name], binding=matching[0].id))
        checked = _live_intersections(c2, plan, r["function"], start, end)
        functions[r["function"]] = (hidden, stats, outputs)
        regions.append(dict(function=r["function"], outputs=[d.id for d in outputs],
                            c2_producer_ordinal=start, c2_final_reader_ordinal=end,
                            readers=readers, concurrent_descriptors_checked=checked))
    layer_calls = {c.call: c for f in plan.functions for c in f.calls if c.callee in FUNCTIONS}
    require(set(layer_calls) == {f"layer_{i}_call" for i in range(64)},
            "actual BufferPlan must retain exactly all64 layer calls")
    for i in range(64):
        call = layer_calls[f"layer_{i}_call"]
        require(call.callee == FUNCTIONS[1 if i % 4 == 3 else 0], "layer call order/callee changed")
        hidden, stats, outputs = functions[call.callee]
        actual = plan.resolve_call_buffer(call.call, hidden.id)
        bound_stats = plan.resolve_call_buffer(call.call, stats.id)
        roots = tuple(plan.resolve_call_buffer(call.call, d.id) for d in outputs)
        _canonical(actual, plan, 10240, "caller hidden", bound_call=True)
        _canonical(bound_stats, plan, 4, "caller stats", bound_call=True)
        for field, (d, size) in enumerate(zip(roots, RESULT_BYTES)):
            _canonical(d, plan, size, "caller output", bound_call=True)
            if field:
                _q8_alignment(d)
            require(not _overlap(actual, d), "caller hidden and output overlap")
        for a, b in combinations(roots, 2):
            require(not _overlap(a, b), "caller normalized/C1/C2 spans overlap")
        # All callee formals are resolved, covering state, convolution and any
        # caller-backed pool that has a distinct local parameter identity.
        for _, ids in plan.function_map[call.callee].parameters:
            for ident in ids:
                other = plan.resolve_call_buffer(call.call, ident)
                require(not _overlap(roots[2], other), "caller C2 overlaps a live input/state formal")
        calls.append(dict(call=call.call, function=call.callee, hidden_root=actual.physical_id,
                          stats_root=bound_stats.physical_id,
                          output_spans=[d.mem_span.to_data() for d in roots]))
    return dict(schema="flagmega.common-q8-physical-audit/v1", regions=regions, calls=calls,
                native_access_verified=False, physical_publication_verified=False, device_verified=False)

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A49 canonical formal, tuple ownership and completion-lifetime contract.

No allocator, synthetic pointer or post-codegen edit lives here. Requests are
consumed by normal distribution/bufferization; the post-plan audit rejects an
unrealized request. Device accesses and same-stream launch completion remain
Kernel/Verifier obligations.
"""
from dataclasses import replace

from .u4_q8_abi import require

CONTRACT_ID = "a49-input-norm-q8-c1/v1"
FUNCTIONS = ("decode_linear_0", "decode_attention_1")


def validate_fused_node(node):
    from triton.flagmega.ir import TupleType
    from triton.flagmega.ir.distributed_inference import tensor_of
    from .u4_norm_q8 import HIDDEN_TYPE, NORM_ATTRS
    from .u4_q8_abi import Q8Layout
    require(node.op == "u4.norm_apply_q8_c1" and len(node.inputs) == 4,
            "expected the registered four-operand fused operation")
    require(isinstance(node.type, TupleType) and len(node.type.fields) == 2,
            "fused output must retain two independently owned tuple fields")
    require(tensor_of(node.type.fields[0]) == HIDDEN_TYPE,
            "normalized output must retain BF16<8>[1,640]")
    q8 = tensor_of(node.type.fields[1])
    require(q8.dtype.value == "uint8" and tuple(d.fixed_value for d in q8.shape) == Q8Layout(5120, 1).shape,
            "fused Q8 output schema differs")
    require((node.attrs.get("k"), node.attrs.get("components")) == (5120, 1)
            and all(node.attrs.get(k) == v for k, v in NORM_ATTRS.items()),
            "fused norm geometry or arithmetic attributes differ")
    require(node.metadata.get("u4_no_output_alias_inputs") is True,
            "fused result must forbid input aliases")


def prepare_norm_q8_ownership(module):
    """Propagate canonical requirements to the true SSA uses before D0.

    The hidden scalar bitcast is a storage view, not a new value. Keeping its
    residual use retains the original hidden's liveness through that consumer.
    No boundary-layout override is used: normal planning must prove its ABI.
    """
    from .u4_norm_q8_regions import verify_input_norm_q8, verify_hidden_epochs
    from triton.flagmega.ir.execution_mesh import ExecutionMesh
    require(module.stage == "stats_combined" and module.metadata.get("u4_input_norm_q8") == CONTRACT_ID,
            "canonical norm/Q8 preparation requires the original open fused stage")
    verify_input_norm_q8(module)
    verify_hidden_epochs(module)
    records = module.metadata["u4_input_norm_q8_records"]
    require(len(records) == 2 and {r["function"] for r in records} == set(FUNCTIONS),
            "all two reusable layer definitions must participate")
    nodes = dict(module.node_map)
    for r in records:
        fused, hidden, stats, residual = (nodes[r[k]] for k in ("fused", "hidden", "stats", "residual"))
        validate_fused_node(fused)
        hidden_view = nodes[residual.inputs[0]]
        require(hidden_view.op == "tensors.bitcast" and hidden_view.inputs == (hidden.id,),
                "residual must still read the exact original hidden through its scalar view")
        fn = module.function_map[r["function"]]
        require(hidden.id in fn.parameters and stats.inputs == (hidden.id, fn.name + "_hidden.norm_stats"),
                "bound stats and hidden must be the same invocation's formals")
        for name in (hidden.id, hidden_view.id, r["normalized"], r["q8"], r["scalar_view"], fused.id):
            n = nodes[name]
            nodes[name] = replace(n, metadata={**n.metadata, "canonical_global": True,
                "u4_norm_q8_ownership": CONTRACT_ID})
        nodes[hidden.id] = replace(nodes[hidden.id], metadata={**nodes[hidden.id].metadata,
            "u4_hidden_epoch": r["hidden_epoch"], "u4_hidden_last_reader": residual.id})
        # The retained scalar bitcast is a canonical storage view. Letting it
        # inherit fixed96 broadcast would request a raw full-vector restore at
        # the formal boundary, before the residual asks for cyclic32 owners.
        # Its dynamic unit placement agrees with the fused canonical source;
        # the residual's ordinary ShardedView changes owners without a copy.
        nodes[hidden_view.id] = replace(nodes[hidden_view.id], metadata={**nodes[hidden_view.id].metadata,
            "execution_mesh": ExecutionMesh.dynamic().to_data()})
        nodes[residual.id] = replace(residual, metadata={**residual.metadata,
            "u4_cyclic32_reader": True, "u4_canonical_hidden_source": hidden.id,
            "u4_canonical_hidden_view": hidden_view.id,
            "u4_hidden_epoch": r["hidden_epoch"]})
        # Each actual is an existing producer/projection: preserving its complete
        # coordinates lets canonical_call_storage_values solve all call roots.
        hindex = fn.parameters.index(hidden.id)
        for call in module.nodes:
            if call.op == "builtin.call" and call.attrs.get("callee") == fn.name:
                actual = nodes[call.inputs[hindex]]
                nodes[actual.id] = replace(actual, metadata={**actual.metadata,
                    "canonical_global": True, "u4_norm_q8_ownership": CONTRACT_ID})
    return replace(module, nodes=tuple(nodes[n.id] for n in module.nodes))


def audit_norm_q8_callers(module):
    """CPU pre-distribution census bound to real call/formal SSA identities."""
    from .u4_norm_q8_regions import verify_hidden_epochs
    verify_hidden_epochs(module)
    rows = []
    for i in range(64):
        call = module.node_map[f"layer_{i}_call"]
        fn = module.function_map[call.attrs["callee"]]
        hidden = fn.name + "_hidden"
        stats = hidden + ".norm_stats"
        actual = call.inputs[fn.parameters.index(hidden)]
        require(module.node_map[hidden].metadata.get("canonical_global") is True
                and module.node_map[actual].metadata.get("canonical_global") is True,
                "every caller/formal hidden must request canonical coordinates")
        rows.append(dict(layer=i, call=call.id, function=fn.name, hidden_formal=hidden,
                         hidden_actual=actual, stats_formal=stats,
                         stats_actual=call.inputs[fn.parameters.index(stats)],
                         epoch=f"{call.id}:invocation"))
    return rows


def _canonical(descriptor, plan, nbytes, label, *, bound_call=False):
    """Actual descriptors, including canonical entry/callee parameter roots."""
    from triton.flagmega.ir.bufferization import BufferDescriptor
    require(isinstance(descriptor, BufferDescriptor), label + " requires a core descriptor")
    require(descriptor.mem_span.nbytes == nbytes and (bound_call or descriptor.mem_span.byte_offset == 0),
            label + " must expose the complete canonical zero-offset span")
    require(descriptor.distributed_type is None or
            descriptor.distributed_storage_kind.value == "canonical_global",
            label + " cannot expose a compact/private distributed span")
    space = plan.memory_space_map[descriptor.mem_span.buffer.memory_space]
    require(space.sharing_scope.value == "chip", label + " cannot use CTA-private storage")
    require(not descriptor.offset_bindings, label + " cannot retain an unbound dynamic offset")


def _disjoint(a, b, *, bound_call=False):
    require(bound_call or a.physical_id != b.physical_id, "live normalized/hidden/Q8 roots must be distinct")
    ar, br = a.mem_span.buffer, b.mem_span.buffer
    # Distinct parameter roots denote caller-bound allocations; the concrete
    # all64 call resolution below checks the eventual backing arena as well.
    if ar.memory_space == br.memory_space and ar.function == br.function and a.storage != "parameter" and b.storage != "parameter":
        require(a.mem_span.offset + a.mem_span.nbytes <= b.mem_span.offset or
                b.mem_span.offset + b.mem_span.nbytes <= a.mem_span.offset,
                "independent live tuple/hidden allocations overlap in one arena")


def validate_norm_q8_physical_contract(module, plan, *, records=None):
    """Reject stale/private/aliased normal plans; does not qualify native code.

    Run after normal Bufferize with its actual module and BufferPlan. Passing
    original graph records is allowed when TIR lowered semantic calls; the
    binding names and call map are still required to exist in that plan.
    """
    from triton.flagmega.ir.bufferization import BufferPlan
    from triton.flagmega.ir import verify_buffer_plan
    from .u4_execution_plan import cyclic32_suffix_type
    from triton.flagmega import ir as fm
    require(isinstance(plan, BufferPlan), "expected the actual regenerated BufferPlan")
    require(verify_buffer_plan(module) == plan, "plan does not match authoritative module buffer metadata")
    records = records or module.metadata.get("u4_input_norm_q8_records", ())
    require(len(records) == 2 and {r["function"] for r in records} == set(FUNCTIONS),
            "physical audit requires both original semantic region records")
    by_function, result = {}, []
    for r in records:
        function = plan.function_map[r["function"]]
        bindings = dict((*function.parameters, *function.values, *function.outputs))
        def one(name):
            require(name in bindings and len(bindings[name]) == 1, "missing one-leaf binding: " + name)
            return plan.buffer_map[bindings[name][0]]
        hidden, norm, q8 = (one(r[k]) for k in ("hidden", "normalized", "q8"))
        stats = one(r["function"] + "_hidden.norm_stats")
        _canonical(stats, plan, 4, "stats formal")
        for d, size, label in ((hidden, 10240, "hidden"), (norm, 10240, "normalized"), (q8, 5760, "Q8")):
            _canonical(d, plan, size, label)
        require(q8.mem_span.buffer.alignment >= 16 and q8.mem_span.offset % 16 == 0,
                "Q8 root lost16-byte alignment")
        _disjoint(hidden, norm)
        _disjoint(hidden, q8)
        _disjoint(norm, q8)
        require(r["fused"] in bindings and tuple(bindings[r["fused"]]) == (norm.id, q8.id),
                "tuple field bindings/projection order differ from native outputs")
        reader = module.node_map[r["residual"]]
        require(len(reader.inputs) == 2, "residual reader must retain two sources")
        left, right = (one(n) for n in reader.inputs)
        for d in (left, right):
            _canonical(d, plan, 10240, "residual source")
            require(d.distributed_type == cyclic32_suffix_type(d.distributed_type,
                    fm.Placement((4, 24), "yx", "bb")),
                    "residual descriptor lost exact cyclic32 owner map")
        require(left.mem_span == hidden.mem_span,
                "residual original-hidden view must alias the same canonical span")
        # Completion ordinals come from normal planner; a parameter root has a
        # call-bounded lifetime rather than a temporary pool lifetime.
        if hidden.mem_span.buffer.live_end is not None:
            require(left.live_end is not None and hidden.mem_span.buffer.live_end >= left.live_end,
                    "canonical hidden allocation dies before its residual reader")
        require(not any(n.metadata.get("boundary_layout") == "raw_parameter_restore"
                        and n.inputs == (r["hidden"],) and n.op not in
                        ("distributed.sharded_view", "tir.buffer_view") for n in module.nodes),
                "normal compiler materialized the forbidden full hidden restore")
        by_function[r["function"]] = (hidden, norm, q8, stats)
    calls = {c.call: c for f in plan.functions for c in f.calls if c.callee in FUNCTIONS}
    require(set(calls) == {f"layer_{i}_call" for i in range(64)},
            "actual BufferPlan must preserve exactly all64 layer calls")
    for i in range(64):
        call = calls[f"layer_{i}_call"]
        require(call.callee == (FUNCTIONS[1] if i % 4 == 3 else FUNCTIONS[0]),
                "call order/callee differs from full64 model")
        hidden, norm, q8, stats = by_function[call.callee]
        actual, normalized, encoded = (plan.resolve_call_buffer(call.call, d.id) for d in (hidden, norm, q8))
        bound_stats = plan.resolve_call_buffer(call.call, stats.id)
        _canonical(actual, plan, 10240, "caller hidden", bound_call=True)
        _canonical(bound_stats, plan, 4, "caller stats", bound_call=True)
        _disjoint(actual, normalized, bound_call=True)
        _disjoint(actual, encoded, bound_call=True)
        _disjoint(normalized, encoded, bound_call=True)
        result.append(dict(call=call.call, function=call.callee, hidden_root=actual.physical_id,
                           stats_root=bound_stats.physical_id, normalized_root=normalized.physical_id,
                           q8_root=encoded.physical_id, hidden_span=actual.mem_span.to_data(),
                           normalized_span=normalized.mem_span.to_data(), q8_span=encoded.mem_span.to_data()))
    return dict(schema="flagmega.norm-q8-physical-audit/v1", calls=result,
                native_access_verified=False, device_verified=False)

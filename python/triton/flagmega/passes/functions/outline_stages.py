# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Outline closed straight-line stages without duplicating reusable bodies.

This is a graph transformation before physical planning. Explicit stages
own stateful work; other stages contain pure tensor work or structural
tuple projections. Stateful stages preserve the original mutation order.
The new calls expose every inter-stage value to ordinary ABI/storage analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import keyword
from collections.abc import Sequence

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import Function, IRModule, Node, PointerType, RefType, TensorType, TupleType, logical_type
from triton.flagmega.ir.model import PURE
from .graph import callee_first_functions, function_nodes


@dataclass(frozen=True)
class FunctionStage:
    name: str
    nodes: tuple[str, ...]


_DERIVED_LAYOUT_ATTRS = {"function_boundary_layout", "post_auto_distributed_boundary_layout"}
_FUNCTION_ATTRS = {"calling_convention", "noinline", "reusable"} | _DERIVED_LAYOUT_ATTRS
_CONSTANT_ROOTS = {"builtin.weight", "builtin.const_asset", "builtin.scalar_const", "builtin.splat_const", "builtin.none"}
_STRUCTURAL = {"builtin.tuple", "builtin.get_item"}


def _mutable_reference(kind):
    kind = logical_type(kind)
    if isinstance(kind, RefType):
        return kind.name != "parameter_bank"
    if isinstance(kind, TupleType):
        return any(_mutable_reference(field) for field in kind.fields)
    if isinstance(kind, TensorType):
        dtype = getattr(kind.dtype, "elem_type", kind.dtype)
        return isinstance(dtype, PointerType)
    return False


def outline_function_stages(module: IRModule, name: str, stages: Sequence[FunctionStage], *, effect_stage: int = 0,
                            effect_stages: tuple[int, ...] | None = None) -> IRModule:
    """Replace every direct call with calls to one shared set of stage functions.

    ``nodes`` explicitly partitions the callee's complete owned body, including
    structural nodes. Globally shared literal/readonly constant roots stay
    global. A stage may consume only original parameters or earlier stages.
    Positional layout proofs are invalidated on the changed signatures;
    numerical/distributed value types are retained verbatim. Ordinary later
    compilation must rebuild physical ABI, allocation and synchronization.
    ``effect_stages`` explicitly permits several sequential stateful stages;
    their calls keep the caller's conservative effect contract. Each original
    operation remains in exactly one body, in its original mutation order.
    """
    from triton.flagmega.ir import verify_module

    def fail(message):
        raise IRVerificationError(f"Function stage outlining: {message}", stage=module.stage)

    verify_module(module)
    if module.execution_functions or module.metadata.get("buffer_plan") is not None or module.metadata.get("codegen_package_plan") is not None:
        fail("requires a pre-allocation graph checkpoint, not a physical plan.")
    function = module.function_map.get(name)
    if function is None or name == module.entry or function.attrs.get("region_kind"):
        fail("requires an existing non-entry graph function, not a structured region.")
    if set(function.attrs) - _FUNCTION_ATTRS:
        fail("unknown semantic function attributes require an explicit stage mapping.")
    callee_first_functions(module)  # Reject recursion before changing closures.
    stages = tuple(stages)
    if type(effect_stage) is not int or not 0 <= effect_stage < len(stages):
        fail("effect stage must be an index into the stage sequence.")
    if effect_stages is not None and (
            effect_stage != 0 or not isinstance(effect_stages, tuple) or not effect_stages
            or any(type(index) is not int or not 0 <= index < len(stages) for index in effect_stages)
            or tuple(sorted(set(effect_stages))) != effect_stages):
        fail("effect stages must be unique ascending stage indices, without effect_stage.")
    effect_owners = set(effect_stages if effect_stages is not None else (effect_stage,))
    if not stages or any(not isinstance(s, FunctionStage) or not isinstance(s.name, str)
                         or not s.name.isidentifier() or keyword.iskeyword(s.name) or not s.nodes
                         or any(not isinstance(value, str) for value in s.nodes) for s in stages):
        fail("stages need non-empty bodies and identifier names.")
    if len({s.name for s in stages}) != len(stages):
        fail("stage names must be unique.")
    nodes = module.node_map
    body = function_nodes(module, function)
    parameters = set(function.parameters)
    foreign = {n.id for f in module.functions if f.name != name for n in function_nodes(module, f)}
    shared = {n.id for n in body if n.id in foreign and n.id not in parameters}
    if any(nodes[value].op not in _CONSTANT_ROOTS or not nodes[value].effect.is_pure or nodes[value].inputs for value in shared):
        fail("a shared nonconstant body value cannot be moved into a stage.")
    owned = {n.id for n in body} - parameters - shared
    flat = tuple(value for stage in stages for value in stage.nodes)
    if len(set(flat)) != len(flat) or set(flat) != owned:
        fail("stage nodes must exactly partition the complete owned function body.")
    # A subspan returned by a device function requires more than the current
    # identity result ABI. Construct declared non-owning views at the caller
    # instead: each stage receives the complete view as a parameter, so its
    # state aliases are ordinary identity returns. No data is read or copied.
    from triton.flagmega.ir.ops.core import get_definition
    from triton.flagmega.ir.memory_effect import MemoryEffect
    # A single stateful stage can also return an opaque tuple containing a
    # view to a pure consumer stage. The number of effect stages does not
    # determine whether that reference crosses the result ABI.
    views = tuple(n for n in body if n.id in owned
                  and isinstance(logical_type(n.type), RefType)
                  and get_definition(n.op).reference_view_input_parameters)
    view_ids = {n.id for n in views}
    for view in views:
        definition = get_definition(view.op)
        if (not view.effect.is_pure or any(v not in parameters | shared for v in view.inputs)
                or any(p.memory_effect != MemoryEffect.NONE for p in definition.parameters)
                or any(not isinstance(logical_type(nodes[p.read(view.inputs)].type), RefType)
                       for p in definition.reference_view_input_parameters)):
            fail("a hoisted reference view requires pure, caller-addressable operands without data access.")
    if views:
        parameters.update(view_ids)
        owned.difference_update(view_ids)
        stages = tuple(replace(s, nodes=tuple(v for v in s.nodes if v not in view_ids)) for s in stages)
        if any(not s.nodes for s in stages):
            fail("a reference-view-only stage has no executable body.")
    owner = {value: index for index, stage in enumerate(stages) for value in stage.nodes}
    body_order = {n.id: index for index, n in enumerate(body)}

    def stateful(node):
        return not node.effect.is_pure or node.op not in _STRUCTURAL and (
            _mutable_reference(node.type) or any(_mutable_reference(nodes[v].type) for v in node.inputs))

    state_order = [owner[n.id] for n in body if n.id in owner and stateful(n)]
    if state_order != sorted(state_order):
        fail("stateful stages must preserve the original mutation order.")
    for index, stage in enumerate(stages):
        if tuple(sorted(stage.nodes, key=body_order.__getitem__)) != tuple(stage.nodes):
            fail("each stage must retain original SSA order.")
        for identity in stage.nodes:
            node = nodes[identity]
            if node.op == "control.for":
                fail("structured loops must remain indivisible; outline their enclosing function separately.")
            if any(owner.get(value, -1) > index for value in node.inputs):
                fail("a stage consumes a later stage's value.")
            if index not in effect_owners and stateful(node):
                fail("stateful/reference accesses must remain in the designated effect stage.")
    calls = [n for n in module.nodes if n.op in {"builtin.call", "tir.call"} and n.attrs.get("callee") == name]
    if any(n.op == "control.for" and n.attrs.get("callee") == name for n in module.nodes):
        fail("a loop body cannot be replaced by a direct-call sequence.")
    if not calls:
        fail("the function has no direct invocation to outline.")
    if any(p.owner in {n.id for n in calls} for p in module.selection_points):
        fail("caller-owned proposals must be regenerated before outlining these calls.")

    new_parameters, fragments, stage_inputs, stage_outputs, redirects = [], [], [], [], {}
    used_ids = set(nodes)
    used_functions = set(module.function_map) | set(module.kernel_callable_map)

    def reserve(identity):
        if identity in used_ids:
            fail(f"generated SSA id {identity!r} already exists.")
        used_ids.add(identity)
        return identity

    for index, stage in enumerate(stages):
        stage_name = f"{name}.{stage.name}"
        if stage_name in used_functions:
            fail(f"generated function {stage_name!r} already exists.")
        used_functions.add(stage_name)
        members = set(stage.nodes)
        incoming = {v for identity in stage.nodes for v in nodes[identity].inputs if v not in members and v not in shared}
        incoming = tuple(sorted(incoming, key=body_order.__getitem__))
        outgoing = set(function.outputs) & members
        outgoing.update(v for node in body if owner.get(node.id, -1) > index for v in node.inputs if v in members)
        outgoing = tuple(value for value in stage.nodes if value in outgoing)
        if not outgoing:
            fail("each stage must publish a live value; effects cannot be kept by an unused call.")
        formals = []
        for ordinal, value in enumerate(incoming):
            identity = reserve(f"{stage_name}.arg{ordinal}")
            new_parameters.append(Node(identity, "builtin.var", (), nodes[value].type,
                                       attrs={"name": identity}, metadata={"outlined_from_value": value}))
            formals.append(identity)
        redirects[index] = dict(zip(incoming, formals, strict=True))
        stage_inputs.append(incoming)
        stage_outputs.append(outgoing)
        attrs = {key: value for key, value in function.attrs.items() if key not in _DERIVED_LAYOUT_ATTRS}
        fragments.append(Function(stage_name, tuple(formals), outgoing, attrs))

    call_ids = {n.id for n in calls}
    result = list(new_parameters)
    for node in module.nodes:
        if node.id in view_ids:
            continue
        if node.id in owner:
            remap = redirects[owner[node.id]]
            result.append(replace(node, inputs=tuple(remap.get(v, v) for v in node.inputs)))
        elif node.id not in call_ids:
            result.append(node)
        else:
            actuals = dict(zip(function.parameters, node.inputs, strict=True))
            # A global readonly root returned directly by the original
            # function does not acquire a stage result or a caller formal.
            actuals.update((value, value) for value in shared)
            for ordinal, view in enumerate(views):
                identity = reserve(f"{node.id}.reference_view{ordinal}")
                result.append(replace(view, id=identity, inputs=tuple(actuals[v] for v in view.inputs),
                                      metadata={**view.metadata, "outlined_from_value": view.id}))
                actuals[view.id] = identity
            for index, fragment in enumerate(fragments):
                output_ids = stage_outputs[index]
                final_single = index == len(fragments) - 1 and len(function.outputs) == 1
                if final_single and output_ids != function.outputs:
                    fail("the last stage must publish the original single result.")
                identity = node.id if final_single else reserve(f"{node.id}.stage{index}")
                types = tuple(nodes[value].type for value in output_ids)
                result_type = types[0] if len(types) == 1 else TupleType(types)
                result.append(Node(identity, node.op, tuple(actuals[v] for v in stage_inputs[index]), result_type,
                                   effect=node.effect if index in effect_owners else PURE,
                                   attrs={"callee": fragment.name},
                                   metadata={**node.metadata, "outlined_from_call": node.id, "outlined_stage": index}))
                if len(output_ids) == 1:
                    actuals[output_ids[0]] = identity
                else:
                    for field, value in enumerate(output_ids):
                        projection = reserve(f"{identity}.result{field}")
                        result.append(Node(projection, "builtin.get_item", (identity,), nodes[value].type,
                                           attrs={"index": field}, metadata={"outlined_from_value": value}))
                        actuals[value] = projection
            if len(function.outputs) != 1:
                result.append(Node(node.id, "builtin.tuple", tuple(actuals[v] for v in function.outputs), node.type,
                                   metadata={**node.metadata, "outlined_from_function": name}))
    functions = tuple(fragment for f in module.functions for fragment in (tuple(fragments) if f.name == name else (f,)))
    record = {"function": name, "stages": tuple({"function": f.name, "nodes": tuple(s.nodes),
                "inputs": incoming, "outputs": outgoing} for f, s, incoming, outgoing in
                zip(fragments, stages, stage_inputs, stage_outputs, strict=True)),
              "calls": tuple(n.id for n in calls),
              **({"effect_stage": effect_stage} if effect_stages is None else {"effect_stages": effect_stages}),
              **({"hoisted_reference_views": tuple(n.id for n in views)} if views else {}),
              "invalidated_boundary_layouts": {key: function.attrs[key] for key in _DERIVED_LAYOUT_ATTRS if key in function.attrs}}
    updated = replace(module, nodes=tuple(result), functions=functions,
                      metadata={**module.metadata, "outlined_function_stages":
                                (*module.metadata.get("outlined_function_stages", ()), record)})
    # Only obsolete formal vars are removed. Constant recipes and unrelated
    # compilation roots retain their complete original identities.
    live = {n.id for f in updated.functions for n in function_nodes(updated, f)}
    updated = replace(updated, nodes=tuple(n for n in updated.nodes if n.id not in parameters or n.id in live))
    return verify_module(updated)


__all__ = ["FunctionStage", "outline_function_stages"]

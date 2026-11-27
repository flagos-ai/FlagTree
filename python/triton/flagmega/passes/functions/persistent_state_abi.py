# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit entry-state ABI selection from an existing callee's ownership.

This is an opt-in interface change, not an equality rewrite: callers must
allocate the newly declared distributed state. Arithmetic and callee ownership
stay unchanged. Only single-use, complete representation adapters may disappear.
"""

from dataclasses import replace

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import PURE, DistributedType, IRModule, TensorType, verify_module
from triton.flagmega.ir.distributed_type import is_fully_replicated
from triton.flagmega.ir.entry_abi import redirect_entry_abi
from triton.flagmega.passes.functions.remove_unused import remove_unused_functions


def specialize_persistent_state_abi(module: IRModule, state_pairs) -> IRModule:
    """Expose explicitly requested (entry input, entry output) state pairs.

    The output must already be a materialized distributed call result. Its
    input must reach that same call through an otherwise unobserved chain of
    full-value Boxing/ShardedView adapters. A shared/full observer, changed
    scalar layout, partial value or different owner map rejects the request.
    Already specialized interfaces are checked identically and are idempotent.
    """
    pairs = tuple(tuple(pair) for pair in state_pairs)
    if not pairs:
        return module
    distributed = module.dialect == "distributed" or "auto_distribution" in module.metadata
    if not distributed or module.prim_functions or module.kernel_definitions or module.execution_functions:
        raise IRVerificationError("Persistent state ABI selection requires distributed graph IR before TIR.")
    nodes = module.node_map
    entry = module.function_map[module.entry]
    users = {node.id: [] for node in module.nodes}
    for node in module.nodes:
        for operand in node.inputs:
            users[operand].append(node.id)
    boundaries = {value for function in module.functions for value in function.outputs}
    changed = {}
    redirected = {}
    seen_inputs, seen_outputs = set(), set()

    def reject(message, value=None):
        raise IRVerificationError(f"Persistent state ABI: {message}", stage=module.stage, node_id=value)

    def logical(kind):
        return kind.tensor if isinstance(kind, DistributedType) else kind

    def full(kind):
        return isinstance(kind, TensorType) or isinstance(kind, DistributedType) and is_fully_replicated(kind)

    for pair in pairs:
        if len(pair) != 2:
            reject("each selection must name an entry input/output pair.")
        input_id, output_id = pair
        if input_id not in entry.parameters or output_id not in entry.outputs:
            reject("selected state must be an entry parameter and result.", input_id)
        if input_id in seen_inputs or output_id in seen_outputs:
            reject("selected states must have distinct inputs and outputs.", input_id)
        seen_inputs.add(input_id)
        seen_outputs.add(output_id)
        parameter, public = nodes[input_id], nodes[output_id]
        result = nodes[public.inputs[0]] if public.op == "distributed.boxing" else public
        owned = result.type
        if not isinstance(owned, DistributedType) or owned.partial is not None or owned.exclusive is not None:
            reject("result must have materialized, non-exclusive ownership.", result.id)
        if is_fully_replicated(owned):
            reject("result must already have a non-replicated owner map.", result.id)
        if logical(parameter.type) != owned.tensor or logical(public.type) != owned.tensor:
            reject("selection cannot change the logical state tensor.", input_id)
        if parameter.type != owned and not full(parameter.type):
            reject("entry input already declares different ownership.", input_id)
        if public.type != owned and not full(public.type):
            reject("entry output already declares different ownership.", output_id)
        if public is not result and (public.effect != PURE or users[public.id]):
            reject("published full state has another observer.", public.id)
        if public is not result and users[result.id] != [public.id]:
            reject("owned state result has another observer.", result.id)
        call = nodes[result.inputs[0]] if result.op == "builtin.get_item" else result
        if call.op != "builtin.call" or call.attrs["callee"] not in module.function_map:
            reject("owned result must come directly from a graph call.", result.id)
        callee = module.function_map[call.attrs["callee"]]
        paths = []
        for index, operand in enumerate(call.inputs):
            reverse_path = []
            value = nodes[operand]
            while value.id != input_id and value.op in {"distributed.boxing", "distributed.sharded_view"}:
                if len(value.inputs) != 1:
                    break
                reverse_path.append(value.id)
                value = nodes[value.inputs[0]]
            if value.id == input_id:
                paths.append((index, tuple(reversed(reverse_path))))
        if len(paths) != 1:
            reject("state must reach exactly one operand of its producing call.", input_id)
        index, adapters = paths[0]
        if nodes[callee.parameters[index]].type != owned or nodes[call.inputs[index]].type != owned:
            reject("callee input and output ownership must match exactly.", input_id)
        path = (input_id, *adapters)
        for step, value_id in enumerate(path):
            value = nodes[value_id]
            if logical(value.type) != owned.tensor or value.effect != PURE:
                reject("ingress adapters must preserve the complete logical tensor.", value_id)
            if value.type != owned and not full(value.type):
                reject("ingress contains another ownership or reduction boundary.", value_id)
            expected = path[step + 1] if step + 1 < len(path) else call.id
            if users[value_id] != [expected] or value_id in boundaries:
                reject("ingress state has a shared or exported observer.", value_id)
        changed[input_id] = replace(parameter, type=owned) if parameter.type != owned else parameter
        updated_call = changed.get(call.id, call)
        operands = list(updated_call.inputs)
        operands[index] = input_id
        changed[call.id] = replace(updated_call, inputs=tuple(operands))
        if public is not result:
            redirected[public.id] = result.id

    outputs = tuple(redirected.get(value, value) for value in entry.outputs)
    metadata = redirect_entry_abi(module.metadata, redirected,
                                  known_values=(*entry.parameters, *outputs), stage=module.stage)
    changed_types = {value for value in seen_inputs if changed[value].type != nodes[value].type}
    points = tuple(point for point in module.selection_points
                   if not (point.kind == "distribution" and point.owner in changed_types))
    point_ids = {point.id for point in points}
    result = replace(module, nodes=tuple(changed.get(node.id, node) for node in module.nodes),
                     functions=tuple(replace(function, outputs=outputs) if function.name == entry.name else function
                                     for function in module.functions), metadata=metadata,
                     selection_points=points,
                     selections=tuple(record for record in module.selections if record.point_id in point_ids))
    # The graph's selections describe materialized distributions. Preserve all
    # unaffected decisions; drop only points whose owners became dead adapters.
    return verify_module(remove_unused_functions(result))


__all__ = ["specialize_persistent_state_abi"]

# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Plan and propagate tensor layouts across reusable function boundaries.

The pass follows nncase's module-level algorithm: collect an identity variant,
an internal canonical variant, and variants demanded by callers; select one
variant per reusable callee with CP-SAT; specialize each callee once; then
rewrite every call to the same ABI.  The transformation runs to a fixed point
so nested layout chains cross one boundary at a time.

Pack, Unpack, Permute (the generalized Transpose form), and Bitcast are the
representable tensor-layout transforms in this stage.  Distributed boundary
layouts remain in the post-AutoDistribution pass because their physical
realization requires target placement policy.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from typing import Mapping

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import Function, IRModule, Node, PURE, TupleType, verify_module
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.tensors.pack import Pack
from triton.flagmega.ir.ops.tensors.unpack import Unpack
from triton.flagmega.ir.ops.core import get_definition
from triton.flagmega.ir.types import VectorType, data_type_to_data
from triton.flagmega.passes.functions.graph import function_nodes
from triton.flagmega.passes.vector_contracts import (
    is_transient_vectorization_boundary,
    retained_vectorization_roots,
)


_MAX_PLANNING_ITERATIONS = 8
_IDENTITY_VARIANT_PENALTY = 1000
_LOCAL_TRANSFORM_COST_SCALE = 10


@dataclass(frozen=True)
class _InputLayout:
    index: int
    parameter_id: str
    packed_type: object
    pack_attrs: dict[str, object]
    pack_ids: frozenset[str]
    logical_type: object
    restore_attrs: dict[str, object]
    transform_op: str = "tensors.pack"
    restore_op: str = "tensors.unpack"


@dataclass(frozen=True)
class _OutputLayout:
    index: int
    unpack_id: str | None
    packed_id: str
    logical_type: object
    packed_type: object
    unpack_attrs: dict[str, object]
    pack_attrs: dict[str, object]
    source_id: str
    source_pack: bool = False
    restore_at_caller: bool = True
    source_op: str = "tensors.pack"
    restore_op: str = "tensors.unpack"


@dataclass(frozen=True)
class _FunctionLayout:
    function: Function
    inputs: dict[int, _InputLayout]
    outputs: dict[int, _OutputLayout]
    candidate_id: str = "internal"
    is_identity: bool = False


def propagate_function_boundary_layouts(module: IRModule) -> IRModule:
    """Select and move Pack/Unpack contracts to internal function calls."""

    current = verify_module(module)
    for iteration in range(_MAX_PLANNING_ITERATIONS):
        rewritten = _propagate_one_iteration(
            current,
            enable_caller_output_demand=iteration == 0,
        )
        if rewritten is current:
            return current
        current = rewritten
    remaining = _collect_candidate_layouts(
        current, enable_caller_output_demand=False)
    if remaining:
        details = ", ".join(
            f"{name}: {[candidate.candidate_id for candidate in candidates]}"
            for name, candidates in sorted(remaining.items())
        )
        raise IRVerificationError(
            "Function boundary layout planning did not converge within "
            f"{_MAX_PLANNING_ITERATIONS} iterations; remaining variants: {details}.",
            stage=current.stage,
        )
    return current


def _propagate_one_iteration(
    module: IRModule,
    *,
    enable_caller_output_demand: bool,
) -> IRModule:
    """Materialize one CP-SAT-selected layout layer."""

    verify_module(module)
    users = _user_map(module)
    calls = _calls_by_callee(module)
    candidates = _collect_candidate_layouts(
        module,
        users=users,
        calls=calls,
        enable_caller_output_demand=enable_caller_output_demand,
    )
    layouts = _select_layouts(module, candidates, users, calls)
    if not layouts:
        return module

    occupied = {node.id for node in module.nodes}
    node_map = module.node_map
    # Caller-demand candidates carry a symbolic id while planning.  Allocate
    # the concrete callee-local producer only for the selected variant.
    for name, layout in tuple(layouts.items()):
        outputs: dict[int, _OutputLayout] = {}
        for index, item in layout.outputs.items():
            if item.source_pack:
                packed_id = _fresh_id(
                    f"{item.source_id}.boundary_pack", occupied)
                occupied.add(packed_id)
                item = replace(item, packed_id=packed_id)
            outputs[index] = item
        layouts[name] = replace(layout, outputs=outputs)

    # Plan raw typed projections before rewriting call sites, so a later call
    # can consume a previous call's packed result without Pack(Unpack(...)).
    raw_projection_ids: dict[str, str] = {}
    projection_layouts: dict[str, _OutputLayout] = {}
    for node in module.nodes:
        if node.op == "builtin.call":
            layout = layouts.get(str(node.attrs["callee"]))
            if (
                layout is not None
                and len(layout.function.outputs) == 1
                and 0 in layout.outputs
            ):
                raw_id = _fresh_id(f"{node.id}.boundary_packed", occupied)
                occupied.add(raw_id)
                raw_projection_ids[node.id] = raw_id
                projection_layouts[node.id] = layout.outputs[0]
            continue
        if node.op != "builtin.get_item" or len(node.inputs) != 1:
            continue
        producer = node_map[node.inputs[0]]
        if producer.op != "builtin.call":
            continue
        layout = layouts.get(str(producer.attrs["callee"]))
        if layout is None:
            continue
        output = layout.outputs.get(int(node.attrs["index"]))
        if output is None:
            continue
        raw_id = _fresh_id(f"{node.id}.boundary_packed", occupied)
        occupied.add(raw_id)
        raw_projection_ids[node.id] = raw_id
        projection_layouts[node.id] = output

    removed_substitutions: dict[str, str] = {}
    removed_owners: dict[str, str] = {}
    parameter_types: dict[str, object] = {}
    raw_parameter_restores: dict[str, Node] = {}
    raw_parameter_use_nodes: dict[str, frozenset[str]] = {}
    output_replacements: dict[tuple[str, int], str] = {}
    for name, layout in layouts.items():
        owned = {node.id for node in function_nodes(module, layout.function)}
        direct_raw_output_parameters = {
            item.packed_id
            for item in layout.outputs.values()
            if item.packed_id in layout.function.parameters
        }
        source_pack_parameters = {
            item.source_id
            for item in layout.outputs.values()
            if item.source_pack and item.source_id in layout.function.parameters
        }
        for item in layout.inputs.values():
            parameter_types[item.parameter_id] = item.packed_type
            for pack_id in item.pack_ids:
                removed_substitutions[pack_id] = item.parameter_id
                removed_owners[pack_id] = item.parameter_id
            raw_users = frozenset(
                user.id
                for user in users.get(item.parameter_id, ())
                if user.id in owned and user.id not in item.pack_ids
            )
            if (
                raw_users
                or item.parameter_id in layout.function.outputs
                or item.parameter_id in direct_raw_output_parameters
                or item.parameter_id in source_pack_parameters
            ):
                restore_id = _fresh_id(
                    f"{item.parameter_id}.boundary_restore", occupied)
                occupied.add(restore_id)
                raw_parameter_restores[item.parameter_id] = Node(
                    restore_id,
                    item.restore_op,
                    (item.parameter_id,),
                    item.logical_type,
                    PURE,
                    item.restore_attrs,
                    {
                        "introduced_by": "FunctionBoundaryLayoutPropagation",
                        "boundary_layout": "raw_parameter_restore",
                        "callee": name,
                        "parameter_index": item.index,
                    },
                )
                raw_parameter_use_nodes[item.parameter_id] = raw_users
        for item in layout.outputs.values():
            output_replacements[(name, item.index)] = (
                raw_parameter_restores[item.packed_id].id
                if item.packed_id in raw_parameter_restores
                else item.packed_id
            )
            if item.unpack_id is not None:
                removed_substitutions[item.unpack_id] = item.packed_id
                removed_owners[item.unpack_id] = item.packed_id

    # Fold a matching caller transform over the compatibility view emitted
    # for a newly transformed tuple field.
    for node in module.nodes:
        if len(node.inputs) != 1:
            continue
        raw_id = raw_projection_ids.get(node.inputs[0])
        if raw_id is None:
            continue
        output = projection_layouts[node.inputs[0]]
        if (
            node.op == output.source_op
            and node.type == output.packed_type
            and node.attrs == output.pack_attrs
        ):
            removed_substitutions[node.id] = raw_id
            removed_owners[node.id] = raw_id

    source_packs: dict[str, list[Node]] = {}
    for name, layout in layouts.items():
        for item in layout.outputs.values():
            if not item.source_pack:
                continue
            source_packs.setdefault(item.source_id, []).append(Node(
                item.packed_id,
                item.source_op,
                (item.source_id,),
                item.packed_type,
                PURE,
                item.pack_attrs,
                {
                    "introduced_by": "FunctionBoundaryLayoutPropagation",
                    "boundary_layout": "packed_output",
                    "callee": name,
                    "output_index": item.index,
                },
            ))

    function_output_roots = {
        output for function in module.functions for output in function.outputs
    }
    nodes: list[Node] = []
    for node in module.nodes:
        if node.id in removed_substitutions:
            continue
        if node.id in parameter_types:
            nodes.append(replace(
                node,
                type=parameter_types[node.id],
                metadata={
                    **dict(node.metadata),
                    "boundary_layout": "packed",
                    "introduced_by": "FunctionBoundaryLayoutPropagation",
                },
            ))
            if node.id in raw_parameter_restores:
                nodes.append(raw_parameter_restores[node.id])
            for source_pack in source_packs.get(node.id, ()):
                source_input = (
                    raw_parameter_restores[node.id].id
                    if node.id in raw_parameter_restores else node.id
                )
                nodes.append(replace(source_pack, inputs=(source_input,)))
            continue
        if node.op == "builtin.call" and str(node.attrs["callee"]) in layouts:
            layout = layouts[str(node.attrs["callee"])]
            helpers: list[Node] = []
            arguments = list(node.inputs)
            for index, item in layout.inputs.items():
                original_argument_id = arguments[index]
                raw_id = raw_projection_ids.get(original_argument_id)
                if raw_id is not None and projection_layouts[original_argument_id].packed_type == item.packed_type:
                    arguments[index] = raw_id
                    continue
                argument_id = _resolve(original_argument_id, removed_substitutions)
                # A projection may already have been rewritten earlier in the
                # node list. Its type is known by the precomputed plan even
                # though the helper does not exist in the original node_map.
                if argument_id in raw_projection_ids.values():
                    arguments[index] = argument_id
                    continue
                argument = node_map[argument_id]
                if argument.type == item.packed_type:
                    arguments[index] = argument_id
                elif (
                    argument.op == item.restore_op
                    and argument.attrs == item.restore_attrs
                    and node_map[argument.inputs[0]].type == item.packed_type
                ):
                    arguments[index] = argument.inputs[0]
                else:
                    helper_id = _fresh_id(
                        f"{node.id}.arg{index}.boundary_pack", occupied)
                    occupied.add(helper_id)
                    helper_type = get_definition(item.transform_op).infer_type(
                        (argument,), item.pack_attrs)
                    if helper_type != item.packed_type:
                        # Discovery proves the callee contract, but this call
                        # may carry an incompatible symbolic/alternate layout.
                        # Keep the original function untouched rather than
                        # fabricating a coercion.
                        raise ValueError(
                            f"Call {node.id!r} argument {index} cannot satisfy the unique "
                            f"packed boundary of @{layout.function.name}."
                        )
                    helper = Node(
                        helper_id,
                        item.transform_op,
                        (argument_id,),
                        helper_type,
                        PURE,
                        item.pack_attrs,
                        {
                            "introduced_by": "FunctionBoundaryLayoutPropagation",
                            "callee": layout.function.name,
                            "parameter_index": index,
                        },
                    )
                    helpers.append(helper)
                    arguments[index] = helper_id
            nodes.extend(helpers)
            rewritten_call = replace(
                node,
                inputs=tuple(arguments),
                type=_result_type_after_layout(module, layout),
                metadata={
                    **dict(node.metadata),
                    "introduced_by": "FunctionBoundaryLayoutPropagation",
                },
            )
            if node.id not in raw_projection_ids:
                nodes.append(rewritten_call)
                continue
            output = projection_layouts[node.id]
            raw = replace(
                rewritten_call,
                id=raw_projection_ids[node.id],
                type=output.packed_type,
                metadata={
                    **dict(rewritten_call.metadata),
                    "boundary_layout": "packed",
                },
            )
            nodes.append(raw)
            if _needs_logical_projection(
                node, users, removed_substitutions, layouts, function_output_roots
            ):
                nodes.append(Node(
                    node.id,
                    output.restore_op,
                    (raw.id,),
                    output.logical_type,
                    PURE,
                    output.unpack_attrs,
                    {
                        **dict(node.metadata),
                        "introduced_by": "FunctionBoundaryLayoutPropagation",
                        "boundary_layout": "logical_view",
                    },
                ))
            else:
                removed_substitutions[node.id] = raw.id
                removed_owners[node.id] = raw.id
            continue
        if node.id in raw_projection_ids:
            output = projection_layouts[node.id]
            raw = Node(
                raw_projection_ids[node.id],
                "builtin.get_item",
                node.inputs,
                output.packed_type,
                PURE,
                node.attrs,
                {
                    **dict(node.metadata),
                    "introduced_by": "FunctionBoundaryLayoutPropagation",
                    "boundary_layout": "packed",
                },
            )
            nodes.append(raw)
            if _needs_logical_projection(
                node, users, removed_substitutions, layouts, function_output_roots
            ):
                nodes.append(Node(
                    node.id,
                    output.restore_op,
                    (raw.id,),
                    output.logical_type,
                    PURE,
                    output.unpack_attrs,
                    {
                        **dict(node.metadata),
                        "introduced_by": "FunctionBoundaryLayoutPropagation",
                        "boundary_layout": "logical_view",
                    },
                ))
            else:
                removed_substitutions[node.id] = raw.id
                removed_owners[node.id] = raw.id
            continue
        rewritten = replace(
            node,
            inputs=tuple(
                raw_parameter_restores[value].id
                if value in raw_parameter_restores
                and node.id in raw_parameter_use_nodes[value]
                else _resolve(value, removed_substitutions)
                for value in node.inputs
            ),
        )
        nodes.append(rewritten)
        for source_pack in source_packs.get(node.id, ()):
            nodes.append(replace(source_pack, inputs=(rewritten.id,)))

    functions = _rewritten_functions(module, layouts)
    functions = tuple(replace(
        function,
        outputs=tuple(
            _resolve(
                output_replacements.get(
                    (function.name, index),
                    raw_parameter_restores[output].id
                    if output in raw_parameter_restores else output,
                ),
                removed_substitutions,
            )
            for index, output in enumerate(function.outputs)
        ),
        attrs={
            **dict(function.attrs),
            **({
                "function_boundary_layout": {
                    "inputs": tuple(sorted(layouts[function.name].inputs)),
                    "outputs": tuple(sorted(layouts[function.name].outputs)),
                }
            } if function.name in layouts else {}),
        },
    ) for function in functions)

    points = tuple(
        replace(point, owner=_resolve(point.owner, removed_owners))
        if point.owner in removed_owners else point
        for point in module.selection_points
    )
    choices = dict(module.metadata.get("function_boundary_layout_choices", {}))
    for function_name in layouts:
        choices.pop(function_name, None)
    decisions = list(module.metadata.get("function_boundary_layout_decisions", ()))
    decisions.extend(
        {
            "function": function_name,
            "candidate": layout.candidate_id,
            "solver": "ortools-cp-sat",
        }
        for function_name, layout in sorted(layouts.items())
    )
    metadata = dict(module.metadata)
    if choices:
        metadata["function_boundary_layout_choices"] = choices
    else:
        metadata.pop("function_boundary_layout_choices", None)
    metadata["function_boundary_layout_decisions"] = tuple(decisions)
    return verify_module(replace(
        module,
        nodes=tuple(nodes),
        functions=functions,
        selection_points=points,
        metadata=metadata,
    ))


def _collect_candidate_layouts(
    module: IRModule,
    *,
    users: dict[str, tuple[Node, ...]] | None = None,
    calls: dict[str, tuple[Node, ...]] | None = None,
    enable_caller_output_demand: bool = True,
) -> dict[str, tuple[_FunctionLayout, ...]]:
    users = _user_map(module) if users is None else users
    calls = _calls_by_callee(module) if calls is None else calls
    result: dict[str, tuple[_FunctionLayout, ...]] = {}
    for function in module.functions:
        function_calls = calls.get(function.name, ())
        if function.name == module.entry or not function_calls:
            continue
        internal = _discover_internal_layout(module, function, users)
        internal = _filter_inapplicable_inputs(module, internal, function_calls)
        identity = _FunctionLayout(
            function, {}, {}, candidate_id="identity", is_identity=True)
        variants = [identity]
        if internal.inputs or internal.outputs:
            variants.append(internal)
        for call in function_calls if enable_caller_output_demand else ():
            demanded = _discover_caller_output_demand(
                module, function, call, users)
            if not demanded:
                continue
            # nncase keeps an already discovered internal output contract and
            # fills only previously empty output ports from caller demand.
            outputs = dict(internal.outputs)
            for index, item in demanded.items():
                outputs.setdefault(index, item)
            variant = _FunctionLayout(
                function,
                dict(internal.inputs),
                outputs,
                candidate_id=f"caller:{call.id}",
            )
            variant = _filter_inapplicable_inputs(
                module, variant, function_calls)
            if variant.inputs or variant.outputs:
                variants.append(variant)
        distinct: list[_FunctionLayout] = []
        keys: set[object] = set()
        for variant in sorted(
            variants,
            key=lambda value: (not value.is_identity, value.candidate_id),
        ):
            key = _layout_key(variant)
            if key in keys:
                continue
            keys.add(key)
            distinct.append(variant)
        if len(distinct) > 1:
            result[function.name] = tuple(distinct)
    return result


def _discover_internal_layout(
    module: IRModule,
    function: Function,
    users: dict[str, tuple[Node, ...]],
) -> _FunctionLayout:
    owned = {node.id for node in function_nodes(module, function)}
    retained_roots = retained_vectorization_roots(module)
    inputs: dict[int, _InputLayout] = {}
    for index, parameter_id in enumerate(function.parameters):
        parameter = module.node_map[parameter_id]
        candidates = tuple(
            node
            for node in users.get(parameter_id, ())
            if node.id in owned
            and node.op in {
                "tensors.pack",
                "tensors.unpack",
                "tensors.permute",
            }
            and node.metadata.get("boundary_layout") != "raw_parameter_restore"
            and not is_transient_vectorization_boundary(
                module, node, retained_roots=retained_roots)
        )
        if not candidates:
            continue
        first = candidates[0]
        if any(
            node.op != first.op
            or node.attrs != first.attrs
            or node.type != first.type
            for node in candidates[1:]
        ):
            continue
        restore_op, restore_attrs = _inverse_transform(first, parameter)
        inputs[index] = _InputLayout(
            index,
            parameter_id,
            first.type,
            dict(first.attrs),
            frozenset(node.id for node in candidates),
            parameter.type,
            restore_attrs,
            first.op,
            restore_op,
        )

    outputs: dict[int, _OutputLayout] = {}
    for index, output_id in enumerate(function.outputs):
        output = module.node_map[output_id]
        if output.op not in {
            "tensors.unpack",
            "tensors.permute",
            "tensors.bitcast",
        } or len(output.inputs) != 1:
            continue
        if output.metadata.get("boundary_layout") == "packed_output":
            # A caller-demand source transform is already the selected ABI,
            # not a fresh internal transform to hoist in reverse.
            continue
        if is_transient_vectorization_boundary(
            module, output, retained_roots=retained_roots
        ):
            # Schedule-only Pack/Unpack expressions disappear before TIR.
            # Moving them into a reusable ABI would leave vector formals and
            # results after their compute has returned to the scalar semantic
            # operation.
            continue
        # Removing the view is safe only when it is a pure function boundary;
        # otherwise internal logical consumers still require it.
        if users.get(output.id):
            continue
        packed = module.node_map[output.inputs[0]]
        source_op, source_attrs = _inverse_transform(output, packed)
        outputs[index] = _OutputLayout(
            index,
            output.id,
            packed.id,
            output.type,
            packed.type,
            dict(output.attrs),
            source_attrs,
            output.id,
            source_op=source_op,
            restore_op=output.op,
        )
    return _FunctionLayout(function, inputs, outputs)


def _discover_caller_output_demand(
    module: IRModule,
    function: Function,
    call: Node,
    users: dict[str, tuple[Node, ...]],
) -> dict[int, _OutputLayout]:
    retained_roots = retained_vectorization_roots(module)
    values: dict[int, str] = {}
    if len(function.outputs) == 1:
        values[0] = call.id
    else:
        for user in users.get(call.id, ()):
            if user.op != "builtin.get_item" or len(user.inputs) != 1:
                continue
            index = int(user.attrs["index"])
            if 0 <= index < len(function.outputs):
                values[index] = user.id

    outputs: dict[int, _OutputLayout] = {}
    for index, value_id in values.items():
        demands = tuple(
            user
            for user in users.get(value_id, ())
            if user.op in {
                "tensors.pack",
                "tensors.unpack",
                "tensors.permute",
                "tensors.bitcast",
            }
            and len(user.inputs) == 1
            and not is_transient_vectorization_boundary(
                module, user, retained_roots=retained_roots)
        )
        if not demands:
            continue
        first = demands[0]
        # One call contributes one ABI variant. As in nncase, deterministic
        # node/user order chooses the first direct demand for each tuple port;
        # other consumers retain the caller-side compatibility view.
        source_id = function.outputs[index]
        source = module.node_map[source_id]
        restore_op, restore_attrs = _inverse_transform(first, source)
        outputs[index] = _OutputLayout(
            index=index,
            unpack_id=None,
            packed_id=f"{source_id}.boundary_pack",
            logical_type=source.type,
            packed_type=first.type,
            unpack_attrs=restore_attrs,
            pack_attrs=dict(first.attrs),
            source_id=source_id,
            source_pack=True,
            source_op=first.op,
            restore_op=restore_op,
        )
    return outputs


def _select_layouts(
    module: IRModule,
    candidates: dict[str, tuple[_FunctionLayout, ...]],
    users: dict[str, tuple[Node, ...]],
    calls: dict[str, tuple[Node, ...]],
) -> dict[str, _FunctionLayout]:
    if not candidates:
        return {}
    try:
        from ortools.sat.python import cp_model
    except ImportError as error:  # pragma: no cover - exercised in packaging.
        raise RuntimeError(
            "FunctionBoundaryLayoutPropagation requires OR-Tools CP-SAT; "
            "install ortools==9.10.4067."
        ) from error

    model = cp_model.CpModel()
    variables = {}
    for function_name, variants in candidates.items():
        choices = []
        for index, _ in enumerate(variants):
            variable = model.NewBoolVar(
                f"function_layout__{_sanitize_name(function_name)}__{index}")
            variables[(function_name, index)] = variable
            choices.append(variable)
        model.AddExactlyOne(choices)

    overrides = module.metadata.get("function_boundary_layout_choices", {})
    if not isinstance(overrides, Mapping):
        raise IRVerificationError(
            "metadata.function_boundary_layout_choices must be a mapping from "
            "function name to candidate id.",
            stage=module.stage,
        )
    for function_name, candidate_id in overrides.items():
        variants = candidates.get(str(function_name))
        if variants is None:
            raise IRVerificationError(
                f"Function-boundary layout override references function "
                f"{function_name!r} with no candidates.",
                stage=module.stage,
            )
        matching = [
            index
            for index, variant in enumerate(variants)
            if variant.candidate_id == str(candidate_id)
        ]
        if not matching:
            valid = tuple(variant.candidate_id for variant in variants)
            raise IRVerificationError(
                f"Unknown function-boundary layout candidate {candidate_id!r} "
                f"for @{function_name}; expected one of {valid!r}.",
                stage=module.stage,
            )
        model.Add(variables[(str(function_name), matching[0])] == 1)

    model.Minimize(sum(
        _estimate_variant_cost(
            module,
            variant,
            users,
            calls[function_name],
        ) * variables[(function_name, index)]
        for function_name, variants in candidates.items()
        for index, variant in enumerate(variants)
    ))
    validation = model.Validate()
    if validation:
        raise IRVerificationError(
            f"Function-boundary layout CP-SAT model is invalid: {validation}",
            stage=module.stage,
        )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = _positive_env_float(
        "FLAGMEGA_LAYOUT_SOLVE_MAX_TIME", 10.0)
    solver.parameters.num_search_workers = _positive_env_int(
        "FLAGMEGA_LAYOUT_SOLVE_WORKERS", max((os.cpu_count() or 2) // 2, 1))
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise IRVerificationError(
            "Function-boundary layout CP-SAT solve failed: "
            f"{solver.StatusName(status)}.",
            stage=module.stage,
        )
    selected = {}
    for function_name, variants in candidates.items():
        variant = next(
            variant
            for index, variant in enumerate(variants)
            if solver.BooleanValue(variables[(function_name, index)])
        )
        if not variant.is_identity:
            selected[function_name] = variant
    return selected


def _estimate_variant_cost(
    module: IRModule,
    layout: _FunctionLayout,
    users: dict[str, tuple[Node, ...]],
    calls: tuple[Node, ...],
) -> int:
    owned = {node.id for node in function_nodes(module, layout.function)}
    transform_count = sum(
        node.op in {
            "tensors.pack",
            "tensors.unpack",
            "tensors.permute",
            "tensors.bitcast",
        }
        for node in module.nodes
        if node.id in owned
    )
    call_count = max(len(calls), 1)
    if layout.is_identity:
        return (
            _IDENTITY_VARIANT_PENALTY
            + transform_count * _LOCAL_TRANSFORM_COST_SCALE
        ) * call_count

    removed = sum(len(item.pack_ids) for item in layout.inputs.values())
    removed += sum(
        item.unpack_id is not None for item in layout.outputs.values())
    added = sum(
        bool(
            tuple(
                user
                for user in users.get(item.parameter_id, ())
                if user.id in owned and user.id not in item.pack_ids
            )
            or item.parameter_id in layout.function.outputs
            or any(
                output.packed_id == item.parameter_id
                or (output.source_pack and output.source_id == item.parameter_id)
                for output in layout.outputs.values()
            )
        )
        for item in layout.inputs.values()
    )
    added += sum(item.source_pack for item in layout.outputs.values())
    local_cost = (
        transform_count - removed + added
    ) * _LOCAL_TRANSFORM_COST_SCALE
    adapter_cost = len(layout.inputs) + sum(
        item.restore_at_caller for item in layout.outputs.values())
    satisfied_demands = 0
    for call in calls:
        values = {0: call.id}
        if len(layout.function.outputs) != 1:
            values = {
                int(user.attrs["index"]): user.id
                for user in users.get(call.id, ())
                if user.op == "builtin.get_item"
            }
        for index, item in layout.outputs.items():
            value_id = values.get(index)
            if value_id is None:
                continue
            satisfied_demands += sum(
                user.op == item.source_op
                and user.attrs == item.pack_attrs
                and user.type == item.packed_type
                for user in users.get(value_id, ())
            )
    return (
        (local_cost + adapter_cost) * call_count
        - satisfied_demands * _LOCAL_TRANSFORM_COST_SCALE
    )


def _filter_inapplicable_inputs(
    module: IRModule,
    layout: _FunctionLayout,
    calls: tuple[Node, ...],
) -> _FunctionLayout:
    """Reject an input contract when one caller cannot apply its transform."""

    applicable: dict[int, _InputLayout] = {}
    for index, item in layout.inputs.items():
        for call in calls:
            argument = module.node_map[call.inputs[index]]
            if argument.type == item.packed_type:
                continue
            if (
                argument.op == item.restore_op
                and argument.attrs == item.restore_attrs
                and module.node_map[argument.inputs[0]].type == item.packed_type
            ):
                continue
            try:
                inferred = get_definition(item.transform_op).infer_type(
                    (argument,), item.pack_attrs)
            except (KeyError, TypeError, ValueError):
                break
            if inferred != item.packed_type:
                break
        else:
            applicable[index] = item
    return replace(layout, inputs=applicable)


def _rewritten_functions(
    module: IRModule,
    layouts: dict[str, _FunctionLayout],
) -> tuple[Function, ...]:
    return tuple(
        replace(
            function,
            outputs=tuple(
                layouts[function.name].outputs[index].packed_id
                if function.name in layouts and index in layouts[function.name].outputs
                else output
                for index, output in enumerate(function.outputs)
            ),
        )
        for function in module.functions
    )


def _result_type_after_layout(module: IRModule, layout: _FunctionLayout):
    fields = tuple(
        layout.outputs[index].packed_type
        if index in layout.outputs else module.node_map[output].type
        for index, output in enumerate(layout.function.outputs)
    )
    return fields[0] if len(fields) == 1 else TupleType(fields)


def _needs_logical_projection(
    projection: Node,
    users: dict[str, tuple[Node, ...]],
    substitutions: dict[str, str],
    layouts: dict[str, _FunctionLayout],
    function_output_roots: set[str],
) -> bool:
    if projection.id in function_output_roots:
        return True
    for user in users.get(projection.id, ()):
        if user.id in substitutions:
            continue
        if user.op == "builtin.call":
            layout = layouts.get(str(user.attrs["callee"]))
            if layout is not None:
                positions = tuple(
                    index for index, value in enumerate(user.inputs)
                    if value == projection.id
                )
                if positions and all(index in layout.inputs for index in positions):
                    continue
        return True
    return False


def _calls_by_callee(module: IRModule) -> dict[str, tuple[Node, ...]]:
    result: dict[str, list[Node]] = {}
    live = {
        node.id
        for function in module.functions
        for node in function_nodes(module, function)
    }
    for node in module.nodes:
        if node.id in live and node.op == "builtin.call":
            result.setdefault(str(node.attrs["callee"]), []).append(node)
    return {name: tuple(values) for name, values in result.items()}


def _user_map(module: IRModule) -> dict[str, tuple[Node, ...]]:
    values: dict[str, list[Node]] = {node.id: [] for node in module.nodes}
    for node in module.nodes:
        for input_id in dict.fromkeys(node.inputs):
            values[input_id].append(node)
    return {node_id: tuple(users) for node_id, users in values.items()}


def _resolve(value: str | None, substitutions: dict[str, str]) -> str | None:
    while value in substitutions:
        replacement = substitutions[value]
        if replacement == value:
            break
        value = replacement
    return value


def _unpack_attrs_from_pack(attrs: Mapping[str, object]) -> dict[str, object]:
    if "axes" in attrs:
        return {"axes": tuple(int(value) for value in attrs["axes"])}
    return {"axis": int(attrs["axis"])}


def _pack_attrs_from_unpack(unpack: Node, packed: Node) -> dict[str, object]:
    packed_tensor = tensor_of(packed.type)
    if not isinstance(packed_tensor.dtype, VectorType):
        raise IRVerificationError(
            f"Boundary Unpack {unpack.id!r} does not consume a VectorType.",
            node_id=unpack.id,
        )
    if "axes" in unpack.attrs:
        axes = tuple(int(value) for value in unpack.attrs["axes"])
        lanes = packed_tensor.dtype.lanes[:len(axes)]
        return {"lanes": lanes, "axes": axes}
    return {
        "lanes": packed_tensor.dtype.lanes,
        "axis": int(unpack.attrs["axis"]),
    }


def _inverse_transform(
    transform: Node,
    source: Node,
) -> tuple[str, dict[str, object]]:
    if transform.op == "tensors.pack":
        return "tensors.unpack", _unpack_attrs_from_pack(transform.attrs)
    if transform.op == "tensors.unpack":
        return "tensors.pack", _pack_attrs_from_unpack(transform, source)
    if transform.op == "tensors.permute":
        axes = tuple(int(value) for value in transform.attrs["axes"])
        inverse = [0] * len(axes)
        for output_axis, input_axis in enumerate(axes):
            inverse[input_axis] = output_axis
        return "tensors.permute", {"axes": tuple(inverse)}
    if transform.op == "tensors.bitcast":
        return "tensors.bitcast", {
            "dtype": data_type_to_data(tensor_of(source.type).dtype)
        }
    raise IRVerificationError(
        f"Unsupported function-boundary transform {transform.op!r}.",
        node_id=transform.id,
    )


def _layout_key(layout: _FunctionLayout):
    if layout.is_identity:
        return ("identity",)
    return (
        tuple(
            (
                index,
                item.packed_type,
                item.transform_op,
                _attrs_key(item.pack_attrs),
            )
            for index, item in sorted(layout.inputs.items())
        ),
        tuple(
            (
                index,
                item.logical_type,
                item.packed_type,
                item.source_op,
                _attrs_key(item.pack_attrs),
                item.restore_op,
                item.source_pack,
                item.restore_at_caller,
            )
            for index, item in sorted(layout.outputs.items())
        ),
    )


def _attrs_key(attrs: Mapping[str, object]):
    return tuple(
        (key, _freeze_attr(value))
        for key, value in sorted(attrs.items())
    )


def _freeze_attr(value):
    if isinstance(value, Mapping):
        return tuple(
            (key, _freeze_attr(item)) for key, item in sorted(value.items())
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_attr(item) for item in value)
    return value


def _sanitize_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def _positive_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive number, got {raw!r}.") from error
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {raw!r}.")
    return value


def _positive_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}.") from error
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {raw!r}.")
    return value


def _fresh_id(stem: str, occupied: set[str]) -> str:
    if stem not in occupied:
        return stem
    index = 1
    while f"{stem}.{index}" in occupied:
        index += 1
    return f"{stem}.{index}"


__all__ = ["propagate_function_boundary_layouts"]

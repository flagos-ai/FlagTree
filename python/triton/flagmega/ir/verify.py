# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Structural and semantic verification for editable FlagMega checkpoints."""

from __future__ import annotations

from weakref import WeakValueDictionary

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.constant_recipe import ConstantPhase
from triton.flagmega.ir.model import (
    AnyType,
    CallableType,
    DistributedType,
    IRModule,
    InvalidType,
    Node,
    NoneType,
    RefType,
    TensorType,
    TupleType,
    logical_type,
)
from triton.flagmega.ir.distributed_type import is_distributable
from triton.flagmega.ir.types import MaskVectorType, PointerType, VectorType
from triton.flagmega.ir.ops.core import get_definition
from triton.flagmega.ir.tir import (
    PrimFunction,
    kernel_dispatch_of,
    verify_execution_functions,
    verify_prim_function,
)


# IRModule recursively freezes its graph and metadata in ``__post_init__``.
# Verification is therefore stable for one object identity.  Compiler layers
# deliberately verify at public boundaries, and several of those boundaries
# see the exact same module; keep those checks cheap without trusting an equal
# but newly constructed/edit-resumed module.  Weak values prevent this cache
# from extending checkpoint lifetime, while the identity check makes reused
# CPython ids harmless.
_VERIFIED_MODULES: WeakValueDictionary[int, IRModule] = WeakValueDictionary()


def verify_module(
    module: IRModule,
    *,
    expected_stage: str | None = None,
    expected_dialect: str | None = None,
) -> IRModule:
    try:
        phase = ConstantPhase(str(module.metadata.get("constant_phase", ConstantPhase.OPEN.value)))
    except ValueError as error:
        raise IRVerificationError(
            f"Unknown constant phase {module.metadata.get('constant_phase')!r}.", stage=module.stage) from error
    if expected_stage is not None and module.stage != expected_stage:
        raise IRVerificationError(
            f"Expected stage {expected_stage!r}, got {module.stage!r}.",
            stage=module.stage,
        )
    if expected_dialect is not None and module.dialect != expected_dialect:
        raise IRVerificationError(
            f"Expected dialect {expected_dialect!r}, got {module.dialect!r}.",
            stage=module.stage,
        )
    if _VERIFIED_MODULES.get(id(module)) is module:
        return module
    if not module.nodes:
        raise IRVerificationError("An IR module must contain at least one node.", stage=module.stage)

    node_map: dict[str, Node] = {}
    for node in module.nodes:
        if node.id in node_map:
            raise IRVerificationError(f"Duplicate node id {node.id!r}.", stage=module.stage, node_id=node.id)
        for input_id in node.inputs:
            if input_id not in node_map:
                raise IRVerificationError(
                    f"Node {node.id!r} references missing or non-topological input {input_id!r}.",
                    stage=module.stage,
                    node_id=node.id,
                )
        _verify_type(node.type, module.stage, node.id)
        try:
            definition = get_definition(node.op)
        except KeyError as error:
            raise IRVerificationError(
                f"No op definition is registered for {node.op!r}.",
                stage=module.stage,
                node_id=node.id,
            ) from error
        definition.verify(node, module)
        node_map[node.id] = node

    assets = tuple(
        node for node in module.nodes
        if node.op == "builtin.const_asset"
        or (node.op == "tir.buffer" and "constant_recipe" in node.metadata)
    )
    if phase == ConstantPhase.OPEN and (assets or module.constant_recipes):
        raise IRVerificationError(
            "constants_open IR cannot contain constant recipes or builtin.const_asset.", stage=module.stage)
    if phase == ConstantPhase.FROZEN:
        _verify_constant_recipes(module, assets)

    function_names: set[str] = set()
    for function in module.functions:
        if function.name in function_names:
            raise IRVerificationError(f"Duplicate function name {function.name!r}.", stage=module.stage)
        function_names.add(function.name)
        for parameter in function.parameters:
            if parameter not in node_map or node_map[parameter].op != "builtin.var":
                raise IRVerificationError(
                    f"Function {function.name!r} parameter {parameter!r} must reference builtin.var.",
                    stage=module.stage,
                    node_id=parameter,
                )
        for output in function.outputs:
            if output not in node_map:
                raise IRVerificationError(
                    f"Function {function.name!r} output {output!r} does not exist.",
                    stage=module.stage,
                    node_id=output,
                )
    if module.entry not in function_names:
        raise IRVerificationError(f"Entry function {module.entry!r} does not exist.", stage=module.stage)

    prim_names: set[str] = set()
    for function in module.prim_functions:
        if not isinstance(function, PrimFunction):
            raise IRVerificationError("IRModule prim_functions must contain PrimFunction values.", stage=module.stage)
        if function.name in function_names or function.name in prim_names:
            raise IRVerificationError(f"Duplicate function/PrimFunction name {function.name!r}.", stage=module.stage)
        prim_names.add(function.name)
        verify_prim_function(function)
    from triton.flagmega.ir.tir import KernelDefinition
    for kernel in module.kernel_definitions:
        if not isinstance(kernel, KernelDefinition):
            raise IRVerificationError("kernel_definitions must contain KernelDefinition values.", stage=module.stage)
        if kernel.name in function_names or kernel.name in prim_names:
            raise IRVerificationError(f"Duplicate kernel name {kernel.name!r}.", stage=module.stage)
        prim_names.add(kernel.name)
        verify_prim_function(kernel)
    verify_execution_functions(module)

    external_signatures = module.metadata.get("function_signatures", {})
    for node in module.nodes:
        if node.op not in {"builtin.call", "tir.call"}:
            continue
        callee_name = str(node.attrs["callee"])
        callee = module.function_map.get(callee_name)
        if callee is not None:
            parameter_types = tuple(node_map[value].type for value in callee.parameters)
            output_types = tuple(node_map[value].type for value in callee.outputs)
        elif (prim_function := module.kernel_callable_map.get(callee_name)) is not None:
            parameter_types = prim_function.runtime_parameter_types
            return_type = prim_function.runtime_return_type
            output_types = return_type.fields if isinstance(return_type, TupleType) else (return_type,)
            dispatch = kernel_dispatch_of(prim_function)
            if dispatch is not None and (
                node.effect.kind.value != dispatch.effect_kind
                or node.effect.resource != dispatch.effect_resource
            ):
                raise IRVerificationError(
                    f"Call {node.id!r} effect does not match KernelDispatch @{callee_name}.",
                    stage=module.stage,
                    node_id=node.id,
                )
            if dispatch is not None:
                selection = module.selection_map.get(f"tir.{node.id}")
                if (
                    selection is not None
                    and selection.candidate_id != dispatch.semantic_candidate
                ):
                    raise IRVerificationError(
                        f"TIR selection {selection.candidate_id!r} does not match materialized "
                        f"KernelDispatch semantic candidate {dispatch.semantic_candidate!r} "
                        f"on node {node.id!r}; "
                        "edit or resume before lower-tir instead of changing only the final "
                        "selection record.",
                        stage=module.stage,
                        node_id=node.id,
                    )
        else:
            signature = (
                external_signatures.get(callee_name)
                if hasattr(external_signatures, "get")
                else None
            )
            if signature is None:
                raise IRVerificationError(
                    f"Call {node.id!r} references missing function {callee_name!r}.",
                    stage=module.stage,
                    node_id=node.id,
                )
            parameter_types = tuple(signature.get("parameters", ()))
            output_types = tuple(signature.get("outputs", ()))
        actual_types = tuple(node_map[value].type for value in node.inputs)
        if actual_types != parameter_types:
            raise IRVerificationError(
                f"Call {node.id!r} argument types do not match @{callee_name}: "
                f"expected {parameter_types!r}, got {actual_types!r}.",
                stage=module.stage,
                node_id=node.id,
            )
        expected_type = output_types[0] if len(output_types) == 1 else TupleType(output_types)
        if not output_types or node.type != expected_type:
            raise IRVerificationError(
                f"Call {node.id!r} result type does not match @{callee_name}.",
                stage=module.stage,
                node_id=node.id,
            )

    point_map = {point.id: point for point in module.selection_points}
    if len(point_map) != len(module.selection_points):
        raise IRVerificationError("Selection point ids must be unique.", stage=module.stage)
    for point in module.selection_points:
        candidate_ids = {candidate.id for candidate in point.candidates}
        if len(candidate_ids) != len(point.candidates):
            raise IRVerificationError(f"Selection point {point.id!r} has duplicate candidates.", stage=module.stage)
        if point.default_candidate not in candidate_ids:
            raise IRVerificationError(
                f"Selection point {point.id!r} default {point.default_candidate!r} is not a candidate.",
                stage=module.stage,
            )
        if point.owner is not None and point.owner not in node_map:
            raise IRVerificationError(
                f"Selection point {point.id!r} owner {point.owner!r} does not exist.",
                stage=module.stage,
            )
    records: set[str] = set()
    for record in module.selections:
        if record.point_id in records:
            raise IRVerificationError(f"Selection {record.point_id!r} is recorded more than once.", stage=module.stage)
        records.add(record.point_id)
        point = point_map.get(record.point_id)
        if point is None:
            raise IRVerificationError(
                f"Selection record references unknown point {record.point_id!r}.", stage=module.stage)
        if record.candidate_id not in {candidate.id for candidate in point.candidates}:
            raise IRVerificationError(
                f"Selection {record.candidate_id!r} is not valid for point {record.point_id!r}.",
                stage=module.stage,
            )
        if point.kind == "distribution" and point.owner is not None:
            materialized = node_map[point.owner].metadata.get("distributed_candidate")
            if materialized is not None and record.candidate_id != materialized:
                raise IRVerificationError(
                    f"Distribution selection {record.candidate_id!r} does not match the materialized "
                    f"candidate {materialized!r} on owner {point.owner!r}; edit the distributed Python IR "
                    "or re-run AutoDistributed instead of changing only its selection record.",
                    stage=module.stage,
                    node_id=point.owner,
                )
    _VERIFIED_MODULES[id(module)] = module
    return module


def _verify_constant_recipes(module: IRModule, assets) -> None:
    recipe_map = {recipe.id: recipe for recipe in module.constant_recipes}
    if len(recipe_map) != len(module.constant_recipes):
        raise IRVerificationError("Constant recipe ids must be unique.", stage=module.stage)
    asset_map = {
        (
            str(node.attrs["recipe"]) if node.op == "builtin.const_asset" else str(node.metadata["constant_recipe"]),
            str(node.attrs["output"]) if node.op == "builtin.const_asset" else str(node.metadata["constant_output"]),
        ): node
        for node in assets
    }
    if len(asset_map) != len(assets):
        raise IRVerificationError("Constant asset recipe/output bindings must be unique.", stage=module.stage)

    for recipe in module.constant_recipes:
        local: dict[str, object] = {}
        # A recipe is a closed graph and its op verification only needs the
        # recipe-local node table plus dialect/stage. ``replace(module, ...)``
        # recursively re-froze the complete multi-megabyte module metadata for
        # every recipe, turning verification into O(recipes * module metadata).
        recipe_view = IRModule(
            dialect=module.dialect,
            stage=module.stage,
            nodes=recipe.nodes,
            functions=(),
            entry=module.entry,
        )
        for node in recipe.nodes:
            if node.id in local:
                raise IRVerificationError(
                    f"Constant recipe {recipe.id!r} has duplicate node {node.id!r}.",
                    stage=module.stage,
                    node_id=node.id,
                )
            missing = set(node.inputs) - set(local)
            if missing:
                raise IRVerificationError(
                    f"Constant recipe {recipe.id!r} node {node.id!r} has non-topological inputs {sorted(missing)}.",
                    stage=module.stage,
                    node_id=node.id,
                )
            definition = get_definition(node.op)
            definition.verify(node, recipe_view)
            is_source = definition.constant_source and not node.inputs and node.effect.is_pure
            is_expression = (
                definition.const_evaluable
                and definition.deterministic
                and node.effect.is_pure
                and bool(node.inputs)
                and all(value in local for value in node.inputs)
            )
            if not is_source and not is_expression:
                raise IRVerificationError(
                    f"Constant recipe {recipe.id!r} contains non-constant op {node.op!r}.",
                    stage=module.stage,
                    node_id=node.id,
                )
            local[node.id] = node
        if len(set(recipe.outputs)) != len(recipe.outputs):
            raise IRVerificationError(
                f"Constant recipe {recipe.id!r} has duplicate outputs.", stage=module.stage)
        for output in recipe.outputs:
            if output not in local:
                raise IRVerificationError(
                    f"Constant recipe {recipe.id!r} output {output!r} does not exist.", stage=module.stage)
            asset = asset_map.get((recipe.id, output))
            if asset is None:
                raise IRVerificationError(
                    f"Constant recipe {recipe.id!r} output {output!r} has no main-graph asset.", stage=module.stage)
            recipe_output_type = recipe.node_map[output].type
            if (
                asset.type != recipe_output_type
                and logical_type(asset.type) != logical_type(recipe_output_type)
            ):
                raise IRVerificationError(
                    f"Constant asset {asset.id!r} type does not match recipe output {output!r}.",
                    stage=module.stage,
                    node_id=asset.id,
                )
    for (recipe_id, output), asset in asset_map.items():
        recipe = recipe_map.get(recipe_id)
        if recipe is None or output not in recipe.outputs:
            raise IRVerificationError(
                f"Constant asset {asset.id!r} references unknown recipe output {recipe_id!r}/{output!r}.",
                stage=module.stage,
                node_id=asset.id,
            )


def _verify_type(value, stage: str, node_id: str) -> None:
    if isinstance(value, InvalidType):
        raise IRVerificationError(
            f"IR contains invalid type: {value.reason}", stage=stage, node_id=node_id
        )
    if isinstance(value, (AnyType, NoneType)):
        return
    if isinstance(value, TensorType):
        for dimension in value.shape:
            if dimension.minimum is not None and dimension.minimum < 0:
                raise IRVerificationError(
                    f"Tensor dimension {dimension} may be negative.", stage=stage, node_id=node_id)
        if isinstance(value.dtype, VectorType) and any(lane <= 0 for lane in value.dtype.lanes):
            raise IRVerificationError("Vector type lanes must be positive.", stage=stage, node_id=node_id)
        if isinstance(value.dtype, PointerType) and value.rank != 0:
            raise IRVerificationError("Pointer tensors must be scalar ABI handles.", stage=stage, node_id=node_id)
        if isinstance(value.dtype, MaskVectorType) and value.dtype.itemsize <= 0:
            raise IRVerificationError("Mask vector type has no physical storage.", stage=stage, node_id=node_id)
        if value.layout.order and sorted(value.layout.order) != list(range(value.rank)):
            raise IRVerificationError("Tensor layout order must be a rank permutation.", stage=stage, node_id=node_id)
        if value.layout.strides and len(value.layout.strides) != value.rank:
            raise IRVerificationError("Tensor layout strides must match rank.", stage=stage, node_id=node_id)
        if any(lane <= 0 for lane in value.layout.vector_lanes):
            raise IRVerificationError("Tensor vector lanes must be positive.", stage=stage, node_id=node_id)
        return
    if isinstance(value, TupleType):
        for field in value.fields:
            _verify_type(field, stage, node_id)
        return
    if isinstance(value, CallableType):
        _verify_type(value.return_type, stage, node_id)
        for parameter in value.parameters:
            _verify_type(parameter, stage, node_id)
        return
    if isinstance(value, RefType):
        for _, field in value.fields:
            _verify_type(field, stage, node_id)
        return
    if isinstance(value, DistributedType):
        _verify_type(value.tensor, stage, node_id)
        if not is_distributable(value.tensor, value.axis_policies, value.placement):
            raise IRVerificationError(
                "Distributed policies are not legal for the tensor and placement.", stage=stage, node_id=node_id)
        if value.partial is not None and any(axis >= value.placement.rank for axis in value.partial.axes):
            raise IRVerificationError(
                "Distributed partial references an invalid placement axis.", stage=stage, node_id=node_id)
        return

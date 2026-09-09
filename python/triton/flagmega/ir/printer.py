# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Compact nncase-style diagnostic printers for FlagMega IR and TIR."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from math import prod
from typing import Any, Mapping

from triton.flagmega.ir.dim_expr import Dimension
from triton.flagmega.ir.distributed_type import (
    BlockCyclicSplit,
    ContiguousSplit,
    Placement,
    SBP,
    SBPBroadCast,
    SBPPartial,
    SBPSplit,
    SplitDistribution,
    SplitStage,
)
from triton.flagmega.ir.model import (
    AnyType,
    CallableType,
    DistributedType,
    IRModule,
    IRType,
    InvalidType,
    Node,
    NoneType,
    RefType,
    TensorLayout,
    TensorType,
    TupleType,
)
from triton.flagmega.ir.ops.core import get_definition
from triton.flagmega.ir.print_symbols import GraphPrintSymbols, function_node_scopes
from triton.flagmega.ir.print_weights import WeightPrintAnalysis
from triton.flagmega.ir.types import DType, MaskVectorType, PointerType, VectorType


_DTYPE_DISPLAY_NAMES = {
    DType.BOOL: "bool",
    DType.INT32: "i32",
    DType.INT64: "i64",
    DType.BFLOAT16: "bf16",
    DType.FLOAT32: "f32",
    DType.FLOAT8_E4M3FN: "F8_E4M3",
}


def text_source(module: IRModule, *, weight_analysis: WeightPrintAnalysis | None = None) -> str:
    """Return the compact derived view appropriate for the module dialect."""

    render = script_source if module.dialect in {"semantic_tir", "bufferized_tir"} else il_source
    return render(module, weight_analysis=weight_analysis)


def companion_suffix(module: IRModule) -> str:
    return ".script" if module.dialect in {"semantic_tir", "bufferized_tir"} else ".il"


def il_source(module: IRModule, *, weight_analysis: WeightPrintAnalysis | None = None) -> str:
    lines = [
        f"// FlagMega {module.dialect} stage={module.stage} semantic_hash={module.semantic_hash}",
    ]
    weight_analysis = weight_analysis or WeightPrintAnalysis.analyze(module)
    _append_prim_functions(lines, module)
    _append_execution_functions(lines, module)
    scopes = function_node_scopes(module)
    for function in module.functions:
        symbols = GraphPrintSymbols(scopes[function.name], _il_node, weight_analysis.values)
        parameters = ", ".join(
            f"%{node_id}: {_type_text(module.node_map[node_id].type)}"
            for node_id in function.parameters
        )
        parameter_types = ", ".join(
            _type_text(module.node_map[node_id].type)
            for node_id in function.parameters
        )
        outputs = ", ".join(_type_text(module.node_map[node_id].type) for node_id in function.outputs)
        lines.append(f"%{function.name} = fn({parameters}): // ({parameter_types}) -> ({outputs})")
        lines.append("{")
        _append_weights_section(lines, symbols, weight_analysis.recipes[function.name], tir=False)
        for node in symbols.body:
            lines.append(f"  {symbols.lhs(node)} = {_il_node(node, symbols.reference)}: // "
                         f"{_type_text(node.type)}{_effect_text(node)} name={node.id!r}")
        returned = ", ".join(symbols.reference(node_id) for node_id in function.outputs)
        lines.append(f"  return ({returned})")
        lines.append("}")
    if module.selection_points:
        lines.append("")
        lines.append("// selections")
        selected = module.selection_map
        for point in module.selection_points:
            record = selected.get(point.id)
            candidate = point.default_candidate if record is None else record.candidate_id
            origin = "unapplied" if record is None else record.origin
            lines.append(f"select {point.id} = {candidate} // kind={point.kind}, origin={origin}")
    lines.append("")
    return "\n".join(lines)


def script_source(module: IRModule, *, weight_analysis: WeightPrintAnalysis | None = None) -> str:
    lines = [
        f"// FlagMega {module.dialect} stage={module.stage} semantic_hash={module.semantic_hash}",
    ]
    weight_analysis = weight_analysis or WeightPrintAnalysis.analyze(module)
    _append_prim_functions(lines, module)
    _append_execution_functions(lines, module)
    scopes = function_node_scopes(module)
    for function in module.functions:
        symbols = GraphPrintSymbols(scopes[function.name], _script_node, weight_analysis.values)
        parameters = ", ".join(
            f"%{node_id}: {_script_type_text(module.node_map[node_id].type)}"
            for node_id in function.parameters
        )
        lines.append(f'T.PrimFunc("{function.name}", {parameters}).Body(')
        _append_weights_section(lines, symbols, weight_analysis.recipes[function.name], tir=True)
        for node in symbols.body:
            lines.append(
                f"  {symbols.lhs(node)} = {_script_node(node, symbols.reference)} // "
                f"{_script_type_text(node.type)}{_effect_text(node)} name={node.id!r}")
        returned = ", ".join(symbols.reference(node_id) for node_id in function.outputs)
        lines.append(f"  T.Return({returned})")
        lines.append(")")
    _append_buffer_plan(lines, module)
    if module.selection_points:
        lines.append("")
        lines.append("// selections")
        for record in module.selections:
            lines.append(
                f"T.Select({record.point_id!r}, {record.candidate_id!r}, "
                f"origin={record.origin!r}, policy={record.policy!r})")
    lines.append("")
    return "\n".join(lines)


def _append_buffer_plan(lines: list[str], module: IRModule) -> None:
    data = module.metadata.get("buffer_plan")
    if not isinstance(data, Mapping):
        return
    from triton.flagmega.ir.bufferization import (
        BUFFER_PLAN_SCHEMA,
        LEGACY_BUFFER_PLAN_SCHEMA,
        BufferPlan,
    )

    if data.get("schema") not in {BUFFER_PLAN_SCHEMA, LEGACY_BUFFER_PLAN_SCHEMA}:
        return
    plan = BufferPlan.from_data(data)
    lines.extend(("", f"// buffer allocation: {plan.optimization_level} ({plan.allocator})"))
    for record in plan.allocation_records:
        objectives = ", ".join(
            f"{name}={value} [{status}, bound={bound}]"
            for name, status, value, bound in record.objectives
        )
        lines.append(f"// @{record.function}/{record.memory_space}: {record.status}"
                     + (f"; {objectives}" if objectives else ""))
    lines.extend(("", "// physical buffers"))
    for physical in plan.physical_buffers:
        lifetime = (
            ""
            if physical.live_start is None
            else f", Lifetime: [{physical.live_start}, {physical.live_end}]"
        )
        owner = "" if physical.function is None else f", Function: @{physical.function}"
        lines.append(
            f"T.PhysicalBuffer({physical.id!r}, Location: {physical.memory_space}, "
            f"Start: {physical.start}, Size: {physical.size}, Alignment: {physical.alignment}"
            f"{owner}{lifetime}, Role: {physical.role})"
        )
    lines.extend(("", "// logical buffer views"))
    for buffer in plan.buffers:
        dimensions = ",".join(str(value) for value in buffer.shape)
        logical_type = (
            _script_type_text(buffer.distributed_type)
            if buffer.distributed_type is not None
            else f"{_data_type_text(buffer.dtype)}[{dimensions}]"
        )
        strides = ",".join(str(value) for value in buffer.strides)
        alias = (
            ""
            if buffer.alias is None
            else f", Alias: {buffer.alias.kind.value}(%{buffer.alias.source})"
        )
        owner = "" if buffer.function is None else f", Function: @{buffer.function}"
        distributed_storage = (
            ""
            if buffer.distributed_storage_kind.value == "compact_local"
            else (
                ", DistributedStorage: "
                f"{buffer.distributed_storage_kind.value}"
            )
        )
        distributed_backing = (
            ""
            if buffer.distributed_backing_type is None
            else (
                ", DistributedBacking: "
                f"{_script_type_text(buffer.distributed_backing_type)}"
            )
        )
        lines.append(
            f"T.Buffer({logical_type}, "
            f"MemSpan: T.MemSpan({buffer.mem_span.buffer.id!r}, "
            f"Start: {buffer.mem_span.start}, Size: {buffer.mem_span.size}), "
            f"Strides: [{strides}], Storage: {buffer.storage}, Alignment: {buffer.alignment}"
            f"{distributed_storage}{distributed_backing}{owner}{alias}, Role: {buffer.role}) // %{buffer.id}"
        )
    kernel_calls = tuple(
        value
        for function in plan.functions
        for value in function.kernel_calls
        if value.workspaces
    )
    if kernel_calls:
        lines.extend(("", "// selected-kernel workspace call ABI"))
        for call in kernel_calls:
            bindings = ", ".join(
                f"%{formal} -> %{actual}" for formal, actual in call.workspaces
            )
            lines.append(
                f"T.KernelCallWorkspace({call.call!r}, Caller: @{call.caller}, "
                f"Callee: @{call.callee}, Bindings: [{bindings}])"
            )


def _append_prim_functions(lines: list[str], module: IRModule) -> None:
    for kernel in module.kernel_definitions:
        parameters = ", ".join(
            f"%{value.name}: {_script_type_text(value.type)} [{value.role.value}]"
            for value in kernel.parameters
        )
        lines.extend(("", f'T.KernelDef("{kernel.name}", kind={kernel.module_kind!r}, {parameters}) {{'))
        _print_tir_stmt(lines, kernel.dispatch, 2)
        lines.append("}")
    if not module.prim_functions:
        return
    for function in module.prim_functions:
        parameters = ", ".join(
            f"%{value.name}: {_script_type_text(value.type)} "
            f"[{value.role.value}"
            f"{'' if value.memory_space is None else ', ' + value.memory_space}]"
            for value in function.parameters
        )
        lines.extend(("", f'T.PrimFunc("{function.name}", kind={function.module_kind!r}, {parameters}) {{'))
        _print_tir_stmt(lines, function.body, 2)
        bindings = ", ".join(
            f"{_tir_value_text(value.value)} -> %{value.storage}"
            for value in function.results.values
        )
        lines.append(f"  T.Return({bindings})")
        lines.append("}")


def _append_execution_functions(lines: list[str], module: IRModule) -> None:
    for function in module.execution_functions:
        parameters = ", ".join(f"%{value}" for value in function.parameters)
        results = ", ".join(f"%{value}" for value in function.results)
        lines.extend((
            "",
            f'T.ExecutionFunc("{function.name}", [{parameters}]) -> [{results}] {{',
        ))
        _print_tir_stmt(lines, function.body, 2)
        lines.append("}")


def _print_tir_stmt(lines: list[str], statement, indent: int) -> None:
    from triton.flagmega.ir.tir import (
        Block,
        Barrier,
        BufferStore,
        Evaluate,
        For,
        IfThenElse,
        KernelDispatch,
        Let,
        PipelineDrain,
        PipelineHandoff,
        PipelineStage,
        PrimFunctionCall,
        KernelInvoke,
        ProducerConsumerRegion,
        Sequential,
    )

    prefix = " " * indent
    if isinstance(statement, Sequential):
        for field in statement.fields:
            _print_tir_stmt(lines, field, indent)
        return
    if isinstance(statement, KernelDispatch):
        arguments = ", ".join(f"%{value}" for value in statement.arguments)
        outputs = ", ".join(f"%{value}" for value in statement.outputs)
        microkernel = (
            None
            if statement.microkernel is None
            else statement.microkernel.implementation
        )
        workspaces = ", ".join(
            f"%{value.name}: {_script_type_text(value.type)}@{value.memory_space}"
            for value in statement.workspaces
        )
        shared_workspaces = ", ".join(
            f"%{value.name}: {_script_type_text(value.type)}@shared"
            for value in statement.shared_workspace_buffers
        )
        alias_text = ""
        if statement.inplace_alias_candidates is not None:
            aliases = ", ".join(
                f"%{value.output} <- %{value.input}"
                for value in statement.inplace_alias_candidates
            )
            alias_text = f"InplaceAliases: [{aliases}], "
        lines.append(
            f"{prefix}T.Kernel(SemanticOp: {statement.semantic_op!r}, "
            f"SemanticCandidate: {statement.semantic_candidate!r}, "
            f"MicroKernel: {microkernel!r}, "
            f"Parameters: {_value_text(statement.resolved_parameters, script=True)}, Inputs: [{arguments}], "
            f"Workspaces: [{workspaces}], SharedWorkspaces: [{shared_workspaces}], "
            f"{alias_text}"
            f"Reads: {list(statement.reads)!r}, "
            f"Writes: {list(statement.writes)!r}) -> ({outputs})"
        )
        return
    if isinstance(statement, ProducerConsumerRegion):
        lines.append(f"{prefix}T.ProducerConsumerRegion {{")
        lines.append(f"{prefix}  produce:")
        _print_tir_stmt(lines, statement.produce_body, indent + 4)
        lines.append(f"{prefix}  consume:")
        _print_tir_stmt(lines, statement.consume_body, indent + 4)
        lines.append(f"{prefix}}}")
        return
    if isinstance(statement, (PrimFunctionCall, KernelInvoke)):
        def bindings(values):
            return ", ".join(
                f"%{value.formal} -> %{value.actual}" for value in values
            )

        shared = ", ".join(
            f"%{value.name}@{value.mem_span.absolute_start}:"
            f"{value.mem_span.absolute_end}"
            for value in statement.shared_workspace_buffers
        )
        pools = ", ".join(
            f"{value.memory_space}=%{value.allocation}"
            f"[{value.offset}:{value.offset + value.nbytes}]"
            for value in statement.memory_pools
        )
        lines.append(
            f"{prefix}T.{'Invoke' if isinstance(statement, KernelInvoke) else 'Call'}({statement.call_id!r}, @{statement.callee}, "
            f"Args: [{bindings(statement.arguments)}], "
            f"Results: [{bindings(statement.results)}], "
            f"Workspaces: [{bindings(statement.workspaces)}], "
            f"MemoryPools: [{pools}], "
            f"DependsOn: {list(statement.dependencies)!r}, "
            f"Shared: [{shared}], TransferSources: "
            f"{list(statement.transfer_sources)!r})"
        )
        return
    if isinstance(statement, Barrier):
        ranges = ", ".join(
            f"{value.storage}:{value.physical_id}[{value.offset}:"
            f"{value.offset + value.nbytes}]/{value.mode}"
            for value in statement.ranges
        )
        lines.append(
            f"{prefix}T.Barrier({statement.scope.value.title()}, "
            f"After: {list(statement.after)!r}, Before: "
            f"{statement.before!r}, Hazards: {list(statement.hazards)!r}, "
            f"Ranges: [{ranges}])"
        )
        return
    if isinstance(statement, PipelineStage):
        lines.append(f"{prefix}T.PipelineStage({statement.stage_id!r}) {{")
        _print_tir_stmt(lines, statement.operation, indent + 2)
        lines.append(f"{prefix}}}")
        return
    if isinstance(statement, PipelineDrain):
        lines.append(f"{prefix}T.PipelineDrain({statement.stage_id!r})")
        return
    if isinstance(statement, PipelineHandoff):
        lines.append(f"{prefix}T.PipelineHandoff({statement.handoff_id!r})")
        return
    if isinstance(statement, BufferStore):
        indices = ", ".join(str(value) for value in statement.indices)
        lines.append(f"{prefix}{statement.buffer.name}[{indices}] = {_tir_value_text(statement.value)}")
        return
    if isinstance(statement, Evaluate):
        lines.append(f"{prefix}T.Evaluate({_tir_value_text(statement.value)})")
        return
    if isinstance(statement, For):
        lines.append(
            f"{prefix}for {statement.loop_var} in T.Range({statement.domain.start}, "
            f"{statement.domain.stop}, {statement.domain.step}) [{statement.mode.value}, "
            f"{statement.partition.value}]:"
        )
        _print_tir_stmt(lines, statement.body, indent + 2)
        return
    if isinstance(statement, Let):
        lines.append(f"{prefix}with T.Let({statement.var.name} = {_tir_value_text(statement.value)}):")
        _print_tir_stmt(lines, statement.body, indent + 2)
        return
    if isinstance(statement, IfThenElse):
        lines.append(f"{prefix}if {_tir_value_text(statement.condition)}:")
        _print_tir_stmt(lines, statement.then_body, indent + 2)
        if statement.else_body.fields:
            lines.append(f"{prefix}else:")
            _print_tir_stmt(lines, statement.else_body, indent + 2)
        return
    if isinstance(statement, Block):
        reads = ", ".join(_tir_region_text(value) for value in statement.reads)
        writes = ", ".join(_tir_region_text(value) for value in statement.writes)
        lines.append(f"{prefix}with T.Block({statement.name!r}, reads=[{reads}], writes=[{writes}]):")
        _print_tir_stmt(lines, statement.init_body, indent + 2)
        _print_tir_stmt(lines, statement.body, indent + 2)
        return
    lines.append(f"{prefix}T.{type(statement).__name__}(...)")


def _tir_value_text(value) -> str:
    from triton.flagmega.ir.tir import Binary, Buffer, BufferLoad, BufferTuple, Immediate, ScalarVar, ValueRef

    if isinstance(value, Buffer):
        return (
            f"T.Buffer({value.name!r}, {_data_type_text(value.elem_type)}"
            f"[{','.join(str(item) for item in value.dimensions)}], "
            f"T.MemSpan({value.mem_span.buffer.id!r}, {value.mem_span.start}, {value.mem_span.size}))"
        )
    if isinstance(value, BufferLoad):
        return f"{value.buffer.name}[{', '.join(str(item) for item in value.indices)}]"
    if isinstance(value, BufferTuple):
        return "(" + ", ".join(_tir_value_text(item) for item in value.buffers) + ")"
    if isinstance(value, ScalarVar):
        return f"%{value.name}"
    if isinstance(value, ValueRef):
        return f"%{value.name}"
    if isinstance(value, Immediate):
        return repr(value.value)
    if isinstance(value, Binary):
        return f"({_tir_value_text(value.lhs)} {value.op} {_tir_value_text(value.rhs)})"
    return str(value)


def _tir_region_text(value) -> str:
    ranges = ", ".join(f"{item.start}:{item.stop}:{item.step}" for item in value.region)
    return f"{value.buffer.name}[{ranges}]"


def _leaf_text(node: Node, *, script: bool = False) -> str | None:
    if node.op not in {"builtin.weight", "builtin.const_asset", "builtin.scalar_const",
                       "builtin.splat_const", "tir.scalar_const"}:
        return None
    prefix = "T." if script else ""
    type_text = _script_type_text(node.type) if script else _type_text(node.type)
    if node.op == "builtin.weight":
        fields = [type_text, repr(node.attrs["name"]), f"Source: {node.attrs['source']!r}"]
        if node.attrs["key"] != node.attrs["name"]:
            fields.append(f"Key: {node.attrs['key']!r}")
        if node.attrs.get("source_hash") is not None:
            fields.append(f"SourceHash: {node.attrs['source_hash']!r}")
        return f"{prefix}WeightRef({', '.join(fields)})"
    if node.op == "builtin.const_asset":
        return (f"{prefix}ConstAssetRef({type_text}, Recipe: {node.attrs['recipe']!r}, "
                f"Output: {node.attrs['output']!r})")
    if node.op in {"builtin.scalar_const", "builtin.splat_const", "tir.scalar_const"}:
        value = repr(node.attrs["value"])
        if node.op == "builtin.splat_const":
            value = f"splat({value})"
        return f"{'T.Const' if script else 'const'}({type_text} : {value})"
    return None


def _il_node(node: Node, reference: Callable[[str], str] | None = None) -> str:
    leaf = _leaf_text(node)
    if leaf is not None:
        return leaf
    inputs = ", ".join(reference(value) if reference else f"%{value}" for value in node.inputs)
    if node.op == "builtin.get_item":
        return f"GetItem({inputs}, {node.attrs['index']})"
    op_name = get_definition(node.op).display_name
    attributes = _attributes(node.attrs)
    arguments = ", ".join(value for value in (attributes, inputs) if value)
    return f"{op_name}({arguments})"


def _script_node(node: Node, reference: Callable[[str], str] | None = None) -> str:
    leaf = _leaf_text(node, script=True)
    if leaf is not None:
        return leaf
    inputs = ", ".join(reference(value) if reference else f"%{value}" for value in node.inputs)
    if node.op == "tir.buffer":
        return (
            f"T.Buffer({_script_type_text(node.type)}, Name: {node.id!r}, storage={node.attrs.get('storage')!r}, "
            f"alignment={node.attrs.get('alignment')}, key={node.attrs.get('key')!r})")
    if node.op == "tir.kernel":
        attributes = (
            f"SemanticOp: {node.attrs.get('semantic_op')!r}, "
            f"Candidate: {node.attrs.get('candidate')!r}, "
            f"Parameters: {_value_text(node.attrs.get('parameters', {}), script=True)}"
        )
        arguments = ", ".join(value for value in (attributes, inputs) if value)
        return f"T.Kernel({arguments})"
    if node.op == "builtin.call":
        return f"Call(@{node.attrs['callee']}, {inputs})"
    if node.op == "tir.call":
        return f"T.Call(@{node.attrs['callee']}, {inputs})"
    if node.op == "builtin.get_item":
        return f"T.GetItem({inputs}, {node.attrs['index']})"
    if node.op == "tir.barrier":
        return f"T.Barrier({_attributes(node.attrs, script=True)})"
    arguments = ", ".join(value for value in (_attributes(node.attrs, script=True), inputs) if value)
    return f"{get_definition(node.op).display_name}({arguments})"


def _append_weights_section(lines: list[str], symbols: GraphPrintSymbols, recipes, *, tir: bool) -> None:
    if not symbols.weights and not recipes:
        return
    lines.extend(("  weights {", "    // display grouping only; not an execution schedule"))
    _append_constant_recipes(lines, recipes, tir=tir, indent="    ")
    render = _script_node if tir else _il_node
    type_text = _script_type_text if tir else _type_text
    separator = " // " if tir else ": // "
    for node in symbols.weights:
        lines.append(f"    {symbols.lhs(node)} = {render(node, symbols.reference)}{separator}"
                     f"{type_text(node.type)}{_effect_text(node)} name={node.id!r}")
    lines.extend(("  }", "  // compute"))


def _append_constant_recipes(lines: list[str], recipes, *, tir: bool, indent: str = "") -> None:
    for recipe in recipes:
        render = _script_node if tir else _il_node
        symbols = GraphPrintSymbols(recipe.nodes, render)
        prefix = "T.ConstantRecipe" if tir else "constant_recipe"
        lines.append(f"{indent}{prefix} {recipe.id} // fingerprint={recipe.fingerprint}")
        lines.append(f"{indent}{{")
        for node in symbols.nodes:
            rendered = render(node, symbols.reference)
            type_text = _script_type_text(node.type) if tir else _type_text(node.type)
            lines.append(f"{indent}  {symbols.lhs(node)} = {rendered}: // {type_text}{_effect_text(node)} name={node.id!r}")
        lines.append(f"{indent}  yield (" + ", ".join(symbols.reference(value) for value in recipe.outputs) + ")")
        lines.append(f"{indent}}}")


def _type_text(value: IRType) -> str:
    if isinstance(value, AnyType):
        return "any"
    if isinstance(value, InvalidType):
        return f"invalid<{value.reason}>"
    if isinstance(value, NoneType):
        return "none"
    if isinstance(value, TensorType):
        return _tensor_type_text(value)
    if isinstance(value, TupleType):
        suffix = "..." if value.is_variadic else ""
        return "(" + ", ".join(_type_text(item) for item in value.fields) + suffix + ")"
    if isinstance(value, CallableType):
        parameters = ", ".join(_type_text(item) for item in value.parameters)
        return f"fn({parameters}) -> {_type_text(value.return_type)}"
    if isinstance(value, RefType):
        fields = ", ".join(f"{name}: {_type_text(field)}" for name, field in value.fields)
        return f"&{value.name}{{{fields}}}"
    if isinstance(value, DistributedType):
        return _distributed_result_type_text(value, wrapper="{")
    return type(value).__name__


def _script_type_text(value: IRType) -> str:
    if isinstance(value, DistributedType):
        return _distributed_result_type_text(value, wrapper="Dist(")
    if isinstance(value, TupleType):
        suffix = "..." if value.is_variadic else ""
        return "(" + ", ".join(_script_type_text(item) for item in value.fields) + suffix + ")"
    if isinstance(value, CallableType):
        parameters = ", ".join(_script_type_text(item) for item in value.parameters)
        return f"fn({parameters}) -> {_script_type_text(value.return_type)}"
    if isinstance(value, RefType):
        fields = ", ".join(f"{name}: {_script_type_text(field)}" for name, field in value.fields)
        return f"&{value.name}{{{fields}}}"
    return _type_text(value)


def _tensor_type_text(value: TensorType) -> str:
    dimensions = ",".join(str(item) for item in value.shape)
    return f"{_data_type_text(value.dtype)}[{dimensions}]{_layout_suffix(value.layout)}"


def _data_type_text(value) -> str:
    if isinstance(value, VectorType):
        lanes = ",".join(str(lane) for lane in value.lanes)
        return f"{_DTYPE_DISPLAY_NAMES[value.elem_type]}<{lanes}>"
    if isinstance(value, PointerType):
        return f"*{_data_type_text(value.elem_type)}"
    if isinstance(value, MaskVectorType):
        return f"mask<{value.style.value},{value.element_bits},{value.lanes}>"
    return _DTYPE_DISPLAY_NAMES[value]


def _layout_suffix(value: TensorLayout) -> str:
    if value == TensorLayout():
        return ""
    properties = [f"layout={value.tag}"]
    if value.order:
        properties.append("order=[" + ",".join(str(axis) for axis in value.order) + "]")
    if value.strides:
        properties.append(
            "strides=["
            + ",".join("?" if stride is None else str(stride) for stride in value.strides)
            + "]"
        )
    if value.vector_lanes:
        properties.append("vector_lanes=[" + ",".join(str(lane) for lane in value.vector_lanes) + "]")
    return "{" + ", ".join(properties) + "}"


def _distributed_definition_text(value: DistributedType, *, script: bool = False) -> str:
    tensor = _tensor_type_text(value.tensor)
    policies = ",".join(_sbp_text(policy) for policy in value.axis_policies)
    partial = "" if value.partial is None else _sbp_text(value.partial)
    body = f"{tensor}, ({policies}), {value.placement}, Partial: {partial}"
    return f"Dist({body})" if script else body


def _distributed_result_type_text(value: DistributedType, *, wrapper: str) -> str:
    tensor = _tensor_type_text(value.tensor)
    policies = ",".join(_sbp_text(policy) for policy in value.axis_policies)
    local_shape = ",".join(_local_dimension_text(value, axis) for axis in range(value.tensor.rank))
    partial = "" if value.partial is None else _sbp_text(value.partial)
    closing = "}" if wrapper == "{" else ")"
    return f"{wrapper}{tensor}, ({policies}), [{local_shape}], {partial}{closing}"


def _local_dimension_text(value: DistributedType, tensor_axis: int) -> str:
    dimension = value.tensor.shape[tensor_axis]
    policy = value.axis_policies[tensor_axis]
    if not isinstance(policy, SBPSplit):
        return str(dimension)

    divisor = prod(value.placement.hierarchy[axis] for axis in policy.hierarchy_axes)
    if dimension.is_fixed:
        quotient, remainder = divmod(dimension.fixed_value, divisor)
        local = str(quotient) if remainder == 0 else f"⌈{dimension}/{divisor}⌉"
    else:
        local = f"ceil_div({dimension},{divisor})"
    owners = "".join(f"@{value.placement.name[axis]}" for axis in policy.hierarchy_axes)
    return local + owners


def _sbp_text(value: SBP) -> str:
    if isinstance(value, SBPBroadCast):
        return "B"
    if isinstance(value, SBPPartial):
        axes = ",".join(str(axis) for axis in value.axes)
        return f"P([{axes}], {value.reduce_op.value.title()})"
    if isinstance(value, SBPSplit):
        return "S(" + ", ".join(_split_stage_text(stage) for stage in value.stages) + ")"
    raise TypeError(f"Unsupported SBP value {type(value).__name__}.")


def _split_stage_text(value: SplitStage) -> str:
    axes = ",".join(str(axis) for axis in value.hierarchy_axes)
    return f"{_split_distribution_text(value.distribution)}@[{axes}]"


def _split_distribution_text(value: SplitDistribution) -> str:
    if isinstance(value, ContiguousSplit):
        return "C" if value.granularity is None else f"C({value.granularity})"
    if isinstance(value, BlockCyclicSplit):
        return f"BC({value.block_size})"
    raise TypeError(f"Unsupported split distribution {type(value).__name__}.")


def _attributes(values: Mapping[str, Any], *, script: bool = False) -> str:
    return ", ".join(
        f"{_title(key)}: {_value_text(value, script=script)}"
        for key, value in sorted(values.items())
    )


def _title(value: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in value.split("_"))


def _value_text(value: Any, *, script: bool = False) -> str:
    if isinstance(value, DistributedType):
        return _distributed_definition_text(value, script=script)
    if isinstance(value, IRType):
        return _script_type_text(value) if script else _type_text(value)
    if isinstance(value, (DType, VectorType, PointerType, MaskVectorType)):
        return _data_type_text(value)
    if isinstance(value, Dimension):
        return str(value)
    if isinstance(value, SBP):
        return _sbp_text(value)
    if isinstance(value, SplitStage):
        return _split_stage_text(value)
    if isinstance(value, SplitDistribution):
        return _split_distribution_text(value)
    if isinstance(value, Placement):
        return str(value)
    if isinstance(value, TensorLayout):
        suffix = _layout_suffix(value)
        return "dense" if not suffix else suffix[1:-1]
    if isinstance(value, Mapping):
        return (
            "{"
            + ", ".join(
                f"{key}: {_value_text(item, script=script)}"
                for key, item in sorted(value.items())
            )
            + "}"
        )
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_value_text(item, script=script) for item in value) + "]"
    if isinstance(value, Enum):
        return value.value
    return repr(value)


def _effect_text(node: Node) -> str:
    if node.effect.is_pure:
        return ""
    return f" !{node.effect.kind.value}<{node.effect.resource}>"

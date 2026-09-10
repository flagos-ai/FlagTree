# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Typed, closed unary IR functions used at operator boundaries.

A Fusion contains ordinary nodes, not source text or an evaluator closure.
The Python decorator is only a convenient frontend to this immutable IR.
Binary expressions can use the parameter more than once and local constants;
external operands must remain explicit operands of the enclosing operator.
"""

from dataclasses import dataclass, replace
from functools import cached_property
from typing import Callable

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.model import Function, IRModule, IRType, Node, logical_type


@dataclass(frozen=True)
class Fusion:
    name: str
    nodes: tuple[Node, ...]
    output: str

    @cached_property
    def fingerprint(self):
        import hashlib
        import json
        return hashlib.sha256(json.dumps(self.to_data(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def __hash__(self):
        # Nodes contain immutable maps, not Python-hashable dicts. Candidate
        # inference caches and e-graph keys need the complete function body.
        return hash(self.fingerprint)

    def __post_init__(self):
        object.__setattr__(self, "nodes", tuple(self.nodes))
        if not self.name or not self.nodes or self.nodes[0].op != "builtin.var":
            raise IRSchemaError("Fusion requires a name and one leading parameter.")
        available = set()
        from triton.flagmega.ir.ops.core import get_definition
        for index, node in enumerate(self.nodes):
            if node.id in available or not set(node.inputs) <= available:
                raise IRSchemaError(f"Fusion {self.name!r} has duplicate or free/non-topological values.")
            if index and (node.op == "builtin.var" or not (get_definition(node.op).const_evaluable or node.op
                                                           in {"builtin.scalar_const", "builtin.splat_const"})):
                raise IRSchemaError("Fusion bodies require deterministic, evaluable expressions without captures.")
            if not node.effect.is_pure or (index and node.op.startswith("distributed.")):
                raise IRSchemaError("Fusion cannot contain effects or distributed communication.")
            if not get_definition(node.op).deterministic:
                raise IRSchemaError("Fusion expressions must be deterministic.")
            available.add(node.id)
        if self.output not in available:
            raise IRSchemaError("Fusion output is unknown.")
        from triton.flagmega.ir.verify import verify_module
        verify_module(self.as_module())

    @property
    def parameter(self):
        return self.nodes[0]

    @property
    def input_type(self):
        return self.parameter.type

    @property
    def output_type(self):
        return next(node.type for node in self.nodes if node.id == self.output)

    def as_module(self):
        return IRModule(dialect="high_level", stage="imported", nodes=self.nodes,
                        functions=(Function(self.name, (self.parameter.id, ), (self.output, )), ), entry=self.name)

    def specialize(self, input_type: IRType) -> tuple[Node, ...]:
        """Re-infer a body on another ownership of the same logical tensor."""
        if input_type == self.input_type:
            return self.nodes
        if logical_type(input_type) != logical_type(self.input_type):
            raise IRSchemaError(f"Fusion {self.name!r} input type disagrees with its parameter.")
        from triton.flagmega.ir.ops.core import get_definition
        values = {self.parameter.id: replace(self.parameter, type=input_type)}
        for node in self.nodes[1:]:
            if node.op in {"builtin.splat_const", "builtin.scalar_const"}:
                # Literal result_type is constructor-only, not a stored attr.
                # A uniform tensor can be generated on each owner's shard;
                # use a same-shaped value's inferred ownership (including
                # any vector-layout changes before the literal).
                value_type = logical_type(node.type)
                if node.op == "builtin.splat_const":
                    for anchor in reversed(tuple(values.values())):
                        if logical_type(anchor.type) == value_type:
                            value_type = anchor.type
                            break
                values[node.id] = replace(node, type=value_type)
                continue
            operands = tuple(values[value] for value in node.inputs)
            definition = get_definition(node.op)
            values[node.id] = replace(node, type=definition.infer_call_type(operands, node.attrs))
        return tuple(values.values())

    def infer_type(self, input_type: IRType) -> IRType:
        return next(node.type for node in self.specialize(input_type) if node.id == self.output)

    def evaluate(self, value, context, input_type=None):
        from triton.flagmega.evaluator.context import EvaluationContext
        from triton.flagmega.ir.ops.core import get_definition
        nodes = self.specialize(input_type or self.input_type)
        module = replace(self.as_module(), nodes=nodes)
        local_context = EvaluationContext(module, torch=context.torch, inputs={}, weights=context.weights,
                                          constant_assets={})
        local_context.dimension_bindings.update(context.dimension_bindings)
        values = {self.parameter.id: value}
        for node in nodes[1:]:
            arguments = tuple(values[key] for key in node.inputs)
            with local_context.call_scope(node, arguments, values.__getitem__):
                values[node.id] = get_definition(node.op).evaluate(node, arguments, local_context)
        return values[self.output]

    def then(self, following: "Fusion") -> "Fusion":
        """Compose ordinary SSA bodies, without erasing numeric operations."""
        following_nodes = following.specialize(self.output_type)
        ids = {following.parameter.id: self.output}
        nodes = list(self.nodes)
        occupied = {node.id for node in nodes}
        for node in following_nodes[1:]:
            new_id = f"composed_{len(nodes)}"
            while new_id in occupied:
                new_id += "_"
            ids[node.id] = new_id
            occupied.add(new_id)
            nodes.append(replace(node, id=new_id, inputs=tuple(ids[value] for value in node.inputs)))
        return Fusion(self.name, tuple(nodes), ids[following.output])

    def rewrite(self, rules) -> "Fusion":
        """Use the ordinary dataflow rules inside this function's SSA scope."""
        from triton.flagmega.rules import DataflowRewriter
        result = DataflowRewriter(rules).rewrite(self.as_module())
        return Fusion(self.name, result.nodes, result.functions[0].outputs[0])

    @classmethod
    def from_call(cls, node: Node, operand: Node):
        if node.inputs != (operand.id, ):
            raise IRSchemaError("Fusion.from_call requires one explicit unary operand.")
        from triton.flagmega.ir.op_fusion import has_ops, semantic_inputs, split_ops
        from triton.flagmega.ir.ops.core import get_definition
        if has_ops(node.attrs):
            definition = get_definition(node.op)
            semantic_operand, = semantic_inputs(definition, (operand, ), node.attrs)
            bare = replace(node, attrs=split_ops(node.attrs), type=definition.infer_call_type((semantic_operand, ),
                                                                                              split_ops(node.attrs)))
            result = cls.from_call(bare, semantic_operand)
            pre = node.attrs.get("pre_ops", {}).get(definition.input_parameters[0].name)
            if pre is not None:
                result = pre.then(result)
            post = node.attrs.get("post_ops", ())
            if post and post[0] is not None:
                result = result.then(post[0])
            return result
        parameter = Node("value", "builtin.var", (), operand.type, attrs={"name": "value"})
        body = replace(node, id="result", inputs=(parameter.id, ), metadata={})
        return cls(node.op.replace(".", "_"), (parameter, body), body.id)

    def to_data(self):
        return {"name": self.name, "nodes": [node.to_data() for node in self.nodes], "output": self.output}

    @classmethod
    def from_data(cls, data):
        return cls(str(data["name"]), tuple(Node.from_data(node) for node in data["nodes"]), str(data["output"]))


def fusion(input_type: IRType, *, name: str | None = None, parameter: str = "value", parameter_name: str | None = None,
           parameter_metadata=None):
    """Trace ``@fm.fusion(type) def f(x): return F.*(x)`` into ordinary IR."""

    def decorate(function: Callable[[Node], Node]) -> Fusion:
        from triton.flagmega.ir.builder import IRBuilder
        from triton.flagmega.ir.ops.core import construction_scope
        builder = IRBuilder(dialect="high_level", stage="imported")
        value = builder.var(parameter_name if parameter_name is not None else parameter, input_type, id=parameter,
                            metadata=parameter_metadata)
        with construction_scope(builder):
            output = function(value)
        if not isinstance(output, Node):
            raise IRSchemaError("Fusion Python function must return an IR expression.")
        return Fusion(name or function.__name__, builder.nodes, output.id)

    return decorate


def iter_fusions(value):
    """Find embedded semantic functions, including TIR dispatch attributes."""
    from collections.abc import Mapping
    from dataclasses import fields, is_dataclass
    if isinstance(value, Fusion):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from iter_fusions(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from iter_fusions(item)
    elif is_dataclass(value) and not isinstance(value, (IRType, type)):
        for field in fields(value):
            yield from iter_fusions(getattr(value, field.name))


__all__ = ["Fusion", "fusion", "iter_fusions"]

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Common register-expression emitter and explicit kernel fusion contracts."""

from collections.abc import Mapping

from triton.flagmega.errors import CodegenError, IRSchemaError
from triton.flagmega.ir import DistributedType, Node, VectorType
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.fusion import Fusion, iter_fusions
from triton.flagmega.ir.op_fusion import has_ops, semantic_inputs, split_ops
from triton.flagmega.ir.ops.core import get_definition

POINTWISE_OPS = frozenset({
    "math.add", "math.mul", "math.div", "math.sigmoid", "math.silu", "math.vectorized_binary", "math.vectorized_unary",
    "tensors.cast", "ntt.vectorized_cast"
})
FUSION_FAMILIES = {**{op: "elementwise" for op in POINTWISE_OPS}, "nn.softmax": "softmax"}
_EXPRESSIONS = POINTWISE_OPS | {"builtin.splat_const", "builtin.scalar_const"}


def decode_fusion_attrs(attrs):

    def decode(value):
        if isinstance(value, Mapping):
            if set(value) == {"$fusion"}:
                return Fusion.from_data(value["$fusion"])
            return {key: decode(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return tuple(decode(item) for item in value)
        return value

    return decode(attrs)


def vector_axes(attrs, rank=None):
    from triton.flagmega.ir.ops.tensors.pack import normalize_axes

    def normalized(value):
        return tuple(value) if rank is None else normalize_axes(tuple(value), rank)

    axes = set()
    if "vectorize_axes" in attrs:
        axes.add(normalized(attrs["vectorize_axes"]))
    for body in iter_fusions(attrs):
        for node in body.nodes[1:]:
            if node.op == "ntt.vectorized_cast":
                axes.add(normalized(node.attrs["vectorize_axes"]))
    if len(axes) > 1:
        raise CodegenError("Fusion vectorized casts require one coherent logical axis mapping.")
    return next(iter(axes), ())


def require_fusion(node, inputs, *, family=None):
    if not has_ops(node.attrs):
        return
    expected = FUSION_FAMILIES.get(node.op)
    if expected is None or (family is not None and family != expected):
        raise CodegenError(f"No PreOps/PostOps boundary emitter for {node.op} / {family}.")
    types = [value.type for value in inputs] + [node.type]
    for value_type in types:
        if isinstance(value_type, DistributedType) and value_type.partial is not None:
            raise CodegenError("PreOps/PostOps require materialized values, not partial reductions.")
    for body in iter_fusions(node.attrs):
        for value in body.nodes[1:]:
            if value.op not in _EXPRESSIONS or has_ops(value.attrs):
                raise CodegenError(f"Fusion expression {value.op!r} has no register emitter.")
        # Softmax has a scalar-axis load/store contract. Typed-vector repacking
        # is supported by elementwise's coordinate mapper, not by this kernel.
        if expected == "softmax" and any(isinstance(tensor_of(value.type).dtype, VectorType) for value in body.nodes):
            raise CodegenError("Softmax fusion requires scalar element types.")
    vector_axes(node.attrs, tensor_of(node.type).rank)
    definition = get_definition(node.op)
    inferred = definition.infer_call_type(inputs, node.attrs)
    if inferred != node.type:
        raise CodegenError("Fused call type disagrees with its semantic boundaries.")


def can_fuse(node, module):
    try:
        require_fusion(node, tuple(module.node_map[value] for value in node.inputs))
    except (CodegenError, IRSchemaError, ValueError):
        return False
    return True


def fusion_rules():
    from triton.flagmega.rules.neutral.pre_post_ops import pre_post_ops_rules
    return pre_post_ops_rules(FUSION_FAMILIES, can_fuse)


def scalar_type(value_type):
    dtype = tensor_of(value_type).dtype
    return dtype.elem_type if isinstance(dtype, VectorType) else dtype


def triton_type(value_type):
    from triton.flagmega.codegen.triton.kernel_call_renderers import _triton_dtype
    return _triton_dtype(scalar_type(value_type).value)


def expression(node, arguments):
    op = node.op
    if op == "math.vectorized_binary":
        op = f"math.{node.attrs['binary_op']}"
    elif op == "math.vectorized_unary":
        op = f"math.{node.attrs['unary_op']}"
    dtype = triton_type(node.type)
    if op in {"tensors.cast", "ntt.vectorized_cast"}:
        return f"({arguments[0]}).to({dtype})"
    if op in {"builtin.splat_const", "builtin.scalar_const"}:
        return f"tl.full((), {node.attrs['value']!r}, {dtype})"
    value = f"({arguments[0]}).to(tl.float32)"
    if op in {"math.add", "math.mul"}:
        operator = "+" if op == "math.add" else "*"
        result = f"({arguments[0]} {operator} {arguments[1]})"
    elif op == "math.div":
        result = f"tl.div_rn({value}, ({arguments[1]}).to(tl.float32))"
    elif op == "math.sigmoid":
        result = f"tl.sigmoid({value})"
    elif op == "math.silu":
        result = f"({value} * tl.sigmoid({value}))"
    else:
        raise CodegenError(f"No Fusion emitter for {node.op}.")
    return f"({result}).to({dtype})"


def emit_body(body, source, prefix):
    if body is None:
        return (), source
    values = {body.parameter.id: source}
    lines = []
    for index, node in enumerate(body.nodes[1:]):
        name = f"{prefix}_{index}"
        lines.append(f"{name} = {expression(node, tuple(values[value] for value in node.inputs))}")
        values[node.id] = name
    return tuple(lines), values[body.output]


def raw_semantics(raw):
    from triton.flagmega.codegen.triton.tensor_transform_renderers import _tensor_type
    from triton.flagmega.codegen.triton.kernel_call_renderers import _buffer
    attrs = decode_fusion_attrs(raw["semantic_attrs"])
    definition = get_definition(raw["semantic_op"])
    inputs = tuple(
        Node(f"arg{index}", "builtin.var", (), _tensor_type(_buffer(raw, "inputs", parameter.name)["abi"]))
        for index, parameter in enumerate(definition.input_parameters))
    operands = semantic_inputs(definition, inputs, attrs)
    result_type = definition.infer_call_type(operands, split_ops(attrs))
    return attrs, Node("base", definition.op_name, tuple(value.id for value in operands), result_type,
                       attrs=split_ops(attrs)), inputs


def elementwise_program(raw):
    attrs, base, _ = raw_semantics(raw)
    definition = get_definition(base.op)
    lines = []
    values = []
    for index, parameter in enumerate(definition.input_parameters):
        code, value = emit_body(
            attrs.get("pre_ops", {}).get(parameter.name), "lhs_value" if index == 0 else "rhs_value",
            f"pre_{parameter.name}")
        lines.extend(code)
        values.append(value)
    lines.append(f"base_result = {expression(base, tuple(values))}")
    code, value = emit_body(next(iter(attrs.get("post_ops", ())), None), "base_result", "post_result")
    return (*lines, *code), value


def softmax_program(raw):
    attrs, base, _ = raw_semantics(raw)
    pre_lines, pre_value = emit_body(attrs.get("pre_ops", {}).get("value"), "_fm_loaded", "pre_value")
    post_lines, post_value = emit_body(next(iter(attrs.get("post_ops", ())), None),
                                       f"_fm_probabilities.to({triton_type(base.type)})", "post_result")
    return {
        "fusion_pre_lines": pre_lines, "fusion_pre_value": pre_value, "fusion_post_lines": post_lines,
        "fusion_post_value": post_value
    }

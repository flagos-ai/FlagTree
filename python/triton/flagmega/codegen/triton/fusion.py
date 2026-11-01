# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Common register-expression emitter and explicit kernel fusion contracts."""

from collections.abc import Mapping
from dataclasses import replace

from triton.flagmega.errors import CodegenError, IRSchemaError
from triton.flagmega.ir import DistributedType, DType, Node, NoneType, TensorType, VectorType, type_from_data
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.fusion import Fusion, iter_fusions
from triton.flagmega.ir.op_fusion import has_ops, semantic_inputs, split_ops
from triton.flagmega.ir.ops.core import get_definition

POINTWISE_OPS = frozenset({
    "math.add", "math.mul", "math.div", "math.sigmoid", "math.silu", "math.vectorized_binary", "math.vectorized_unary",
    "tensors.cast", "ntt.vectorized_cast"
})
FUSION_FAMILIES = {**{op: "elementwise" for op in POINTWISE_OPS}, "nn.softmax": "softmax",
                   "tensors.broadcast_to": "broadcast_to", "math.reduce_sum": "reduce_sum",
                   "distributed.boxing": "distributed_boxing", "distributed.force_boxing": "distributed_boxing"}
_EXPRESSIONS = POINTWISE_OPS | {"builtin.splat_const", "builtin.scalar_const"}


def decode_fusion_attrs(attrs):

    def decode(value):
        if isinstance(value, Mapping):
            if set(value) == {"$fusion"}:
                return Fusion.from_data(value["$fusion"])
            if set(value) == {"$ir_type"}:
                return type_from_data(value["$ir_type"])
            return {key: type_from_data(item) if key == "new_type" and isinstance(item, Mapping)
                    and item.get("kind") in {"tensor", "distributed", "tuple", "ref", "none"}
                    else decode(item) for key, item in value.items()}
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
    if expected is None and _packed_accumulator_cast(node, inputs):
        # Dense kernels accumulate in FP32 and round at the final store.
        # Keep this explicit boundary separate from automatic generic fusion:
        # no arbitrary MatMul PreOps/PostOps emitter is implied here.
        expected = "dense_matmul"
    if expected is None or (family is not None and family != expected):
        raise CodegenError(f"No PreOps/PostOps boundary emitter for {node.op} / {family}.")
    if expected == "distributed_boxing":
        target = node.attrs["new_type"]
        if (node.attrs.get("pre_ops") or len(inputs) != 1 or inputs[0].type == target
                or any(not isinstance(kind, (TensorType, DistributedType))
                       or getattr(kind, "partial", None) is not None or getattr(kind, "exclusive", None) is not None
                       for kind in (inputs[0].type, target))):
            raise CodegenError("Numeric Boxing fusion requires a materialized tensor ownership copy with PostOps only.")
    types = [value.type for value in inputs] + [node.type]
    for value_type in types:
        if isinstance(value_type, DistributedType) and value_type.partial is not None:
            raise CodegenError("PreOps/PostOps require materialized values, not partial reductions.")
    for body in iter_fusions(node.attrs):
        if expected in {"broadcast_to", "reduce_sum", "distributed_boxing"} and not _coordinate_preserving_casts(body):
            raise CodegenError("Broadcast/reduction/Boxing fusion requires coordinate-preserving numeric casts.")
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


def _coordinate_preserving_casts(body):
    def geometry(kind):
        tensor = tensor_of(kind)
        dtype = VectorType(DType.FLOAT32, tensor.dtype.lanes) if isinstance(tensor.dtype, VectorType) else DType.FLOAT32
        tensor = replace(tensor, dtype=dtype)
        return replace(kind, tensor=tensor) if isinstance(kind, DistributedType) else tensor

    return all(value.op in {"tensors.cast", "ntt.vectorized_cast"}
               and geometry(value.type) == geometry(body.input_type) for value in body.nodes[1:])


def _packed_accumulator_cast(node, inputs):
    post = node.attrs.get("post_ops", ())
    if (node.op != "ntt.packed_matmul" or node.attrs.get("pre_ops") or len(post) != 1 or post[0] is None
            or node.attrs.get("fused_reduce", False) or DType(node.attrs.get("output_data_type")) is not DType.FLOAT32
            or len(inputs) != 4 or any(not isinstance(value.type, NoneType) for value in inputs[2:])):
        return False
    body = post[0]
    if (len(body.nodes) != 2 or body.nodes[-1].op not in {"tensors.cast", "ntt.vectorized_cast"}
            or body.nodes[-1].inputs != (body.parameter.id,)):
        return False
    if body.nodes[-1].op == "ntt.vectorized_cast":
        rank = tensor_of(body.input_type).rank
        if any(axis % rank != rank - 1 for axis in body.nodes[-1].attrs["vectorize_axes"]):
            return False

    def scalar(kind):
        dtype = tensor_of(kind).dtype
        return dtype.elem_type if isinstance(dtype, VectorType) else dtype

    return (scalar(inputs[0].type) is DType.BFLOAT16 and scalar(inputs[1].type) is DType.BFLOAT16
            and scalar(body.input_type) is DType.FLOAT32 and scalar(body.output_type) is DType.BFLOAT16)


def can_fuse(node, module):
    try:
        require_fusion(node, tuple(module.node_map[value] for value in node.inputs))
    except (CodegenError, IRSchemaError, ValueError):
        return False
    return True


def fusion_rules():
    from triton.flagmega.rules.neutral.pre_post_ops import pre_post_ops_rules
    from triton.flagmega.rules.neutral.commute_cast_view import commute_cast_view_rules
    from triton.flagmega.rules.neutral.post_ops_view import post_ops_through_view_rules
    return (*commute_cast_view_rules(), *pre_post_ops_rules(FUSION_FAMILIES, can_fuse),
            *post_ops_through_view_rules(("distributed.boxing", "distributed.force_boxing"), can_fuse))


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
        result = (f"({value} / (1.0 + libdevice.exp(-{value})))"
                  if node.attrs.get("math_mode") == "accurate" else f"({value} * tl.sigmoid({value}))")
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

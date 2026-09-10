# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Shared PreOps/PostOps semantics installed by the op-definition decorator."""

from dataclasses import replace
from functools import wraps

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.fusion import Fusion
from triton.flagmega.ir.model import Node, TensorType, DistributedType, TupleType


def has_ops(attrs):
    return bool(attrs.get("pre_ops") or attrs.get("post_ops"))


def split_ops(attrs):
    return {key: value for key, value in attrs.items() if key not in {"pre_ops", "post_ops"}}


def normalize_ops(definition, attrs):
    from triton.flagmega.ir.ops.core import ParameterInfo
    pre = {}
    parameters = {parameter.name: parameter for parameter in definition.input_parameters}
    for parameter, body in attrs.get("pre_ops", {}).items():
        if isinstance(parameter, ParameterInfo):
            if parameter.owner is not definition:
                raise IRSchemaError("PreOps ParameterInfo belongs to another operator.")
            parameter = parameter.name
        if parameter not in parameters or parameters[parameter].variadic:
            raise IRSchemaError(f"Invalid PreOps parameter {parameter!r} for {definition.op_name}.")
        if not isinstance(body, Fusion):
            raise IRSchemaError("PreOps must bind typed Fusion functions.")
        pre[parameter] = body
    post = attrs.get("post_ops", ())
    post = (post, ) if isinstance(post, Fusion) else tuple(post)
    if any(body is not None and not isinstance(body, Fusion) for body in post):
        raise IRSchemaError("PostOps must contain Fusion functions or None per result field.")
    return {**({"pre_ops": pre} if pre else {}), **({"post_ops": post} if any(post) else {})}


def semantic_inputs(definition, inputs, attrs):
    pre = attrs.get("pre_ops", {})
    result = list(inputs)
    for parameter in definition.input_parameters:
        body = pre.get(parameter.name)
        if body is not None:
            source = parameter.read(inputs)
            result[parameter.input_index] = replace(source, type=body.infer_type(source.type))
    return tuple(result)


def result_fields(result_type):
    return result_type.fields if isinstance(result_type, TupleType) else (result_type, )


def post_type(result_type, attrs):
    fields = result_fields(result_type)
    post = attrs.get("post_ops", ())
    if post and len(post) != len(fields):
        raise IRSchemaError("PostOps count must equal the number of result fields.")
    types = tuple(body.infer_type(value) if body is not None else value
                  for body, value in zip(post, fields)) if post else fields
    if any(not isinstance(value, (TensorType, DistributedType)) for value in (*fields, *types)):
        raise IRSchemaError("PreOps/PostOps require tensor results, not Ref/state/opaque values.")
    return TupleType(types) if isinstance(result_type, TupleType) else types[0]


def install_op_fusion(definition):
    """Leave the no-fusion path direct; wrap only common op entry points.

    Attributes are stripped before calling each op's normalizer/inference, so
    operators can continue to own strict, closed ParameterInfo schemas.
    """
    normalize = definition.normalize_attrs.__func__
    infer = definition.infer_type.__func__
    evaluate = definition.evaluate.__func__
    ir_attrs = definition.ir_attrs.__func__
    cost = definition.cost.__func__
    cost_factors = definition.cost_factors.__func__
    materialize_numpy = definition.materialize_numpy.__func__

    @wraps(normalize)
    def normalized(cls, attrs):
        if not has_ops(attrs):
            return normalize(cls, split_ops(attrs) if "pre_ops" in attrs or "post_ops" in attrs else attrs)
        return {**normalize(cls, split_ops(attrs)), **normalize_ops(cls, attrs)}

    @wraps(ir_attrs)
    def serialized(cls, attrs):
        if not has_ops(attrs):
            return ir_attrs(cls, attrs)
        return {**ir_attrs(cls, split_ops(attrs)), **normalize_ops(cls, attrs)}

    @wraps(infer)
    def inferred(cls, inputs, attrs):
        if not has_ops(attrs):
            return infer(cls, inputs, attrs)
        operands = semantic_inputs(cls, inputs, attrs)
        bare = split_ops(attrs)
        cls.verify_parameter_types(operands)
        if not cls.infer_effect(operands, bare).is_pure:
            raise IRSchemaError("PreOps/PostOps cannot cross an effectful operator boundary.")
        result = infer(cls, operands, bare)
        if any(
                isinstance(value, DistributedType) and value.partial is not None
                for value in (*[operand.type for operand in inputs], *result_fields(result))):
            raise IRSchemaError("PreOps/PostOps require materialized values, not partial reductions.")
        return post_type(result, attrs)

    @wraps(evaluate)
    def evaluated(cls, node, arguments, context):
        if not has_ops(node.attrs):
            return evaluate(cls, node, arguments, context)
        from triton.flagmega.evaluator.context import EvaluationContext
        inputs = tuple(
            Node(value, "builtin.var", (), context.types[value], attrs={"name": value}) for value in node.inputs)
        operands = semantic_inputs(cls, inputs, node.attrs)
        values = list(arguments)
        for parameter in cls.input_parameters:
            body = node.attrs.get("pre_ops", {}).get(parameter.name)
            if body is not None:
                values[parameter.input_index] = body.evaluate(parameter.read(arguments), context,
                                                              parameter.read(inputs).type)
        bare = replace(node, attrs=split_ops(node.attrs), type=infer(cls, operands, split_ops(node.attrs)))
        # A local evaluator context exposes the semantic operand/result types,
        # not the different storage types at the enclosing call boundary.
        local = EvaluationContext(replace(context.module,
                                          nodes=(*operands, bare)), torch=context.torch, inputs=context.inputs,
                                  weights=context.weights, constant_assets=context.constant_assets)
        local.dimension_bindings.update(context.dimension_bindings)
        with local.call_scope(bare, tuple(values), context.evaluate):
            result = evaluate(cls, bare, tuple(values), local)
        post = node.attrs.get("post_ops", ())
        fields = result if isinstance(bare.type, TupleType) else (result, )
        if post:
            fields = tuple(
                body.evaluate(value, context, value_type) if body is not None else value
                for body, value, value_type in zip(post, fields, result_fields(bare.type)))
        return fields if isinstance(bare.type, TupleType) else fields[0]

    @wraps(cost)
    def costed(cls, node):
        if not has_ops(node.attrs):
            return cost(cls, node)
        from triton.flagmega.ir.fusion import iter_fusions
        from triton.flagmega.ir.distributed_inference import tensor_of
        from triton.flagmega.ir.ops.core import get_definition, tensor_nbytes
        post = node.attrs.get("post_ops", ())
        fields = result_fields(node.type)
        base_fields = tuple(body.input_type if body is not None else value
                            for body, value in zip(post, fields)) if post else fields
        base_type = TupleType(base_fields) if isinstance(node.type, TupleType) else base_fields[0]
        metric = cost(cls, replace(node, type=base_type, attrs=split_ops(node.attrs)))
        factors = [metric.flops]
        for body in iter_fusions(node.attrs):
            for value in body.nodes[1:]:
                if value.op not in {"builtin.splat_const", "builtin.scalar_const"}:
                    factors.append(get_definition(value.op).cost(replace(value, type=tensor_of(value.type))).flops)
        written = [tensor_nbytes(tensor_of(value)) for value in fields]
        return replace(
            metric, flops=None if None in factors else sum(factors),
            # Mixed input storage types cannot be reconstructed from
            # this result-only cost API. Do not count hidden buffers.
            bytes_read=None if node.attrs.get("pre_ops") else metric.bytes_read,
            bytes_written=None if None in written else sum(written), notes=(*metric.notes, "pre-post-fusion"))

    @wraps(materialize_numpy)
    def materialized(cls, node, arguments, context):
        if has_ops(node.attrs):
            from triton.flagmega.errors import NumpyMaterializationUnsupported
            raise NumpyMaterializationUnsupported(
                "PreOps/PostOps need the semantic constant evaluator, not a byte-layout materializer.")
        return materialize_numpy(cls, node, arguments, context)

    @wraps(cost_factors)
    def factored(cls, inputs, attrs, return_type):
        if not has_ops(attrs):
            return cost_factors(cls, inputs, attrs, return_type)
        from triton.flagmega.ir.ops.core import get_definition, tensor_nbytes, _local_cost_type
        operands = semantic_inputs(cls, inputs, attrs)
        bare = split_ops(attrs)
        base_type = cls.infer_call_type(operands, bare)
        metric = cost_factors(cls, operands, bare, base_type)
        if metric is None:
            return None
        arithmetic = {
            key: getattr(metric, key)
            for key in ("cpu_cycles", "elementwise_operations", "simt_fma_operations")
        }
        bodies = [(attrs["pre_ops"][parameter.name], parameter.type_of(inputs))
                  for parameter in cls.input_parameters
                  if parameter.name in attrs.get("pre_ops", {})]
        bodies.extend((body, value_type)
                      for body, value_type in zip(attrs.get("post_ops", ()), result_fields(base_type))
                      if body is not None)
        for body, value_type in bodies:
            nodes = body.specialize(value_type)
            node_map = {node.id: node for node in nodes}
            for node in nodes[1:]:
                if node.op in {"builtin.scalar_const", "builtin.splat_const"}:
                    continue
                factors = get_definition(node.op).cost_factors(tuple(node_map[value] for value in node.inputs),
                                                               node.attrs, node.type)
                if factors is None:
                    return None
                for key in arithmetic:
                    arithmetic[key] += getattr(factors, key)
        # Register bodies have no independent materialized inputs/outputs.
        # The typed candidate API, unlike result-only cost(), can account for
        # the real storage at every external call boundary.
        reads = [tensor_nbytes(_local_cost_type(value.type)) for value in inputs]
        writes = [tensor_nbytes(_local_cost_type(value)) for value in result_fields(return_type)]
        if None in reads or None in writes:
            return None
        return replace(metric, **arithmetic, block_local_memory_load_bytes=sum(reads),
                       block_local_memory_store_bytes=sum(writes))

    definition.normalize_attrs = classmethod(normalized)
    definition.ir_attrs = classmethod(serialized)
    definition.infer_type = classmethod(inferred)
    definition.evaluate = classmethod(evaluated)
    definition.cost = classmethod(costed)
    definition.cost_factors = classmethod(factored)
    definition.materialize_numpy = classmethod(materialized)


__all__ = ["has_ops", "split_ops", "normalize_ops", "semantic_inputs", "install_op_fusion"]

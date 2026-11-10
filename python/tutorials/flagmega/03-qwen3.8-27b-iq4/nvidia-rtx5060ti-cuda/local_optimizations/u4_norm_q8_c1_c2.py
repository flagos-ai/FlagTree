"""A56 input NormApply + independent C1/C2 Q8 semantics. Physical support is owner supplied.

CPU evaluation delegates to authoritative NormApply and then the existing Q8
oracle. It does not establish native rsqrt/FMA/FTZ or full-model acceptance.
"""
from dataclasses import replace

from triton.flagmega.errors import EvaluationError, IRSchemaError
from triton.flagmega.ir import DType, TupleType, VectorType, tensor_type
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.core import (
    OpCost, PythonCall, NodeRef, attribute_parameter, input_parameter, op_definition,
)
from triton.flagmega.ir.ops.nn.norm_apply import NormApply
from triton.flagmega.ir.type_pattern import is_tensor
from .u4_ops import (
    _U4Base, _common, _complete, encoding_attrs, encode_q8_reference,
    register_q8_tuple_producer,
)

CONTRACT_ID = "a56-input-norm-common-c0-disjoint-c1-c2/v1"
NORM_ATTRS = dict(axis=-1, epsilon=9.999999974752427e-7, use_mean=False,
                  round_before_scale=False, output_dtype="bfloat16",
                  input_dtype=None, math_mode="fast")
HIDDEN_TYPE = tensor_type(VectorType(DType.BFLOAT16, (8,)), (1, 640))
STATS_TYPE = tensor_type(DType.FLOAT32, (1, 1, 1))
AFFINE_TYPE = tensor_type(VectorType(DType.FLOAT32, (8,)), (640,))
Q8_C1_TYPE = tensor_type(DType.UINT8, (1, 160, 36))
Q8_C2_TYPE = tensor_type(DType.UINT8, (2, 160, 36))


def norm_q8_attrs(*, source_epoch, hidden_epoch):
    return {**encoding_attrs(k=5120, components=2, source_epoch=source_epoch),
            **NORM_ATTRS, "hidden_epoch": hidden_epoch}


def validate_bound_stats(hidden, stats):
    """Validate semantic dependency, not merely same-shaped statistics."""
    if (stats.op != "nn.bind_norm_stats" or len(stats.inputs) != 2
            or stats.inputs[0] != hidden.id or stats.attrs.get("axis") != 1
            or stats.attrs.get("use_mean") is not False):
        raise IRSchemaError("A56 requires statistics explicitly bound to this exact hidden SSA.")


@op_definition("u4.norm_apply_q8_c1_c2")
class NormApplyQ8C1C2(_U4Base):
    hidden = input_parameter(is_tensor())
    stats = input_parameter(is_tensor())
    scale = input_parameter(is_tensor())
    bias = input_parameter(is_tensor())
    axis = attribute_parameter(default=-1)
    epsilon = attribute_parameter(default=NORM_ATTRS["epsilon"])
    use_mean = attribute_parameter(default=False)
    round_before_scale = attribute_parameter(default=False)
    output_dtype = attribute_parameter(default="bfloat16")
    input_dtype = attribute_parameter(default=None)
    math_mode = attribute_parameter(default="fast")
    k = attribute_parameter()
    components = attribute_parameter()
    source_epoch = attribute_parameter()
    hidden_epoch = attribute_parameter()
    format_id = attribute_parameter()
    numerical_id = attribute_parameter()
    encoding_id = attribute_parameter()
    q8_schema = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attrs):
        attrs = _common(super().normalize_attrs(attrs))
        if attrs["k"] != 5120 or attrs["components"] != 2:
            raise IRSchemaError("A56 NormApply Q8 is fixed K=5120/C1+C2.")
        for key, expected in NORM_ATTRS.items():
            if type(attrs[key]) is not type(expected) or attrs[key] != expected:
                raise IRSchemaError(f"A56 changed fixed NormApply {key}.")
        if not isinstance(attrs["hidden_epoch"], str) or not attrs["hidden_epoch"].strip():
            raise IRSchemaError("A56 requires the original hidden invocation epoch.")
        return attrs

    @classmethod
    def infer_type(cls, inputs, attrs):
        _complete(inputs)
        expected = (HIDDEN_TYPE, STATS_TYPE, AFFINE_TYPE, AFFINE_TYPE)
        if tuple(tensor_of(n.type) for n in inputs) != expected:
            raise IRSchemaError("A56 requires complete BF16x8 hidden, F32 stats, F32x8 scale/bias.")
        # The graph rewrite separately proves binding and every caller epoch;
        # physical inference may legitimately present canonical view operands.
        return TupleType((HIDDEN_TYPE, Q8_C1_TYPE, Q8_C2_TYPE))

    @classmethod
    def verify(cls, node, module):
        super().verify(node, module)
        if module.stage == "stats_combined":
            hidden, stats = (module.node_map[x] for x in node.inputs[:2])
            validate_bound_stats(hidden, stats)
            parts = node.attrs["source_epoch"].split(":invocation:")
            if (len(parts) != 2 or not all(parts)
                    or node.attrs["hidden_epoch"] != parts[0] + ":invocation:" + hidden.id):
                raise IRSchemaError("A56 fused source/hidden invocation epochs differ.")

    @classmethod
    def evaluate(cls, node, arguments, context):
        if any(value.device.type != "cpu" for value in arguments):
            raise EvaluationError("A56 diagnostic evaluator is CPU only.")
        import torch
        normalized = NormApply.evaluate(replace(node, type=node.type.fields[0]), arguments, context)
        c2 = encode_q8_reference(normalized.float().numpy(), 2)
        # One C0 computation, two independent owning arrays. No prefix alias.
        c1 = c2[:1].copy()
        return normalized, torch.from_numpy(c1), torch.from_numpy(c2)

    @classmethod
    def cost(cls, node):
        return OpCost(flops=None, bytes_read=51204, bytes_written=27520,
                      notes=("A56 shared C0 computation; two disjoint C0 stores plus residual C1",
                             "logical distinct bytes only; native traffic/cost unmeasured"))

    @classmethod
    def python_call(cls, node):
        return PythonCall(
            "__import__('local_optimizations.u4_norm_q8_c1_c2', fromlist=['NormApplyQ8C1C2']).NormApplyQ8C1C2.construct",
            tuple(NodeRef(x) for x in node.inputs),
            {**node.attrs, "name": node.id, "metadata": node.metadata})


register_q8_tuple_producer(NormApplyQ8C1C2.op_name, 1)
register_q8_tuple_producer(NormApplyQ8C1C2.op_name, 2)


def norm_apply_q8_c1_c2(hidden, stats, scale, bias, *, source_epoch, hidden_epoch, **kwargs):
    return NormApplyQ8C1C2.construct(hidden, stats, scale, bias,
                                  **norm_q8_attrs(source_epoch=source_epoch, hidden_epoch=hidden_epoch),
                                  **kwargs)

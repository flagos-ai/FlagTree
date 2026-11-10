"""A56 explicit immutable C2 dependency; physical implementation is owner supplied."""
from dataclasses import replace

from triton.flagmega.errors import EvaluationError, IRSchemaError
from triton.flagmega.ir import DType, DistributedType, tensor_type
from triton.flagmega.ir.distributed_inference import all_broadcast, tensor_of
from triton.flagmega.ir.ops.core import (
    NodeRef, PythonCall, attribute_parameter, input_parameter, op_definition,
)
from triton.flagmega.ir.ops.nn.gdn_recurrent_core import GatedDeltaNetRecurrentCore
from triton.flagmega.ir.memory_effect import MemoryEffect
from triton.flagmega.ir.type_pattern import is_tensor, has_rank, is_ref
from .u4_ops import Q8_CONTRACT_KEYS, _common, _q8, verify_q8_provenance, encode_q8_reference


@op_definition("u4.gdn_recurrent_core_q8_c2")
class GDNRecurrentCoreQ8C2(GatedDeltaNetRecurrentCore):
    """Preserve the nine original operands, then append one declared read-only C2."""
    state = input_parameter(is_ref(), memory_effect=MemoryEffect.for_fields(recurrent=MemoryEffect.READ_WRITE))
    qkv = input_parameter(is_tensor() & has_rank(2))
    z = input_parameter(is_tensor() & has_rank(2))
    projection_input = input_parameter(is_tensor() & has_rank(2))
    b_weight = input_parameter(is_tensor() & has_rank(2))
    a_weight = input_parameter(is_tensor() & has_rank(2))
    a_log = input_parameter(is_tensor())
    dt_bias = input_parameter(is_tensor())
    norm_weight = input_parameter(is_tensor())
    num_key_heads = attribute_parameter()
    num_value_heads = attribute_parameter()
    key_head_dim = attribute_parameter()
    value_head_dim = attribute_parameter()
    epsilon = attribute_parameter()
    qk_norm_mode = attribute_parameter(default="clamp")
    qk_norm_epsilon = attribute_parameter(default=1e-12)
    round_normalized_qk = attribute_parameter(default=False)
    round_beta = attribute_parameter(default=True)
    round_core = attribute_parameter(default=False)
    gate_math_mode = attribute_parameter(default="accurate")
    norm_math_mode = attribute_parameter(default="fast")
    b_weight_encoding = attribute_parameter(default=None)
    a_weight_encoding = attribute_parameter(default=None)
    value_head_order = attribute_parameter(default="grouped")

    q8_c2 = input_parameter(is_tensor() & has_rank(3))
    k = attribute_parameter()
    components = attribute_parameter()
    source_epoch = attribute_parameter()
    format_id = attribute_parameter()
    numerical_id = attribute_parameter()
    encoding_id = attribute_parameter()
    q8_schema = attribute_parameter()
    const_evaluable = False

    @classmethod
    def normalize_attrs(cls, attrs):
        original = super().normalize_attrs(attrs)
        encoding = _common({key: attrs[key] for key in Q8_CONTRACT_KEYS})
        if encoding["k"] != 5120 or encoding["components"] != 2:
            raise IRSchemaError("A56 GDN requires exact K5120/C2.")
        return {**original, **encoding}

    @classmethod
    def infer_type(cls, inputs, attrs):
        result = GatedDeltaNetRecurrentCore.infer_type(inputs[:9], attrs)
        encoded, projection = inputs[9], inputs[3]
        _q8(encoded, attrs)
        if tensor_of(projection.type) != tensor_type(DType.BFLOAT16, (1, 5120)):
            raise IRSchemaError("A56 GDN projection must retain complete rounded BF16[1,5120].")
        if isinstance(projection.type, DistributedType):
            if (not isinstance(encoded.type, DistributedType)
                    or encoded.type.placement != projection.type.placement
                    or not all_broadcast(encoded.type)):
                raise IRSchemaError("A56 GDN C2 must be complete on the actual projection placement.")
        elif isinstance(encoded.type, DistributedType):
            raise IRSchemaError("A56 GDN C2/projection placement mismatch.")
        return result

    @classmethod
    def verify(cls, node, module):
        super().verify(node, module)
        q8 = module.node_map[node.inputs[9]]
        verify_q8_provenance(module, q8, node.attrs)
        # Resolve real sources at logical and distributed stages alike.
        projection = module.node_map[node.inputs[3]]
        seen = set()
        while projection.op.startswith("distributed."):
            if projection.id in seen or len(projection.inputs) != 1:
                raise IRSchemaError("A56 projection must have one acyclic actual source.")
            seen.add(projection.id)
            projection = module.node_map[projection.inputs[0]]
        parts = node.attrs["source_epoch"].split(":invocation:")
        if len(parts) != 2 or not all(parts) or parts[1] != projection.id:
            raise IRSchemaError("A56 GDN C2 does not encode this exact projection SSA epoch.")
        current = q8
        while current.op.startswith("distributed."):
            current = module.node_map[current.inputs[0]]
        producer = module.node_map[current.inputs[0]]
        if (producer.op != "u4.norm_apply_q8_c1_c2"
                or current.attrs.get("index") != 2
                or projection.op != "tensors.bitcast"
                or projection.attrs != {"dtype": "bfloat16"}):
            raise IRSchemaError("A56 GDN must read the matching normalized/C2 tuple fields.")
        normalized = module.node_map[projection.inputs[0]]
        if (normalized.op != "builtin.get_item" or normalized.attrs != {"index": 0}
                or normalized.inputs != (producer.id,)):
            raise IRSchemaError("A56 GDN normalized and C2 values have different producers.")

    @classmethod
    def evaluate(cls, node, arguments, context):
        # The original CPU semantic oracle does not emulate native IQ4 DP4A.
        # Validate the extra argument against its exact BF16 source, then reuse
        # that oracle unchanged. Device equivalence remains an explicit gate.
        import torch
        projection, encoded = arguments[3], arguments[9]
        if projection.device.type != "cpu" or encoded.device.type != "cpu":
            raise EvaluationError("A56 diagnostic GDN evaluator is CPU only.")
        expected = torch.from_numpy(encode_q8_reference(projection.float().numpy(), 2))
        if encoded.dtype != torch.uint8 or not torch.equal(encoded, expected):
            raise EvaluationError("A56 GDN C2 payload differs from its exact rounded BF16 source.")
        original_attrs = {k: v for k, v in node.attrs.items() if k not in Q8_CONTRACT_KEYS}
        original = replace(node, op=GatedDeltaNetRecurrentCore.op_name,
                           inputs=node.inputs[:9], attrs=original_attrs)
        return GatedDeltaNetRecurrentCore.evaluate(original, arguments[:9], context)

    @classmethod
    def python_call(cls, node):
        return PythonCall(
            "__import__('local_optimizations.u4_gdn_q8_c2', fromlist=['GDNRecurrentCoreQ8C2']).GDNRecurrentCoreQ8C2.construct",
            tuple(NodeRef(x) for x in node.inputs),
            {**node.attrs, "name": node.id, "metadata": node.metadata})

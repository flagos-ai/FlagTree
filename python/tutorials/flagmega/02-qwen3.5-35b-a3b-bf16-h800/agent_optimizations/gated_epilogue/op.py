"""Rounding-explicit gated branch/residual merge with additive statistics."""

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import DType, DimConst, SBP, TupleType
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.core import (
    NodeRef, OpCost, OpDefinition, PythonCall, attribute_parameter, input_parameter, op_definition, tensor_nbytes,
)
from triton.flagmega.ir.ops.nn._norm import norm_stats_value
from triton.flagmega.ir.ops.nn._sparse_experts import (
    distributed_inputs, element_type, floating_tensor, lanes, pack_result, scale_policy, unpack_value,
)
from triton.flagmega.ir.ops.nn.norm_stats import NormStats


@op_definition("local.gated_residual_norm_stats", display_name="Local.GatedResidualNormStats")
class GatedResidualNormStats(OpDefinition):
    supports_broadcast_lifting = False
    routed = input_parameter(floating_tensor(2, packed=True))
    shared = input_parameter(floating_tensor(2, packed=True))
    gate_logit = input_parameter(floating_tensor(2))
    residual = input_parameter(floating_tensor(2, packed=True))
    use_mean = attribute_parameter(default=False)

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = super().normalize_attrs(attributes)
        if not isinstance(attrs["use_mean"], bool):
            raise IRSchemaError("GatedResidualNormStats use_mean must be boolean.")
        return attrs

    @classmethod
    def infer_type(cls, inputs, attrs):
        types = {p.name: p.type_of(inputs) for p in cls.input_parameters}
        tensors = {name: tensor_of(value) for name, value in types.items()}
        routed, shared, gate, residual = (tensors[p.name] for p in cls.input_parameters)
        if (element_type(routed.dtype) != DType.BFLOAT16 or element_type(shared.dtype) != DType.BFLOAT16
                or gate.dtype != DType.BFLOAT16 or element_type(residual.dtype) != DType.FLOAT32):
            raise IRSchemaError("GatedResidualNormStats requires BF16 branches/gate and an FP32 residual.")
        scalar_shape = (residual.shape[0], residual.shape[1] * lanes(residual.dtype))
        for value in (routed, shared):
            if (value.shape[0], value.shape[1] * lanes(value.dtype)) != scalar_shape:
                raise IRSchemaError("GatedResidualNormStats branch scalar shapes must match the residual.")
        if gate.shape[1] != DimConst(1) or gate.shape[0] not in (residual.shape[0], DimConst(1)):
            raise IRSchemaError("GatedResidualNormStats gate must be scalar or one value per token.")
        placement = distributed_inputs(types)
        if placement is not None:
            policies = types["residual"].axis_policies
            for name in ("routed", "shared"):
                source = types[name]
                scaled = (source.axis_policies[0],
                          scale_policy(source.axis_policies[1], lanes(tensors[name].dtype), lanes(residual.dtype)))
                if scaled != policies:
                    raise IRSchemaError("GatedResidualNormStats branches must share scalar local ownership.")
            gate_token = SBP.broadcast() if gate.shape[0].is_fixed and gate.shape[0].fixed_value == 1 else policies[0]
            if types["gate_logit"].axis_policies != (gate_token, SBP.broadcast()):
                raise IRSchemaError("GatedResidualNormStats gate must be available to each hidden owner.")
        stats = NormStats.prepare((cls.residual.read(inputs),), {"axis": 1, "use_mean": attrs["use_mean"]}).result_type
        return TupleType((types["residual"], stats))

    @classmethod
    def evaluate(cls, node, arguments, context):
        values = {p.name: unpack_value(p.read(arguments), context.types[p.read(node.inputs)])
                  for p in cls.input_parameters}
        gate = values["gate_logit"].float().sigmoid().to(context.torch.bfloat16).float()
        scaled = (values["shared"].float() * gate).to(context.torch.bfloat16).float()
        merged = values["routed"].float() + scaled
        output = values["residual"].float() + merged
        stats = norm_stats_value(output, axis=1, use_mean=node.attrs["use_mean"])
        return pack_result(output, node.type.fields[0], context), stats

    @classmethod
    def cost(cls, node):
        sizes = tuple(tensor_nbytes(tensor_of(t)) for t in node.type.fields)
        return OpCost(bytes_written=None if None in sizes else sum(sizes),
                      notes=("gated-bf16-product-fp32-ordered-residual", "local-additive-stats"))

    @classmethod
    def python_call(cls, node):
        return PythonCall("__import__('agent_optimizations.gated_epilogue', fromlist=['gated_residual_norm_stats']).gated_residual_norm_stats",
                          tuple(NodeRef(n) for n in node.inputs),
                          {**dict(node.attrs), "name": node.id, "metadata": dict(node.metadata)})


def gated_residual_norm_stats(routed, shared, gate_logit, residual, *, use_mean=False, name=None, metadata=None):
    return GatedResidualNormStats.construct(routed, shared, gate_logit, residual,
                                            use_mean=use_mean, name=name, metadata=metadata)

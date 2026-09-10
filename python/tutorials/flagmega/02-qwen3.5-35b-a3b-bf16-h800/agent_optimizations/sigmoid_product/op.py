"""A BF16 boundary around a materialized FP32 sigmoid/product."""

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import DType, DistributedType, VectorType
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.core import (
    NodeRef,
    OpCost,
    OpDefinition,
    PythonCall,
    input_parameter,
    op_definition,
    tensor_nbytes,
)
from triton.flagmega.ir.type_pattern import is_tensor


@op_definition("local.sigmoid_product", display_name="Local.SigmoidProduct")
class SigmoidProduct(OpDefinition):
    supports_broadcast_lifting = False
    const_evaluable = True
    value = input_parameter(is_tensor())
    gate = input_parameter(is_tensor())

    @classmethod
    def infer_type(cls, inputs, attrs):
        value, gate = (p.type_of(inputs) for p in cls.input_parameters)
        dtype = tensor_of(value).dtype
        scalar = dtype.elem_type if isinstance(dtype, VectorType) else dtype
        if scalar != DType.BFLOAT16 or value != gate:
            raise IRSchemaError("SigmoidProduct requires identical BF16 operand types and local ownership.")
        if isinstance(value, DistributedType) and value.partial is not None:
            raise IRSchemaError("SigmoidProduct requires materialized operands, not partial reductions.")
        return value

    @classmethod
    def evaluate(cls, node, arguments, context):
        value, gate = (p.read(arguments).float() for p in cls.input_parameters)
        return (value * gate.sigmoid()).to(context.torch.bfloat16)

    @classmethod
    def cost(cls, node):
        size = tensor_nbytes(node.type)
        return OpCost(bytes_read=None if size is None else size * 2, bytes_written=size,
                      notes=("bf16-input-fp32-sigmoid-product-bf16-output", ))

    @classmethod
    def python_call(cls, node):
        return PythonCall("__import__('agent_optimizations.sigmoid_product', fromlist=['sigmoid_product']).sigmoid_product",
                          tuple(NodeRef(n) for n in node.inputs), {"name": node.id, "metadata": dict(node.metadata)})


def sigmoid_product(value, gate, *, name=None, metadata=None):
    return SigmoidProduct.construct(value, gate, name=name, metadata=metadata)

"""One materialized floating-point product, with a genuinely scalar operand."""

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import DType, DistributedType, VectorType
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import is_fully_replicated
from triton.flagmega.ir.ops.core import (
    NodeRef, OpCost, OpDefinition, PythonCall, input_parameter, op_definition, tensor_elements, tensor_nbytes,
)
from triton.flagmega.ir.type_pattern import is_tensor


@op_definition("local.scalar_scale", display_name="Local.ScalarScale")
class ScalarScale(OpDefinition):
    value = input_parameter(is_tensor())
    scale = input_parameter(is_tensor())
    const_evaluable = True

    @classmethod
    def infer_type(cls, inputs, attrs):
        value, scale = cls.value.type_of(inputs), cls.scale.type_of(inputs)
        vt, st = tensor_of(value), tensor_of(scale)
        dtype = vt.dtype.elem_type if isinstance(vt.dtype, VectorType) else vt.dtype
        if dtype not in {DType.BFLOAT16, DType.FLOAT32} or st.dtype != dtype:
            raise IRSchemaError("ScalarScale requires matching BF16/FP32 scalar element types.")
        if any(not d.is_fixed or d.fixed_value != 1 for d in st.shape):
            raise IRSchemaError("ScalarScale scale must contain exactly one scalar.")
        if isinstance(value, DistributedType) and value.partial is not None:
            raise IRSchemaError("ScalarScale requires a materialized value, not a partial reduction.")
        if isinstance(scale, DistributedType):
            if not is_fully_replicated(scale):
                raise IRSchemaError("ScalarScale scale must be fully replicated.")
            if not isinstance(value, DistributedType) or value.placement != scale.placement:
                raise IRSchemaError("ScalarScale operands must use the same placement.")
        return value

    @classmethod
    def evaluate(cls, node, arguments, context):
        return cls.value.read(arguments) * cls.scale.read(arguments).reshape(())

    @classmethod
    def cost(cls, node):
        tensor = tensor_of(node.type)
        size = tensor_nbytes(tensor)
        dtype = tensor.dtype.elem_type if isinstance(tensor.dtype, VectorType) else tensor.dtype
        return OpCost(flops=tensor_elements(tensor), bytes_read=None if size is None else size + dtype.itemsize,
                      bytes_written=size, notes=("scalar-broadcast-fused",))

    @classmethod
    def python_call(cls, node):
        return PythonCall("__import__('agent_optimizations.scalar_scale', fromlist=['scalar_scale']).scalar_scale",
                          tuple(NodeRef(n) for n in node.inputs), {"name": node.id, "metadata": dict(node.metadata)})


def scalar_scale(value, scale, *, name=None, metadata=None):
    return ScalarScale.construct(value, scale, name=name, metadata=metadata)

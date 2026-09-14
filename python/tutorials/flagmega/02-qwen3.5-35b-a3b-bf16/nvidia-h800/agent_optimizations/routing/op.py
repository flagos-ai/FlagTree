"""Explicit full-softmax -> stable TopK -> selected-probability normalization."""

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import DType, Node
from triton.flagmega.ir.axis import normalize_axis
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.core import (
    NodeRef, OpCost, OpDefinition, PythonCall, attribute_parameter, input_parameter, op_definition, tensor_nbytes,
)
from triton.flagmega.ir.ops.nn.softmax import Softmax
from triton.flagmega.ir.ops.tensors.top_k import TopK
from triton.flagmega.ir.type_pattern import is_tensor


@op_definition("local.staged_routing", display_name="Local.StagedRouting")
class StagedRouting(OpDefinition):
    value = input_parameter(is_tensor())
    k = attribute_parameter()
    axis = attribute_parameter(default=-1)
    index_dtype = attribute_parameter(default="int64")
    const_evaluable = True

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = super().normalize_attrs(attributes)
        TopK.normalize_attrs({**attrs, "largest": True, "sorted": True})
        return attrs

    @classmethod
    def infer_type(cls, inputs, attrs):
        source = cls.value.type_of(inputs)
        tensor = tensor_of(source)
        axis = normalize_axis(attrs["axis"], tensor.rank)
        if tensor.dtype != DType.FLOAT32 or axis != tensor.rank - 1 or attrs["k"] <= 0:
            raise IRSchemaError("Staged routing requires FP32, a last-axis reduction and positive k.")
        Softmax.infer_type(inputs, {"axis": axis})
        return TopK.infer_type(inputs, {**attrs, "largest": True, "sorted": True})

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = cls.value.read(arguments)
        axis, k = node.attrs["axis"], node.attrs["k"]
        probabilities = value.float().softmax(dim=axis)
        order = context.torch.argsort(probabilities, dim=axis, descending=True, stable=True).narrow(axis, 0, k)
        selected = probabilities.gather(axis, order)
        return selected / selected.sum(axis, keepdim=True), order.to(getattr(context.torch, node.attrs["index_dtype"]))

    @classmethod
    def cost(cls, node):
        sizes = [tensor_nbytes(t) for t in node.type.fields]
        return OpCost(bytes_written=None if None in sizes else sum(sizes), notes=("explicit-staged-routing", ))

    @classmethod
    def python_call(cls, node):
        return PythonCall(
            "__import__('agent_optimizations.routing', fromlist=['staged_routing']).staged_routing",
            (NodeRef(node.inputs[0]), ), {**node.attrs, "name": node.id, "metadata": dict(node.metadata)})


def staged_routing(value: Node, *, k: int, axis: int = -1, index_dtype: str = "int64", name=None, metadata=None):
    return StagedRouting.construct(value, k=k, axis=axis, index_dtype=index_dtype, name=name, metadata=metadata)

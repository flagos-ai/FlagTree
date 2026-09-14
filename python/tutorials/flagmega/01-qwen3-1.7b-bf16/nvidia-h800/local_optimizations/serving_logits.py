"""An explicit serving output ABI: the engine, not this graph, samples tokens."""

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import DType, Node
from triton.flagmega.ir.ops.core import (
    NodeRef, OpCost, OpDefinition, PythonCall, input_parameter, op_definition,
)
from triton.flagmega.ir.type_pattern import has_rank, is_tensor


@op_definition("tutorial.serving_logits", display_name="Serving.Logits")
class ServingLogits(OpDefinition):
    logits = input_parameter(is_tensor() & has_rank(2))
    const_evaluable = True

    @classmethod
    def infer_type(cls, inputs, attrs):
        value_type = cls.logits.type_of(inputs)
        if attrs or value_type.dtype != DType.FLOAT32:
            raise IRSchemaError("Serving logits require float32[batch, vocabulary], without attributes.")
        return value_type

    @classmethod
    def evaluate(cls, node, arguments, context):
        return cls.logits.read(arguments)

    @classmethod
    def cost(cls, node):
        # This boundary is removed by an explicit local lowering rule.
        return OpCost.exact_zero(notes=("lowered serving ABI boundary",))

    @classmethod
    def python_call(cls, node):
        # Ordinary Python import + handwritten builder. A fresh process needs
        # only this tutorial on PYTHONPATH, not prior registry side effects.
        return PythonCall(
            "__import__('local_optimizations', fromlist=['serving_logits']).serving_logits",
            (NodeRef(cls.logits.read(node.inputs)),),
            {"name": node.id, "metadata": dict(node.metadata)},
        )


def serving_logits(logits: Node, *, name: str | None = None, metadata=None) -> Node:
    return ServingLogits.construct(logits, name=name, metadata=metadata)

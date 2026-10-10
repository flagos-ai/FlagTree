"""Locally registered passes/rules; no compiler-global pipeline mutation."""

from dataclasses import replace

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import DType, Node, RefType, TensorType
from triton.flagmega.passes import DataflowPass, FunctionalPass
from triton.flagmega.rules import RewriteRedirect, RewriteRule

from .serving_logits import ServingLogits


def expose_serving_logits(module):
    if module.stage != "imported":
        raise IRSchemaError("Serving ABI specialization must precede distribution/bufferization.")
    entry = module.function_map[module.entry]
    values = [module.node_map[value] for value in entry.outputs]
    logits = [value for value in values if isinstance(value.type, TensorType)
              and value.type.dtype == DType.FLOAT32 and value.type.rank == 2]
    states = [value for value in values if isinstance(value.type, RefType)]
    tokens = [value for value in values if isinstance(value.type, TensorType)
              and value.type.dtype == DType.INT32 and value.op == "nn.greedy_sample"]
    if len(logits) == 1 and len(states) == 1 and len(values) == 2:
        return module  # Already specialized: editable checkpoints are resumable.
    if len(logits) != 1 or len(states) != 1 or len(tokens) != 1 or len(values) != 3:
        raise IRSchemaError("Expected exactly logits, greedy token and paged state entry outputs.")
    marker = Node("serving_logits", ServingLogits.op_name, (logits[0].id,), logits[0].type)
    updated = replace(entry, outputs=(marker.id, states[0].id))
    return replace(module, nodes=module.nodes + (marker,), functions=tuple(
        updated if function.name == entry.name else function for function in module.functions))


def serving_passes():
    # Extra local semantic fusion rules belong here as ordinary RewriteRules.
    # This concrete specialization fuses the model output boundary into the
    # serving engine: its logits stay, its now-unobserved argmax is DCE'd.
    lower_boundary = RewriteRule(
        "tutorial.lower_serving_logits",
        matches=lambda node, module: node.op == ServingLogits.op_name,
        rewrite=lambda node, module: RewriteRedirect(ServingLogits.logits.read(node.inputs)),
    )
    return (
        FunctionalPass("ExposeServingLogits", expose_serving_logits),
        DataflowPass("LowerServingLogitsAndPruneSampling", (lower_boundary,)),
    )

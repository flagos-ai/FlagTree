# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Block-local log decay scan with explicit rounding topology.

Each block restarts the prefix. Each scan group uses a Hillis--Steele tree;
group totals are then propagated in order. Padding has alpha=1 and therefore
retains the last valid prefix. The CPU evaluator is an arithmetic reference,
not a bitwise oracle for a target's approximate log2 instruction.
"""

import math

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.model import DistributedType, SBP, tensor_type
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, attribute_parameter, input_parameter, op_definition, tensor_nbytes
from triton.flagmega.ir.type_pattern import has_rank, is_tensor
from triton.flagmega.ir.types import DType


@op_definition("nn.delta_rule_log_prefix", namespace="nn", functional_name="delta_rule_log_prefix",
               display_name="NN.DeltaRuleLogPrefix")
class DeltaRuleLogPrefix(OpDefinition):
    supports_broadcast_lifting = False
    alpha = input_parameter(is_tensor() & has_rank(2))
    block_size = attribute_parameter(default=64)
    scan_group_size = attribute_parameter(default=32)
    epsilon = attribute_parameter(default=1e-10)
    log2_mode = attribute_parameter(default="fast")

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = super().normalize_attrs(attributes)
        for name in ("block_size", "scan_group_size"):
            size = attrs[name]
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0 or size & (size - 1):
                raise IRSchemaError(f"DeltaRuleLogPrefix {name} must be a positive power of two.")
        if attrs["scan_group_size"] > attrs["block_size"]:
            raise IRSchemaError("DeltaRuleLogPrefix scan_group_size must not exceed block_size.")
        epsilon = attrs["epsilon"]
        if isinstance(epsilon, bool) or not isinstance(epsilon,
                                                       (int, float)) or not math.isfinite(epsilon) or epsilon < 0:
            raise IRSchemaError("DeltaRuleLogPrefix epsilon must be finite and nonnegative.")
        attrs["epsilon"] = float(epsilon)
        if attrs["log2_mode"] not in ("fast", "accurate"):
            raise IRSchemaError("DeltaRuleLogPrefix log2_mode must be fast or accurate.")
        return attrs

    @classmethod
    def infer_type(cls, inputs, attrs):
        source = cls.alpha.type_of(inputs)
        alpha = tensor_of(source)
        if alpha.dtype != DType.FLOAT32:
            raise IRSchemaError("DeltaRuleLogPrefix alpha must have FP32 scalar elements.")
        if alpha.shape[1].is_fixed and alpha.shape[1].fixed_value <= 0:
            raise IRSchemaError("DeltaRuleLogPrefix requires positive heads.")
        block = attrs["block_size"]
        result = tensor_type("float32", ((alpha.shape[0] + block - 1) // block, alpha.shape[1], block))
        if isinstance(source, DistributedType):
            broadcast = SBP.broadcast()
            if source.partial is not None or source.axis_policies[0] != broadcast:
                raise IRSchemaError("DeltaRuleLogPrefix requires a materialized token axis, without partials.")
            return DistributedType(result, (broadcast, source.axis_policies[1], broadcast), source.placement)
        return result

    @classmethod
    def evaluate(cls, node, arguments, context):
        return delta_rule_log_prefix(cls.alpha.read(arguments), block_size=node.attrs["block_size"],
                                     scan_group_size=node.attrs["scan_group_size"], epsilon=node.attrs["epsilon"],
                                     torch=context.torch)

    @classmethod
    def cost(cls, node):
        return OpCost(bytes_written=tensor_nbytes(tensor_of(node.type)),
                      notes=("block-local-grouped-log2-scan", "fp32-addition-tree"))


def delta_rule_log_prefix(alpha, *, block_size=64, scan_group_size=32, epsilon=1e-10, torch):
    tokens, heads = alpha.shape
    blocks = (tokens + block_size - 1) // block_size
    values = alpha.new_ones((blocks * block_size, heads))
    values[:tokens] = alpha
    # Padding has log decay zero, including when epsilon is large enough to
    # change log2(1 + epsilon); it is not an extra recurrence step.
    valid = torch.arange(blocks * block_size, device=alpha.device) < tokens
    values = torch.where(valid[:, None], torch.log2(values + epsilon), 0)
    values = values.reshape(blocks, block_size, heads).transpose(1, 2)
    groups = values.reshape(blocks, heads, block_size // scan_group_size, scan_group_size)
    distance = 1
    while distance < scan_group_size:
        groups = torch.cat((groups[..., :distance], groups[..., distance:] + groups[..., :-distance]), dim=-1)
        distance *= 2
    groups = groups.clone()
    for group in range(1, block_size // scan_group_size):
        groups[..., group, :] += groups[..., group - 1, -1:]
    return groups.reshape(blocks, heads, block_size).contiguous()


__all__ = ["DeltaRuleLogPrefix"]

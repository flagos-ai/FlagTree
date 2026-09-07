# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Collective argmax candidates from producer and target-owned leaf layouts."""

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import DistributedType, Node
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.nn.greedy_sample import GreedySample
from triton.flagmega.passes.auto_distributed.candidates import (
    DistributedCandidate, DistributedCandidateProviderBase,
)
from triton.flagmega.passes.auto_distributed.candidate_identity import distributed_candidate_id


class GreedySampleCandidateProvider(DistributedCandidateProviderBase):
    op_names = frozenset({"nn.greedy_sample"})
    allows_partial_inputs = False
    is_exhaustive = True

    def _enumerate_candidates(self, context):
        node = context.source_call
        tensor = tensor_of(context.module.node_map[GreedySample.logits.read(node.inputs)].type)
        values = dict.fromkeys((
            *(GreedySample.logits.read(context.available_input_types) if context.available_input_types else ()),
            *context.leaf_candidate_types(tensor),
        ))
        result = []
        for value in values:
            if not isinstance(value, DistributedType) or value.placement != context.placement or value.tensor != tensor:
                continue
            typed = Node("<logits>", "builtin.var", (), value, attrs={"name": "<logits>"})
            try:
                output = GreedySample.infer_type((typed,), node.attrs)
            except IRSchemaError:
                continue
            factors = GreedySample.cost_factors((typed,), node.attrs, output)
            result.append(DistributedCandidate(
                distributed_candidate_id(node.id, "greedy_sample", output, (value,)),
                output, (value,),
                1 << 20 if factors is None else context.operation_cost_model.get_latency(factors, output),
                "local-argmax-and-global-index-merge",
                objective_kind="heuristic" if factors is None else "analytic",
                objective_model="flagmega.dynamic-argmax/v1" if factors is None else context.operation_cost_model.identity,
                objective_evidence=("op-definition-cost-factors", "target-owned-input-split", "materialized-argmax"),
            ))
        return tuple(result)

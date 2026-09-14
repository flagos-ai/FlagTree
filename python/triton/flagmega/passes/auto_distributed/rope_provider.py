# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Pair-local RoPE candidates with explicit input layout adaptation."""

from triton.flagmega.ir.distributed_inference import broadcast_ir_type, tensor_of
from triton.flagmega.passes.auto_distributed.inference_providers import TypeInferenceCandidateProvider
from triton.flagmega.passes.auto_distributed.rotary_layouts import rotary_layouts


class RoPECandidateProvider(TypeInferenceCandidateProvider):
    def __init__(self):
        super().__init__(frozenset({"nn.rope", "ntt.vectorized_rope"}))

    def _enumerate_candidates(self, context):
        results = {candidate.id: candidate for candidate in super()._enumerate_candidates(context)}
        node = context.source_call
        tensors = tuple(tensor_of(context.module.node_map[value].type) for value in node.inputs)
        tables = tuple(broadcast_ir_type(tensor, context.placement) for tensor in tensors[1:])
        for layout in dict.fromkeys(rotary_layouts(context, tensors[0], 1, 2, node.attrs.get("rotary_dim"))):
            candidate = self._candidate(context, (layout, *tables))
            if candidate is not None:
                results[candidate.id] = candidate
        return tuple(results.values())

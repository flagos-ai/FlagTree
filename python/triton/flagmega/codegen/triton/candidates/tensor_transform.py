# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Owner-local tensor indexing operations, independent of vector scheduling."""

from .core import TritonCandidateProposal


class TensorTransformCandidateProvider:
    op_names = frozenset({
        "tensors.pad", "tensors.slice", "tensors.slice_to_shape", "tensors.pack", "tensors.unpack", "tensors.concat",
        "tensors.broadcast_to"
    })

    def propose(self, node, context):
        family = "slice" if node.op in {"tensors.slice", "tensors.slice_to_shape"} else node.op.removeprefix("tensors.")
        candidates = tuple(
            context.configure_implementation(implementation)
            for implementation in context.implementations(family, indexing="local"))
        if not candidates:
            return None
        return TritonCandidateProposal(
            candidates, context.choose_default(family, candidates, portable_fallback=f"tir.{family}.local"))

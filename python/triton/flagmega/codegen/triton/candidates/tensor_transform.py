# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Owner-local tensor indexing operations, independent of vector scheduling."""

from triton.flagmega.ir.axis import normalize_axis
from triton.flagmega.ir.distributed_inference import tensor_of

from .core import TritonCandidateProposal


class TensorTransformCandidateProvider:
    op_names = frozenset({
        "tensors.pad", "tensors.slice", "tensors.slice_to_shape", "tensors.pack", "tensors.unpack", "tensors.concat",
        "tensors.broadcast_to", "tensors.gather"
    })

    def propose(self, node, context):
        family = "slice" if node.op in {"tensors.slice", "tensors.slice_to_shape"} else node.op.removeprefix("tensors.")
        implementations = context.implementations(family, indexing="local")
        if family == "concat":
            result_type = tensor_of(context.module.node_map[node.id].type)
            axis = normalize_axis(node.attrs["axis"], result_type.rank)
            extent = result_type.shape[axis].fixed_value
            if extent is not None:
                implementations = tuple(
                    implementation for implementation in implementations
                    if int(implementation.parameters["elements_per_program"]) <= extent
                )
            if not implementations:
                available = context.implementations(family, indexing="local")
                if not available:
                    return None
                implementations = (min(
                    available,
                    key=lambda implementation: int(implementation.parameters["elements_per_program"])), )
        candidates = tuple(context.configure_implementation(implementation) for implementation in implementations)
        if not candidates:
            return None
        candidates = tuple(sorted(candidates, key=lambda candidate: int(candidate.parameters["elements_per_program"])))
        return TritonCandidateProposal(
            candidates, context.choose_default(family, candidates,
                                                 portable_fallback=candidates[0].id
                                                 if family == "concat" else f"tir.{family}.local"))

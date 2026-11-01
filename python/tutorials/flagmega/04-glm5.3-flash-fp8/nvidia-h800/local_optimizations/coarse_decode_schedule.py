"""Measured complete-K MoE and cooperative mHC choices for this workload."""

from dataclasses import replace

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.selection import apply_plan, override_plan
from triton.flagmega.targets.nvidia.mhc import mhc_cooperative_contiguous_tf32_implementations
from triton.flagmega.targets.nvidia.sparse_experts import sparse_experts_complete_k_implementations


COMPLETE_K = "tir.sparse_experts_gate_up.complete_k_wgmma_n32"
COOPERATIVE_TF32 = "tir.mhc_pre.cooperative_contiguous_tf32"
POLICY = "glm53-complete-k-and-cooperative-mhc/v1"


def implementation_model(base):
    additions = (
        *mhc_cooperative_contiguous_tf32_implementations(),
        *(
            replace(value, parameters={**value.parameters, "num_stages": 3})
            for value in sparse_experts_complete_k_implementations()
        ),
    )
    implementations = list(base.implementations)
    for value in additions:
        existing = base.implementation(value.id)
        if existing is None:
            implementations.append(value)
        elif existing != value:
            raise IRVerificationError(f"The measured decode implementation has a conflicting contract: {value.id}")

    def configure(value):
        if value.family == "partitioned_sparse_experts_down" and value.variant == "wgmma":
            return replace(value, parameters={**value.parameters, "output_tile_n": 256})
        if value.family == "mhc_pre" and value.variant == "cooperative_stream_fma":
            return replace(
                value,
                parameters={**value.parameters, "independent_collapse_pre": True},
                facts={**value.facts, "internal_grid_barriers": 1},
            )
        return value

    return replace(base, implementations=tuple(configure(value) for value in implementations))


def select(module):
    choices = []
    for point in module.selection_points:
        if point.kind != "tir" or point.owner not in module.node_map:
            continue
        node = module.node_map[point.owner]
        record = module.selection_map.get(point.id)
        if record is None:
            continue
        identity = None
        if (
            node.op == "nn.sparse_experts_gate_up"
            and record.candidate_id == "tir.sparse_experts_gate_up.cooperative_wgmma"
        ):
            identity = COMPLETE_K
        elif (
            node.op == "nn.mhc_pre"
            and node.attrs.get("projection_reduction") == "contiguous_tf32"
            and record.candidate_id == "tir.mhc_pre.contiguous_tf32"
        ):
            identity = COOPERATIVE_TF32
        if identity is not None and identity in {candidate.id for candidate in point.candidates}:
            choices.append((point.id, identity))
    if not choices:
        return module
    return apply_plan(
        module,
        override_plan(
            module,
            choices,
            policy=POLICY,
            rationale="Retain the ordered FP8 scale-block FMA and existing TF32 splits; complete-K tasks and parallel split projection were measured with independent full-model numerical checks.",
        ),
    )

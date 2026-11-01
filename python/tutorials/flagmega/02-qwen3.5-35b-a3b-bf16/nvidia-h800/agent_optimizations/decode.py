"""Reproduce the measured native-BF16 decode policy without private scripts."""

from dataclasses import replace

from triton.flagmega import ir as fm
from .qkv import install as install_qkv
from .shared_router import create_target as router_target, install as install_router


QKV_PIPELINE = "tir.qkv_parallel_linear.packed_partial_mma_n_tiled_table_pipeline"
ORDERED_QKV_PIPELINE = "tir.qkv_parallel_linear.packed_ordered_mma_table_pipeline"
PROJECTION_K = {
    "tir.dense_matmul.packed_tensor_descriptor_smem_pipeline_gemv": 512,
    "tir.dense_matmul.packed_tensor_descriptor_table_smem_pipeline_gemv_tn64_bk512": 256,
    "tir.dense_matmul.packed_tensor_descriptor_table_smem_pipeline_gemv_norm_stats": 512,
}


def install():
    install_qkv()
    install_router()


def create_target(*, qkv_kernel="mma", bufferize_opt_level="optimized"):
    if qkv_kernel not in {"mma", "gemv"}:
        raise ValueError("qkv_kernel must be mma or gemv")
    target = router_target()
    model = target.triton_implementation_model
    entries = []
    ordered_projections = {}
    for value in model.implementations:
        if value.id in PROJECTION_K:
            k = PROJECTION_K[value.id]
            parameters = {**value.parameters, "block_k": k}
            if "reduction_unroll" in parameters:
                parameters["reduction_unroll"] = min(parameters["reduction_unroll"], k // parameters["reduction_group"])
            channel_indices = {i for c in value.transfer_pipeline.channels for i in c.shared_workspace_indices}
            workspaces = []
            for index, workspace in enumerate(value.shared_workspaces):
                if index in channel_indices:
                    shape = tuple(d.fixed_value for d in workspace.type.shape)
                    expected = (value.transfer_pipeline.capacity, value.parameters["block_k"] // 16,
                                value.parameters["tile_n"] // 8, 2, 64)
                    if shape != expected:
                        raise ValueError(f"Packed projection stage contract changed: {value.id}")
                    workspace = replace(workspace, type=fm.tensor_type(workspace.type.dtype, (shape[0], k // 16, *shape[2:])))
                workspaces.append(workspace)
            value = replace(value, parameters=parameters, shared_workspaces=tuple(workspaces))
            # Native ordered arithmetic is a separate implementation. Replacing
            # the SIMT GEMV also forced unordered BF16 imports to execute sixteen
            # replicated MMA rows for their single-token projection.
            ordered = replace(value, id=value.id + "_ordered_mma",
                parameters={**parameters, "accumulation_mode": "ordered_mma", "mma_block_k": 128},
                facts={**value.facts, "ordered_mma_accumulation": True})
            ordered_projections[value.id] = ordered.id
            entries.append(ordered)
        elif value.id == "tir.add_norm_stats.persistent_rms":
            value = replace(value, parameters={**value.parameters, "tile": 2048})
        elif value.id == "tir.paged_attention_partial.decode_t64":
            value = replace(value, parameters={**value.parameters, "compute_num_warps": 8})
        elif value.id == QKV_PIPELINE:
            stages, k = 2, 64
            n = value.parameters["block_n"]
            value = replace(value, parameters={**value.parameters, "block_k": k, "num_stages": stages},
                            shared_workspaces=(replace(value.shared_workspaces[0],
                                type=fm.tensor_type("bfloat16", (stages, n // 8, k // 16, 2, 64))),
                                *value.shared_workspaces[1:]),
                            transfer_pipeline=replace(value.transfer_pipeline, capacity=stages))
        entries.append(value)
    preferences = {**model.preferences,
                   "paged_attention_partial": tuple(dict.fromkeys((
                       "tir.paged_attention_partial.decode_t64",
                       *model.preferences["paged_attention_partial"]))),
                   "add_norm_stats": tuple(dict.fromkeys(("tir.add_norm_stats.persistent_rms", *model.preferences["add_norm_stats"]))),
                   "qkv_parallel_linear": tuple(dict.fromkeys((
                       ORDERED_QKV_PIPELINE,
                       QKV_PIPELINE if qkv_kernel == "mma" else
                       "tir.qkv_parallel_linear.packed_fused_gemv",
                       *model.preferences["qkv_parallel_linear"]))) }
    for family in ("dense_matmul", "dense_matmul_norm_stats"):
        if family in preferences:
            preferences[family] = tuple(identity for original in preferences[family]
                for identity in ((original, ordered_projections[original])
                                 if original in ordered_projections else (original,)))
    target.triton_implementation_model = replace(model, implementations=tuple(entries), preferences=preferences)
    return target.with_bufferize_opt_level(bufferize_opt_level)

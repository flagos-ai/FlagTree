"""Reproduce the measured native-BF16 decode policy without private scripts."""

from dataclasses import replace

from triton.flagmega import ir as fm
from .qkv import install as install_qkv
from .shared_router import create_target as router_target, install as install_router


QKV_PIPELINE = "tir.qkv_parallel_linear.packed_partial_mma_n_tiled_table_pipeline"
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
        elif value.id == "tir.add_norm_stats.persistent_rms":
            value = replace(value, parameters={**value.parameters, "tile": 2048})
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
                   "add_norm_stats": tuple(dict.fromkeys(("tir.add_norm_stats.persistent_rms", *model.preferences["add_norm_stats"]))),
                   "qkv_parallel_linear": (QKV_PIPELINE if qkv_kernel == "mma" else
                                           "tir.qkv_parallel_linear.packed_fused_gemv",)}
    target.triton_implementation_model = replace(model, implementations=tuple(entries), preferences=preferences)
    return target.with_bufferize_opt_level(bufferize_opt_level)

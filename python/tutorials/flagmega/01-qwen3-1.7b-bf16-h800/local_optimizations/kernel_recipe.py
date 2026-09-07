"""Explicit transport experiments on reviewed kernel candidates."""

from triton.flagmega.selection import override_plan


def select_tir(module, *, residual_kernel="direct"):
    if residual_kernel not in {"direct", "staged", "async"}:
        raise ValueError(f"Unknown residual transport {residual_kernel}")
    paired = "tir.dense_matmul_glu.packed_tensor_descriptor_table_paired_smem_pipeline_inline_gemv"
    base = "tir.dense_matmul.packed_tensor_descriptor_table_smem_pipeline_gemv_norm_stats"
    choices = []
    residual_points = 0
    for point in module.selection_points:
        if point.kind != "tir":
            continue
        ids = {candidate.id for candidate in point.candidates}
        if paired in ids:
            choices.append((point.id, paired))
        if any(
            c.parameters.get("epilogue") == "residual_norm_stats" for c in point.candidates
        ):
            selected = base + {"direct": "", "staged": "_lhs8192", "async": "_lhs8192_async"}[residual_kernel]
            if selected not in ids:
                raise ValueError(f"Requested residual transport {selected} is not legal at {point.id}")
            choices.append((point.id, selected))
            residual_points += 1
    if residual_kernel != "direct" and not residual_points:
        raise ValueError("Requested residual transport has no applicable GEMV points")
    return override_plan(module, choices,
        rationale=f"Paired gate/up TMA staging; explicit residual GEMV transport={residual_kernel}")

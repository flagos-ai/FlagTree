"""The measured Agent pipeline keeps the native projection contract."""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_agent_attention_reduction_fits_one_warp_per_token():
    sys.path.insert(0, str(ROOT))
    try:
        from agent_optimizations.decode import create_target
        from agent_optimizations.shared_router import Registry
        from triton.flagmega.codegen.triton.templates import KernelTemplateSpec
        model = create_target().triton_implementation_model
        candidates = tuple(v.id for v in model.implementations
                           if v.family == "paged_attention_partial" and v.facts.get("portable_triton"))
        identity = model.choose_default("paged_attention_partial", candidates)
        assert identity == "tir.paged_attention_partial.decode_t64"
        implementation = next(v for v in model.implementations if v.id == identity)
        assert implementation.parameters["compute_num_warps"] == 8
        spec = KernelTemplateSpec("paged_attention_partial", "decode_t64", "nvidia", "sm90")
        source = Registry().render_kernel(spec, {}).source
        assert "[1, 1], [1, 32], [8, 1], [1, 0]" in source
        assert "key_value = tle.gpu.set_layout(" in source
        assert "context_length <= context_shards * 32" in source
        assert "if online_bf16:" in source
    finally:
        sys.path.remove(str(ROOT))


def test_agent_retains_hopper_ordered_expert_catalog():
    sys.path.insert(0, str(ROOT))
    try:
        from agent_optimizations.decode import create_target
        model = create_target().triton_implementation_model
        implementations = model.implementations
        for family in ("sparse_experts_gate_up", "sparse_experts_down"):
            ordered = tuple(v.id for v in implementations
                            if v.family == family and v.facts.get("ordered_mma_accumulation"))
            assert ordered
            assert model.choose_default(family, ordered) in ordered
    finally:
        sys.path.remove(str(ROOT))


@pytest.mark.parametrize("identity", [
    "tir.dense_matmul.packed_tensor_descriptor_smem_pipeline_gemv",
    "tir.dense_matmul.packed_tensor_descriptor_table_smem_pipeline_gemv_tn64_bk512",
    "tir.dense_matmul.packed_tensor_descriptor_table_smem_pipeline_gemv_norm_stats",
])
def test_agent_projection_pipeline_keeps_ordered_accumulation(identity):
    sys.path.insert(0, str(ROOT))
    try:
        from agent_optimizations.decode import create_target
        from triton.flagmega.codegen.triton.candidates.dense_matmul import _supports_reduction_order
        model = create_target().triton_implementation_model
        implementation = next(v for v in model.implementations if v.id == identity + "_ordered_mma")
        assert implementation.parameters.get("accumulation_mode") == "ordered_mma"
        assert implementation.parameters["mma_block_k"] == 128
        assert _supports_reduction_order(implementation, {"reduction_order": "mma_k16"})
        assert implementation.parameters["block_k"] % implementation.parameters["mma_block_k"] == 0
    finally:
        sys.path.remove(str(ROOT))


@pytest.mark.parametrize("identity", [
    "tir.dense_matmul.packed_tensor_descriptor_smem_pipeline_gemv",
    "tir.dense_matmul.packed_tensor_descriptor_table_smem_pipeline_gemv_tn64_bk512",
    "tir.dense_matmul.packed_tensor_descriptor_table_smem_pipeline_gemv_norm_stats",
])
def test_agent_unordered_projection_does_not_inherit_native_mma_policy(identity):
    sys.path.insert(0, str(ROOT))
    try:
        from agent_optimizations.decode import create_target
        from triton.flagmega.codegen.triton.candidates.dense_matmul import _supports_reduction_order
        model = create_target().triton_implementation_model
        baseline = next(v for v in model.implementations if v.id == identity)
        assert baseline.parameters.get("accumulation_mode", "simt") == "simt"
        assert not _supports_reduction_order(baseline, {"reduction_order": "mma_k16"})
        ordered = next(v for v in model.implementations if v.id == identity + "_ordered_mma")
        assert model.choose_default(baseline.family, (baseline.id, ordered.id)) == baseline.id
        assert model.choose_default(baseline.family, (ordered.id,)) == ordered.id
    finally:
        sys.path.remove(str(ROOT))

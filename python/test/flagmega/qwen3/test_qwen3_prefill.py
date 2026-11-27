# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Dense Qwen3 prefill (M=num_tokens>1) importer + the M>1 tensor-core (WMMA)
selection on AMD RDNA. The prefill graph returns all-token logits without the
greedy-sample argmax fusion, so the lm_head stays an un-fused M>1 GEMM that the
tir.dense_matmul.mma (WMMA) path claims automatically on a WMMA-capable target.

The multi-layer prefill decoder also lowers end to end: the after-attention
fused residual/RMS-norm (ntt.gather_reduce_add_norm_apply) and the materialized
norm statistics (ntt.add_norm_stats) iterate one independent per-token RMS per
leading row, so a short multi-token prompt compiles through code generation."""

import pytest

from triton.flagmega.codegen.triton.tir_package import describe_tir_package
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import ImporterError
from triton.flagmega.options import CompileOptions
from triton.flagmega.importer import import_qwen3_model

from .helpers import full_checkpoint, full_checkpoint_wmma

AMD_TARGETS = ("amd-gfx1100", "amd-gfx1201")


def test_prefill_shapes_and_no_argmax_fusion():
    decode = import_qwen3_model(full_checkpoint(num_hidden_layers=2))
    prefill = import_qwen3_model(
        full_checkpoint(num_hidden_layers=2), execution_phase="prefill", num_tokens=8
    )

    assert decode.node_map["input_ids"].type.shape[0].fixed_value == 1
    assert prefill.node_map["input_ids"].type.shape[0].fixed_value == 8
    # The decoder layers run over all 8 tokens, but only the last token's hidden
    # is sliced out before the final norm + LM head (next-token prediction), so
    # logits/lm_head are M=1 -- the 152K-vocab LM head is not paid per token.
    assert any(node.id == "prefill_last_hidden" for node in prefill.nodes)
    assert prefill.node_map["logits"].type.shape[0].fixed_value == 1
    assert prefill.node_map["lm_head"].type.shape[0].fixed_value == 1
    # Decode fuses lm_head with greedy-sample; prefill emits bare last-token
    # logits with no sampler.
    assert any(node.op == "nn.greedy_sample" for node in decode.nodes)
    assert not any(node.op == "nn.greedy_sample" for node in prefill.nodes)


def test_decode_requires_single_token():
    with pytest.raises(ImporterError):
        import_qwen3_model(
            full_checkpoint(num_hidden_layers=1), execution_phase="decode-1", num_tokens=4
        )


@pytest.mark.parametrize("target_name", AMD_TARGETS)
def test_m_gt_1_matmul_auto_selects_wmma(target_name):
    # A standalone M>1 (GEMM) bf16 matmul auto-selects the tensor-core (WMMA)
    # tir.dense_matmul.mma on a WMMA-capable target, while M==1 stays the packed
    # gemv. Prefill's lm_head is M=1 (last-token), so this WMMA path is what the
    # per-token layer GEMMs take, not the lm_head.
    from triton.flagmega import ir as fm

    def tir_default(rows):
        builder = fm.IRBuilder(dialect="high_level", stage="imported")
        value = builder.var(
            "value", fm.tensor_type("bfloat16", (rows, 2048)), id="value"
        )
        weight = builder.weight(
            "weight", fm.tensor_type("bfloat16", (2048, 2048)),
            source="memory", key="weight", id="weight",
        )
        projection = builder.call(
            "math.matmul", (value, weight),
            fm.tensor_type("bfloat16", (rows, 2048)), id="projection",
            attrs={"transpose_a": False, "transpose_b": True},
        )
        builder.function("main", (value,), (projection,))
        module = Compiler(CompileOptions(target=target_name)).compile(
            builder.build(entry="main"), stop_after="propose-tir"
        ).module
        point = next(
            p for p in module.selection_points if p.id.startswith("tir.projection")
        )
        return point.default_candidate

    assert tir_default(128) == "tir.dense_matmul.mma"
    assert tir_default(1) != "tir.dense_matmul.mma"


# A spread of token counts exercises the batched norm-statistics epilogue across
# different owner-local row counts and ragged (non-power-of-two, non-divisible)
# token layouts -- the regime that reached review repeatedly because the only
# existing coverage ran a single decode row.
@pytest.mark.parametrize("num_tokens", [3, 5, 33, 128])
@pytest.mark.parametrize("target_name", AMD_TARGETS)
def test_prefill_full_lowering_completes(target_name, num_tokens):
    # A multi-token prefill lowers through code generation end to end, including
    # the after-attention fused residual/RMS-norm. Before the batched
    # norm-statistics epilogue existed this raised an IRVerificationError
    # ("No reviewed Triton TIR candidate ... ntt.gather_reduce_add_norm_apply"):
    # the fused combine's provider only offered a single decode row.
    module = import_qwen3_model(
        full_checkpoint_wmma(num_hidden_layers=2),
        execution_phase="prefill",
        num_tokens=num_tokens,
    )
    compiled = Compiler(CompileOptions(target=target_name)).compile(module)
    assert describe_tir_package(compiled.module)


@pytest.mark.parametrize("target_name", AMD_TARGETS)
def test_prefill_token_sharded_materialized_norm_stats(target_name):
    # A longer prompt token-shards the after-attention statistics so the whole
    # normalized feature axis stays owner-local: NormStats materializes the
    # result (partial=None) and the local RMS implementation must accept that
    # complete per-token statistic. Before the fix this raised a CodegenError
    # ("AddNormStats local_partial_rms requires an explicit Sum-partial ...").
    module = import_qwen3_model(
        full_checkpoint_wmma(num_hidden_layers=2),
        execution_phase="prefill",
        num_tokens=17,
    )
    compiled = Compiler(CompileOptions(target=target_name)).compile(module)
    assert describe_tir_package(compiled.module)

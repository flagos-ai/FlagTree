# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Dense Qwen3 prefill (M=num_tokens>1) importer + the M>1 tensor-core (WMMA)
selection on AMD RDNA. The prefill graph returns all-token logits without the
greedy-sample argmax fusion, so the lm_head stays an un-fused M>1 GEMM that the
tir.dense_matmul.mma (WMMA) path claims automatically on a WMMA-capable target.

Full end-to-end prefill execution is not covered here: the RMS norm-statistics
epilogue is single-row (decode) only by design -- both the fused
ntt.matmul_norm_stats and the ntt.add_norm_stats combine require M==1 -- so the
multi-layer decoder cannot lower-tir yet. That M>1 norm-stats codegen is a
separate follow-up; this test pins the importer + the WMMA selection milestone."""

import pytest

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
    assert prefill.node_map["logits"].type.shape[0].fixed_value == 8
    assert prefill.node_map["lm_head"].type.shape[0].fixed_value == 8
    # Decode fuses lm_head with greedy-sample; prefill returns bare all-token
    # logits so lm_head stays an un-fused GEMM.
    assert any(node.op == "nn.greedy_sample" for node in decode.nodes)
    assert not any(node.op == "nn.greedy_sample" for node in prefill.nodes)


def test_decode_requires_single_token():
    with pytest.raises(ImporterError):
        import_qwen3_model(
            full_checkpoint(num_hidden_layers=1), execution_phase="decode-1", num_tokens=4
        )


@pytest.mark.parametrize("target_name", AMD_TARGETS)
def test_prefill_lm_head_auto_selects_wmma(target_name):
    module = import_qwen3_model(
        full_checkpoint_wmma(num_hidden_layers=2),
        execution_phase="prefill",
        num_tokens=64,
    )
    proposed = Compiler(CompileOptions(target=target_name)).compile(
        module, stop_after="propose-tir"
    ).module

    lm_head = next(
        point for point in proposed.selection_points if point.id == "tir.logits"
    )
    assert lm_head.default_candidate == "tir.dense_matmul.mma"
    assert any(
        candidate.id == "tir.dense_matmul.mma" for candidate in lm_head.candidates
    )

# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Opt-in H800 regression for the exact Qwen3.8-27B layer-0 shapes."""

from __future__ import annotations

import os

import pytest

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.triton import (
    describe_tir_package,
    render_tir_package,
)
from triton.flagmega.compiler import Compiler
from triton.flagmega.importer import Qwen35Layer0Importer, TensorInfo
from triton.flagmega.ir import DType
from triton.flagmega.runtime import load as load_runtime


torch = pytest.importorskip("torch")


_PREFIX = "model.language_model.layers.0."
_EMBEDDING_KEY = "model.language_model.embed_tokens.weight"


def _device_wrapper(source: str, symbol: str) -> str:
    return source.split(f"def {symbol}(", 1)[1].split("\n@triton.jit", 1)[0]


class _ZeroFullShapeCheckpoint:
    def __init__(self) -> None:
        self._config = {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": {
                "layer_types": ["linear_attention"],
                "vocab_size": 248320,
                "num_hidden_layers": 64,
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 48,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "rms_norm_eps": 1e-6,
            },
            "quantization_config": {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "weight_block_size": [128, 128],
            },
        }
        self._infos = {
            _PREFIX + name: TensorInfo(_PREFIX + name, dtype, shape, "synthetic-full-shape.safetensors")
            for name, shape, dtype in (
                ("input_layernorm.weight", (5120,), DType.BFLOAT16),
                ("linear_attn.in_proj_qkv.weight", (10240, 5120), DType.FLOAT8_E4M3FN),
                ("linear_attn.in_proj_qkv.weight_scale_inv", (80, 40), DType.FLOAT32),
                ("linear_attn.in_proj_z.weight", (6144, 5120), DType.FLOAT8_E4M3FN),
                ("linear_attn.in_proj_z.weight_scale_inv", (48, 40), DType.FLOAT32),
                ("linear_attn.in_proj_b.weight", (48, 5120), DType.BFLOAT16),
                ("linear_attn.in_proj_a.weight", (48, 5120), DType.BFLOAT16),
                ("linear_attn.conv1d.weight", (10240, 1, 4), DType.BFLOAT16),
                ("linear_attn.A_log", (48,), DType.FLOAT32),
                ("linear_attn.dt_bias", (48,), DType.FLOAT32),
                ("linear_attn.norm.weight", (128,), DType.BFLOAT16),
                ("linear_attn.out_proj.weight", (5120, 6144), DType.FLOAT8_E4M3FN),
                ("linear_attn.out_proj.weight_scale_inv", (40, 48), DType.FLOAT32),
                ("post_attention_layernorm.weight", (5120,), DType.BFLOAT16),
                ("mlp.gate_proj.weight", (17408, 5120), DType.FLOAT8_E4M3FN),
                ("mlp.gate_proj.weight_scale_inv", (136, 40), DType.FLOAT32),
                ("mlp.up_proj.weight", (17408, 5120), DType.FLOAT8_E4M3FN),
                ("mlp.up_proj.weight_scale_inv", (136, 40), DType.FLOAT32),
                ("mlp.down_proj.weight", (5120, 17408), DType.FLOAT8_E4M3FN),
                ("mlp.down_proj.weight_scale_inv", (40, 136), DType.FLOAT32),
            )
        }
        self._infos[_EMBEDDING_KEY] = TensorInfo(
            _EMBEDDING_KEY,
            DType.BFLOAT16,
            (248320, 5120),
            "synthetic-embedding.safetensors",
        )

    @property
    def config(self):
        return self._config

    @property
    def keys(self):
        return tuple(sorted(self._infos))

    def tensor_info(self, key):
        return self._infos[key]

    def load_tensor(self, key, *, device="cpu"):
        info = self._infos[key]
        dtype = {
            DType.BFLOAT16: torch.bfloat16,
            DType.FLOAT32: torch.float32,
            DType.FLOAT8_E4M3FN: torch.float8_e4m3fn,
        }[info.dtype]
        if key.endswith("weight_scale_inv"):
            return torch.ones(info.shape, dtype=dtype, device=device)
        return torch.zeros(info.shape, dtype=dtype, device=device)


def test_qwen3_8_27b_sharded_codegen_uses_local_call_abi_and_explicit_boxing():
    compiled = Compiler().compile(
        Qwen35Layer0Importer(_ZeroFullShapeCheckpoint()).import_module()
    ).module

    assert compiled.metadata["launch_contract"]["cooperative_grid"] is True
    assert compiled.selection_map[
        "tir.input_norm.vectorized.compute"
    ].candidate_id == (
        "semantic.ntt.gather_reduce_norm_apply"
    )
    package = compiled.metadata["codegen_package_plan"]
    assert package["kind"] == "tir_call_graph"

    descriptor = describe_tir_package(compiled)
    calls = descriptor["render_calls"]
    assert len(calls) == 15
    assert sum(call["family"] == "distributed_boxing" for call in calls) == 1
    qkv = next(call for call in calls if call["call"].endswith(".qkv"))
    qkv_output = qkv["outputs"][0]["buffers"][0]["abi"]
    assert qkv_output["logical_shape"] == (1, 10240)
    assert qkv_output["local_capacity_shape"] == (1, 80)
    # The consumer has the same owner map, so the QKV backing stays compact
    # and the kernel iterates only its 80-element owner-local ABI.
    assert qkv_output["storage_kind"] == "compact_local"
    recurrent = next(call for call in calls if call["family"] == "gdn_recurrent")
    assert recurrent["workspaces"][0]["formal"] == "core_scratch"
    assert recurrent["workspaces"][0]["buffers"][0]["abi"]["logical_shape"] == (48, 128)

    source = render_tir_package(descriptor, "flagmega-test")
    compile(source, "generated_qwen3_8_layer.py", "exec")
    assert source.count("# tir.call ") == 15
    # Each selected PrimFunction is a real device function.  The entry does
    # not reinterpret DistributedType as an owner-selection predicate.
    entry = source[source.index("def flagmega_main(") :]
    assert "shard_id(" not in entry
    collective_calls = sum(
        call["execution_kind"] == "collective" for call in calls
    )
    scheduled_grid_barriers = sum(
        event["kind"] == "barrier" and event["scope"] == "grid"
        for event in descriptor["entry_events"]
    )
    scheduled_block_barriers = sum(
        event["kind"] == "barrier" and event["scope"] == "block"
        for event in descriptor["entry_events"]
    )
    assert entry.count("distributed_barrier(") == scheduled_grid_barriers
    assert entry.count("tl.debug_barrier()") == scheduled_block_barriers
    # Only Boxing owns a separate collective launch.  The two fused
    # gather-reduce norm kernels are synchronized-local kernels whose grid
    # synchronization is represented by the schedule events above.
    assert collective_calls == 1
    assert scheduled_grid_barriers >= collective_calls
    for call in calls:
        symbol = "_flagmega_main_call_" + str(calls.index(call)) + "_" + "".join(
            character.lower() if character.isalnum() or character == "_" else "_"
            for character in call["call"]
        ).strip("_")
        wrapper = _device_wrapper(source, symbol)
        if call["execution_kind"] != "local_shard":
            continue
        assert "if shard_index == 0" not in wrapper
    embedding_wrapper = _device_wrapper(
        source, "_flagmega_main_call_1_token_embedding"
    )
    # The embedding result is split over the 8x16 mesh and therefore writes
    # only its 5120 / 128 owner-local elements.
    assert "tl.range(0, 40, 16)" in embedding_wrapper
    assert "shard_index * 16" not in embedding_wrapper
    assert "distributed_barrier(" not in embedding_wrapper
    output_wrapper = _device_wrapper(
        source, "_flagmega_main_call_14_output_vectorized_compute"
    )
    assert "tl.range(0, 40, 256)" in output_wrapper
    assert "tl.program_id" not in output_wrapper
    assert "distributed_barrier(" not in output_wrapper


@pytest.mark.skipif(
    os.environ.get("FLAGMEGA_RUN_FULL_SHAPE") != "1",
    reason="set FLAGMEGA_RUN_FULL_SHAPE=1 to run the >3GB exact-shape H800 regression",
)
def test_qwen3_8_27b_fp8_exact_shape_single_entry_runs_twice(tmp_path):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    checkpoint = _ZeroFullShapeCheckpoint()
    imported = Qwen35Layer0Importer(checkpoint, revision="synthetic-zero-full-shape").import_module()
    compiled = Compiler().compile(imported).module
    artifact = write_artifact(
        compiled,
        tmp_path / "qwen3.8-27b-fp8-layer0",
        target="nvidia-sm90",
        checkpoint=checkpoint,
        emit_executable=True,
    )
    runtime = load_runtime(artifact, device="cuda:0")
    input_ids = torch.tensor([1], dtype=torch.int32, device="cuda:0")
    output = torch.empty((1, 5120), dtype=torch.bfloat16, device="cuda:0")
    state = runtime.create_state()

    runtime.prepare(input_ids, state, output=output)
    runtime.run_into(output, input_ids, state)
    runtime.run_into(output, input_ids, state)
    torch.cuda.synchronize()

    assert runtime.prepare_count == 1
    assert runtime.resource_report["spill_bytes"] == 0
    assert torch.equal(output, torch.zeros_like(output))
    assert torch.equal(state.convolution, torch.zeros_like(state.convolution))
    assert torch.equal(state.recurrent, torch.zeros_like(state.recurrent))

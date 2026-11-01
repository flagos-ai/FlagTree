# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Pairwise source accumulation survives packed normalization statistics."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_profile_qk_warp_hierarchy(tmp_path):
    from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
    from triton.flagmega.importer import MemoryCheckpoint, import_model, VLLM_AE10_INDUCTOR_LEVEL3
    from python.test.flagmega.importer.qwen3_5_moe.helpers import checkpoint, configuration

    config = configuration()
    config["text_config"]["head_dim"] = 256
    source = checkpoint(config)
    shapes = {"q_proj.weight": (1024, 16), "k_proj.weight": (256, 16),
              "v_proj.weight": (256, 16), "qkvg.weight": (1536, 16),
              "q_norm.weight": (256,), "k_norm.weight": (256,), "o_proj.weight": (16, 512)}
    infos = {key: replace(source.tensor_info(key), shape=shapes[key.split(".self_attn.")[1]])
             if ".self_attn." in key else source.tensor_info(key) for key in source.keys}
    profile = import_model(MemoryCheckpoint(config, infos), mode="prefill", num_tokens=32,
                           numerical_profile=VLLM_AE10_INDUCTOR_LEVEL3)
    apply = profile.node_map["prefill_attention_q_norm"]
    attrs = profile.node_map[apply.inputs[1]].attrs
    fixture = json.loads((Path(__file__).resolve().parents[4] / "data" /
                          "native_qk_norm_hierarchical_tree.json").read_text())

    class Norm(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type("bfloat16", (1, 1, 256)))
            weight = self.input("weight", fm.tensor_type("float32", (256,)))
            zero = self.input("zero", weight.type)
            stats = fm.F.nn.norm_stats(x, **attrs)
            result = fm.F.nn.norm_apply(x, stats, weight, zero, **apply.attrs)
            self.function("main", (x, weight, zero), (result,))

    module = Norm(dialect="high_level", stage="imported", entry="main").build()
    values = {"x": torch.tensor([[fixture["input"]]], dtype=torch.bfloat16, device="cuda"),
              "weight": torch.tensor(fixture["weight"], dtype=torch.float32, device="cuda"),
              "zero": torch.zeros(256, device="cuda")}
    expected = torch.tensor([[fixture["output"]]], dtype=torch.bfloat16, device="cuda")
    eager = TorchEvaluator(DictWeightResolver({})).run(module, values)[0]
    torch.testing.assert_close(eager, expected, rtol=0, atol=0)
    compiled = Compiler().compile(module).module
    runtime = load(write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90",
                                 emit_executable=True), device="cuda:0")
    try:
        args = tuple(values[name] for name in ("x", "weight", "zero"))
        runtime.prepare(*args)
        actual = runtime.run(*args)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_qk_norm_pair_reduction(tmp_path):
    fixture = json.loads((Path(__file__).resolve().parents[4] / "data" /
                          "native_qk_norm_pair_reduction.json").read_text())

    class Norm(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type("bfloat16", (1, 256)))
            weight = self.input("weight", fm.tensor_type("float32", (256,)))
            zero = self.input("zero", weight.type)
            stats = fm.F.nn.norm_stats(x, axis=-1, use_mean=False, reduction_group_size=2)
            result = fm.F.nn.norm_apply(x, stats, weight, zero, axis=-1, epsilon=1e-6,
                                       use_mean=False, round_before_scale=False, output_dtype="bfloat16")
            self.function("main", (x, weight, zero), (result,))

    module = Norm(dialect="high_level", stage="imported", entry="main").build()
    module = Compiler().compile(module).module
    runtime = load(write_artifact(module, tmp_path / "artifact", target="nvidia-sm90",
                                 emit_executable=True), device="cuda:0")
    x = torch.tensor([fixture["input"]], dtype=torch.bfloat16, device="cuda")
    weight = torch.tensor(fixture["weight"], dtype=torch.float32, device="cuda")
    expected = torch.tensor([fixture["output"]], dtype=torch.bfloat16, device="cuda")
    try:
        runtime.prepare(x, weight, torch.zeros_like(weight))
        actual = runtime.run(x, weight, torch.zeros_like(weight))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()

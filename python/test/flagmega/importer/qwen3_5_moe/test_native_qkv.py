# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import CheckpointWeightResolver, TorchEvaluator
from triton.flagmega.importer import Qwen35MoeImporter
from triton.flagmega.ir.ops.nn._gdn_state import create_gdn_state
from triton.flagmega.ir.ops.nn._paged_attention_state import create_paged_attention_state
from triton.flagmega.ir.print_weights import WeightPrintAnalysis
from python.test.flagmega.importer.qwen3_5_moe.helpers import checkpoint


@pytest.mark.parametrize("tokens", [1, 3])
def test_import_exposes_native_qkv_and_gate_without_activation_slices(tokens, tmp_path):
    importer = Qwen35MoeImporter(checkpoint(), layer=1, execution_phase="decode" if tokens == 1 else "prefill",
                                num_tokens=tokens)
    module = importer.import_module()
    projections = [node for node in module.nodes if node.op == "nn.qkv_parallel_linear"]
    assert len(projections) == 1
    assert [tuple(d.fixed_value for d in field.shape) for field in projections[0].type.fields] == [
        (tokens, 16), (tokens, 8), (tokens, 8),
    ]
    constants = WeightPrintAnalysis.analyze(module).values
    slices = [node for node in module.nodes if node.op == "tensors.slice" and node.id not in constants]
    assert all(node.id == "prefill_last_hidden" for node in slices)
    assert fm.load_module(fm.emit_module(module, tmp_path / "module.py")) == module


def test_query_gate_weight_deinterleave_preserves_values_and_state():
    source = checkpoint(with_values=True)
    importer = Qwen35MoeImporter(source, layer=1, block_size=2, num_blocks=2)
    module = importer.import_module()
    assert any(node.op == "nn.qkv_parallel_linear" for node in module.nodes)
    _, trace = TorchEvaluator(CheckpointWeightResolver(source)).run_with_trace(module, {
        "input_ids": torch.tensor([7], dtype=torch.int32),
        "gated_delta_net_state": create_gdn_state(importer.gdn_config),
        "paged_attention_state": create_paged_attention_state(importer.paged_config),
    })
    hidden = trace["decode_attention_input_norm"]
    weight = source.load_tensor("model.language_model.layers.1.self_attn.q_proj.weight")
    expected = (hidden @ weight.T).reshape(1, 2, 2, 8)
    qkv = trace["decode_attention_qkv"]
    torch.testing.assert_close(qkv[0].reshape(1, 2, 8), expected[:, :, 0], rtol=0, atol=0)
    torch.testing.assert_close(trace["decode_attention_gate"], expected[:, :, 1].reshape(1, 16), rtol=0, atol=0)


@pytest.mark.parametrize("profile", ["nncase", "vllm-ae10e855a-inductor-level3"])
def test_native_projection_matches_four_way_reference_across_state_updates(profile):
    from triton.flagmega.importer.model import apply_numerical_profile

    source = checkpoint(with_values=True)
    importers = [Qwen35MoeImporter(source, block_size=2, num_blocks=3, fused_qkvg_projection=fused)
                 for fused in (False, True)]
    evaluators = [TorchEvaluator(CheckpointWeightResolver(source)) for _ in importers]
    modules = [apply_numerical_profile(importer.import_module(), profile) for importer in importers]
    states = [(create_gdn_state(importer.gdn_config), create_paged_attention_state(importer.paged_config))
              for importer in importers]
    for token in (1, 4, 2, 5):
        results = [evaluator.run(module, {"input_ids": torch.tensor([token], dtype=torch.int32),
                                         "gated_delta_net_state": gdn, "paged_attention_state": paged})
                   for evaluator, module, (gdn, paged) in zip(evaluators, modules, states)]
        torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
        torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)
        for layer in range(2):
            torch.testing.assert_close(states[0][0].convolution_layer(layer), states[1][0].convolution_layer(layer),
                                       rtol=0, atol=0)
            torch.testing.assert_close(states[0][0].recurrent_layer(layer), states[1][0].recurrent_layer(layer),
                                       rtol=0, atol=0)


def test_selected_native_qkv_and_gate_execute_exact_binary_fraction_inputs(tmp_path):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
    from triton.flagmega.runtime import load

    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type("bfloat16", (1, 64)), id="x")
            weights = [self.weight(name, fm.tensor_type("bfloat16", (64, size)), source="memory", key=name, id=name)
                       for name, size in (("q", 64), ("k", 32), ("v", 32), ("gate", 64))]
            none = fm.F.builtin.none()
            qkv = fm.F.nn.qkv_parallel_linear(x, *weights[:3], *(none,) * 9,
                                              num_heads=2, num_kv_heads=1, output_data_type="bfloat16", name="qkv")
            gate = fm.F.math.matmul(x, weights[3], name="gate_result")
            self.function("main", (x,), (*fm.F.tensors.get_items(qkv, 0, 1, 2), gate))

    compiled = Compiler().compile(Graph(dialect="nn", stage="decomposed", entry="main").build()).module
    implementations = [k.dispatch.microkernel.implementation for k in compiled.kernel_definitions
                       if "qkv_parallel_linear" in k.dispatch.semantic_op]
    assert implementations == ["tir.qkv_parallel_linear.packed_fused_gemv"]
    generator = torch.Generator().manual_seed(917)
    weights = {name: (torch.randint(-3, 4, (64, size), generator=generator).float() / 8).bfloat16()
               for name, size in (("q", 64), ("k", 32), ("v", 32), ("gate", 64))}
    checkpoint = MemoryCheckpoint({}, {name: TensorInfo(name, fm.DType.BFLOAT16, tuple(value.shape), "memory")
                                       for name, value in weights.items()}, weights)
    runtime = load(write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", checkpoint=checkpoint,
                                 emit_executable=True), device="cuda:0")
    x = (torch.randint(-3, 4, (1, 64), generator=generator).float() / 8).bfloat16().cuda()
    actual = tuple(torch.empty((1, size), dtype=torch.bfloat16, device="cuda") for size in (64, 32, 32, 64))
    runtime.prepare(x, *actual)
    runtime.run_into(x, *actual)
    for result, weight in zip(actual, weights.values(), strict=True):
        torch.testing.assert_close(result, (x.float() @ weight.float().cuda()).bfloat16(), rtol=0, atol=0)

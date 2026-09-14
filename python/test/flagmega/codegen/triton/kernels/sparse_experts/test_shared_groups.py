# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.importer.qwen3_5_moe.config import Qwen35MoeConfig
from triton.flagmega.importer.qwen3_5_moe.decoder import build_moe
from python.test.flagmega.importer.qwen3_5_moe.helpers import configuration
from python.test.flagmega.codegen.triton.kernels.sparse_experts.helpers import execute_and_reference


def imported_moe(torch, tokens, shared_width):
    config = configuration()
    config["text_config"]["shared_expert_intermediate_size"] = shared_width
    config = Qwen35MoeConfig.parse(config)
    shapes = {
        "mlp.gate.weight": (4, 16),
        "mlp.experts.gate_proj": (4, 8, 16),
        "mlp.experts.up_proj": (4, 8, 16),
        "mlp.experts.down_proj": (4, 16, 8),
        "mlp.shared_expert_gate.weight": (1, 16),
        "mlp.shared_expert.gate_proj.weight": (shared_width, 16),
        "mlp.shared_expert.up_proj.weight": (shared_width, 16),
        "mlp.shared_expert.down_proj.weight": (16, shared_width),
    }

    class Graph(fm.Module):
        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (tokens, 16)), id="value")
            weights = {name: self.weight(name, fm.tensor_type("bfloat16", shape),
                                         source="model.safetensors", key=name, id=name)
                       for name, shape in shapes.items()}
            result = build_moe(value, weights, config, prefix="moe")
            self.function("main", (value,), (result,))

    module = Graph(dialect="nn", stage="imported", entry="main").build()
    generator = torch.Generator().manual_seed(183)
    weights = {name: (torch.randn(shape, generator=generator) * 0.25).bfloat16() for name, shape in shapes.items()}
    infos = {name: TensorInfo(name, fm.DType.BFLOAT16, shape, "model.safetensors") for name, shape in shapes.items()}
    checkpoint = MemoryCheckpoint({}, infos, weights)
    return module, checkpoint


@pytest.mark.parametrize("tokens", [1, 3])
@pytest.mark.parametrize("shared_width", [8, 12])
def test_imported_moe_keeps_shared_group_independent_on_device(tmp_path, tokens, shared_width):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    module, checkpoint = imported_moe(torch, tokens, shared_width)
    output, expected, _ = execute_and_reference(module, tmp_path, torch, checkpoint=checkpoint)
    torch.testing.assert_close(output, expected, rtol=0.016, atol=0.03125)

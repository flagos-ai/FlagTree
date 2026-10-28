# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Broadcast FP8 scales are indexed by the weight's global block coordinates."""

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.ir.ops.nn.sparse_experts_down import SparseExpertsDown
from triton.flagmega.ir.ops.nn.sparse_experts_gate_up import SparseExpertsGateUp

from python.test.flagmega.codegen.triton.kernels.sparse_experts.helpers import execute_and_reference, stage_module
from python.test.flagmega.devices import is_nvidia_sm90


def _module(definition, block, split_k, cyclic=False):
    base = stage_module(definition, dtype="bfloat16", weight_dtype="float8_e4m3fn",
                        block_scaled=True, weight_block=block, hidden=256, intermediate=256, tokens=1)
    placement = fm.Placement((2, 2), "yx", "bb")
    broadcast = fm.SBP.broadcast()
    n = (fm.SBP.split_block_cyclic((0,), 32) if cyclic else
         fm.SBP.split_contiguous((0,), 128))
    k = fm.SBP.split_contiguous((1,), 128) if split_k else broadcast
    policies = {
        "gate_weight": (broadcast, n, broadcast),
        "up_weight": (broadcast, n, broadcast),
        "down_weight": (broadcast, n, k),
        "activations": (broadcast, broadcast, k),
    }

    class Graph(fm.Module):
        def forward(self):
            inputs = []
            for parameter in definition.input_parameters:
                tensor = base.node_map[parameter.name].type
                value_type = fm.DistributedType(
                    tensor, policies.get(parameter.name, (broadcast,) * tensor.rank), placement)
                inputs.append(self.input(parameter.name, value_type, id=parameter.name))
            result = definition.construct(*inputs, weight_block_n=block, weight_block_k=block,
                                          name="sharded_expert")
            result = fm.F.distributed.boxing(result, result.type.tensor)
            self.function("main", tuple(inputs), (result,))

    return Graph(dialect="nn", stage="frozen_constants", entry="main",
                 metadata={"auto_distribution": {"placement": placement.to_data()}}).build()


@pytest.mark.parametrize("definition,split_k", [(SparseExpertsGateUp, False),
                                               (SparseExpertsDown, False), (SparseExpertsDown, True)])
@pytest.mark.parametrize("block", (16, 128))
@pytest.mark.parametrize("cyclic", (False, True))
def test_sharded_fp8_weights_use_global_scale_blocks(tmp_path, definition, split_k, block,
                                                    cyclic):
    torch = pytest.importorskip("torch")
    if not is_nvidia_sm90():
        pytest.skip("SM90 CUDA required")
    module = _module(definition, block, split_k, cyclic)
    values = {}
    activation = definition.input_parameters[0].name
    for name in module.function_map[module.entry].parameters:
        tensor = fm.logical_type(module.node_map[name].type)
        shape = tuple(dim.fixed_value for dim in tensor.shape)
        if name == "router_expert_ids":
            value = torch.zeros(shape, dtype=torch.int32)
        elif name == activation or name.endswith("input_scale"):
            value = torch.ones(shape, dtype=getattr(torch, tensor.dtype.value))
        elif name.endswith("proj_scale"):
            row = torch.arange(shape[1])[:, None]
            column = torch.arange(shape[2])[None, :]
            value = (2. ** ((row + column) % 3)).expand(shape).contiguous()
        else:
            value = torch.zeros(shape)
            value[..., ::128] = 1.0
            value = value.to(getattr(torch, tensor.dtype.value))
        values[name] = value
    actual, expected, runtime = execute_and_reference(module, tmp_path, torch, overrides=values)
    try:
        torch.testing.assert_close(actual, expected, rtol=.005, atol=1e-5)
        source = (tmp_path / "artifact/generated_kernels.py").read_text()
        assert f"sparse_experts_{'gate_up' if definition is SparseExpertsGateUp else 'down'}/simt" in source
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()

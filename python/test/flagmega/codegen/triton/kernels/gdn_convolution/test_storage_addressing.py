# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A channel shard's logical origin is not its physical pointer offset."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.ir.ops.nn._gdn_state import GatedDeltaNetStateConfig, create_gdn_state
from triton.flagmega.ir.ops.nn.gdn_convolution import gated_delta_net_convolution
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule


@pytest.mark.parametrize("local_input", [False, True])
@pytest.mark.parametrize("local_weight", [False, True])
@pytest.mark.parametrize("local_result", [False, True])
@pytest.mark.parametrize("tokens", [1, 3])
def test_channel_shards_respect_each_operand_storage(tmp_path, local_input, local_weight, local_result, tokens):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    config = GatedDeltaNetStateConfig(1, 2, 4, 8, 8, 4, 32)
    channels = config.conv_dim
    placement = fm.Placement((2, 4), "yx", "bb")
    broadcast = fm.SBP.broadcast()
    split = fm.SBP.split_contiguous((0, 1))
    qkv_type = fm.tensor_type("bfloat16", (tokens, channels))
    weight_type = fm.tensor_type("bfloat16", (channels, 1, 4))
    qkv_distributed = fm.DistributedType(qkv_type, (broadcast, split), placement)
    weight_distributed = fm.DistributedType(weight_type, (split, broadcast, broadcast), placement)

    class Graph(fm.Module):

        def forward(self):
            qkv = self.input("qkv", qkv_type if local_input else qkv_distributed, id="qkv")
            weight = self.input("weight", weight_type if local_weight else weight_distributed, id="weight")
            state = self.input("state", config.ref_type, id="state")
            source = fm.F.distributed.force_boxing(qkv, qkv_distributed) if local_input else qkv
            coefficients = fm.F.distributed.force_boxing(weight, weight_distributed) if local_weight else weight
            conv = fm.F.nn.gated_delta_net_convolution(source, state, coefficients, conv_kernel_size=4, name="conv")
            result = fm.F.tensors.get_item(conv, 0)
            if local_result:
                result = fm.F.distributed.force_boxing(result, qkv_type)
            self.function("main", (qkv, weight, state), (result, state))

    module = Graph(dialect="distributed", stage="frozen_constants", entry="main",
                   metadata={"auto_distribution": {"placement": placement.to_data()}}).build()
    compiled = Compiler().compile(module).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    state = create_gdn_state(config, device="cuda:0")
    history = (torch.arange(channels * 3, device="cuda").reshape(channels, 3) % 19 - 9).bfloat16() / 8
    state.update_convolution_layer(history)
    state.recurrent.fill_(17)
    expected_state = state.clone()
    qkv = (torch.arange(tokens * channels, device="cuda").reshape(tokens, channels) + 1).bfloat16() / 32
    weight = ((torch.arange(channels * 4, device="cuda").reshape(channels, 1, 4) % 11 - 5).bfloat16() / 8)
    output = torch.full_like(qkv, float("nan"))
    values = {"qkv": qkv, "weight": weight}
    state_buffers = dict(runtime.buffer_plan.function_map["main"].parameters)["state"]
    fields = dict(zip(state_buffers, (state.convolution, state.recurrent)))
    arguments = tuple(output if spec["role"] == "result" else (
        fields[spec["buffer"]] if spec["value"] == "state" else values[spec["value"]])
                      for spec in runtime.external_arguments)
    GeneratedTirCallGraphModule.prepare(runtime, *arguments)
    for _ in range(3):
        expected, _ = gated_delta_net_convolution(qkv=qkv, state=expected_state, conv_weight=weight, conv_kernel_size=4,
                                                  torch=torch)
        output.fill_(float("nan"))
        GeneratedTirCallGraphModule.run_into(runtime, *arguments)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, rtol=.008, atol=.008)
        torch.testing.assert_close(state.convolution, expected_state.convolution, rtol=0, atol=0)
        torch.testing.assert_close(state.recurrent, expected_state.recurrent, rtol=0, atol=0)
        qkv.add_(.125)

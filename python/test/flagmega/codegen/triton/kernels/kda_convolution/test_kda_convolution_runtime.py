# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The generated convolution must address q/k/v per (part, head, channel).

The flat ``[3*heads*head_dim, kernel]`` layout this replaces had one channel
axis on which head ``h`` owned three discontiguous ranges, so no SBP policy
could describe per-owner head ownership.  Part-separate buffers plus a
``[3, heads, head_dim, ...]`` state and weight keep each part's head axis
contiguous.  These tests pin the emitted addressing and prove the arithmetic
is bit-identical to the flat form.
"""

import pathlib

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule

HEADS, DIM, KERNEL = 2, 4, 4
NAMES = ("q", "k", "v", "state", "weight")


def _module():
    projection = fm.tensor_type("bfloat16", (1, HEADS, DIM))
    types = (projection, projection, projection,
             fm.tensor_type("bfloat16", (3, HEADS, DIM, KERNEL - 1)),
             fm.tensor_type("bfloat16", (3, HEADS, DIM, KERNEL)))

    class Graph(fm.Module):
        def forward(self):
            operands = tuple(self.input(name, value_type, id=name)
                             for name, value_type in zip(NAMES, types))
            values = fm.F.nn.kda_convolution(*operands, kernel_size=KERNEL, name="conv")
            items = fm.F.tensors.get_items(values, 0, 1, 2, 3, name_prefix="conv")
            self.function("main", operands, items)

    return Graph(dialect="high_level", stage="imported", entry="main").build()


def _artifact(tmp_path: pathlib.Path):
    return write_artifact(Compiler().compile(_module()).module, tmp_path / "kda-conv",
                          target="nvidia-sm90", emit_executable=True)


def _values(torch, *, device="cpu"):
    generator = torch.Generator().manual_seed(13)

    def normal(*shape):
        return torch.randn(shape, generator=generator).bfloat16()

    values = {
        "q": normal(1, HEADS, DIM),
        "k": normal(1, HEADS, DIM),
        "v": normal(1, HEADS, DIM),
        "state": normal(3, HEADS, DIM, KERNEL - 1),
        "weight": normal(3, HEADS, DIM, KERNEL),
    }
    return {name: value.to(device) for name, value in values.items()}


def _reference(torch, values):
    current = torch.stack([values[name].float().squeeze(0) for name in ("q", "k", "v")])
    state, weight = values["state"].float(), values["weight"].float()
    convolved = torch.nn.functional.silu(
        (state * weight[..., : KERNEL - 1]).sum(dim=-1) + weight[..., KERNEL - 1] * current)
    shifted = torch.cat([state[..., 1:], current.unsqueeze(-1)], dim=-1)
    return convolved.bfloat16(), shifted.bfloat16()


def test_generated_source_keeps_the_parts_on_separate_buffers(tmp_path):
    source = (_artifact(tmp_path) / "generated_kernels.py").read_text("utf-8")

    assert "# flagmega-kernel: kda_convolution/local platform=generic" in source
    body = source.split("# flagmega-kernel: kda_convolution/local")[1].split("@triton.jit")[1]
    # One unrolled block per part, each naming its own result buffer, and each
    # reading the shared state/weight at its own literal part index.
    for part in range(3):
        assert f"# part {part} of q/k/v" in body
        assert f"(({part})) * " in body
    results = {line.split("tl.store(")[1].split(" ")[0]
               for line in body.splitlines() if "tl.store(" in line}
    assert len(results) == 4, f"expected three result buffers plus the state, got {results}"


def test_generated_kernel_matches_the_flat_channel_arithmetic(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    runtime = load(_artifact(tmp_path), device="cuda:0")
    values = _values(torch, device="cuda")
    outputs = {
        **{f"conv_{part}": torch.empty((1, HEADS, DIM), dtype=torch.bfloat16, device="cuda")
           for part in range(3)},
        "conv_3": torch.empty((3, HEADS, DIM, KERNEL - 1), dtype=torch.bfloat16, device="cuda"),
    }
    arguments = tuple(outputs[spec["value"]] if spec["role"] == "result" else values[spec["value"]]
                      for spec in runtime.external_arguments)
    GeneratedTirCallGraphModule.prepare(runtime, *arguments)
    GeneratedTirCallGraphModule.run_into(runtime, *arguments)
    torch.cuda.synchronize()

    expected, expected_state = _reference(torch, values)
    for part in range(3):
        torch.testing.assert_close(outputs[f"conv_{part}"], expected[part].unsqueeze(0),
                                   rtol=0, atol=0)
    torch.testing.assert_close(outputs["conv_3"], expected_state, rtol=0, atol=0)


def test_the_head_major_layout_is_a_pure_relabeling(tmp_path):
    """Guard against a vacuous pass: the flat view must hold the same elements."""
    torch = pytest.importorskip("torch")
    values = _values(torch)
    convolved, shifted = _reference(torch, values)

    flat_state = values["state"].float().reshape(3 * HEADS * DIM, KERNEL - 1)
    flat_weight = values["weight"].float().reshape(3 * HEADS * DIM, KERNEL)
    flat_current = torch.stack(
        [values[name].float().squeeze(0) for name in ("q", "k", "v")]).reshape(3 * HEADS * DIM)
    flat = torch.nn.functional.silu(
        (flat_state * flat_weight[:, : KERNEL - 1]).sum(dim=1)
        + flat_weight[:, KERNEL - 1] * flat_current).bfloat16()

    assert torch.equal(convolved.reshape(3 * HEADS * DIM), flat)
    assert torch.equal(shifted.reshape(3 * HEADS * DIM, KERNEL - 1)[:, : KERNEL - 2],
                       flat_state[:, 1:].bfloat16())

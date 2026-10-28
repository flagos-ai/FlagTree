# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The selected per-thread grouping is independent of pointer alignment."""

from dataclasses import replace

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import CodegenError, IRVerificationError
from triton.flagmega.runtime import load
from triton.flagmega.targets import NvidiaSm90Target
from python.test.flagmega.devices import is_nvidia_sm90


def _reference(value):
    rows, width = value.shape
    padded = torch.zeros((rows, max(2048, 1 << (width - 1).bit_length())))
    padded[:, :width] = value.float()
    squares = padded.square().reshape(rows, -1, 8)
    subtotal = squares[:, :, 0]
    for element in range(1, 8):
        subtotal = subtotal + squares[:, :, element]
    lanes = torch.arange(32)
    subtotal = subtotal.reshape(rows, -1, 32)
    for offset in (16, 8, 4, 2, 1):
        subtotal = subtotal + subtotal[:, :, lanes ^ offset]
    warps = subtotal[:, :, 0]
    indices = torch.arange(warps.shape[1])
    offset = warps.shape[1] // 2
    while offset:
        warps = warps + warps[:, indices ^ offset]
        offset //= 2
    return warps[:, 0].reshape(1, rows, 1)


def _module(width, packed):
    class Graph(fm.Module):
        def forward(self):
            dtype = fm.vector_type("bfloat16", (8,)) if packed else "bfloat16"
            value = self.input("value", fm.tensor_type(dtype, (2, width // 8 if packed else width)), id="value")
            stats = fm.F.nn.norm_stats(value, axis=-1, use_mean=False)
            self.function("main", (value,), (stats,))

    return Graph(dialect="high_level", stage="imported", entry="main").build()


@pytest.mark.parametrize("layout", ["unknown", True, 8, []])
def test_invalid_contiguous_reduction_layout_is_rejected(tmp_path, layout):
    compiler = Compiler()
    module = compiler.compile(_module(512, True)).module
    with pytest.raises((CodegenError, IRVerificationError), match="reduction_layout"):
        definitions = tuple(
            replace(
                item,
                dispatch=replace(
                    item.dispatch,
                    microkernel=replace(
                        item.dispatch.microkernel,
                        parameters={**item.dispatch.microkernel.parameters, "reduction_layout": layout},
                    ),
                ),
            )
            if item.dispatch.microkernel.family == "norm_stats"
            else item
            for item in module.kernel_definitions
        )
        write_artifact(
            replace(module, kernel_definitions=definitions),
            tmp_path / "artifact",
            target="nvidia-sm90",
            emit_executable=True,
        )


@pytest.mark.parametrize("width,packed", [(1536, True), (512, True), (1031, False)])
def test_contiguous_eight_reduction_preserves_order_for_misaligned_inputs(tmp_path, width, packed):
    if not is_nvidia_sm90():
        pytest.skip("requires SM90")
    host = torch.randn((2, width), generator=torch.Generator().manual_seed(0)).bfloat16()
    expected = _reference(host).cuda()
    storage = torch.empty(host.numel() + 1, dtype=torch.bfloat16, device="cuda")
    value = storage[1:]
    value.copy_(host.flatten())
    value = value.reshape(2, width // 8, 8) if packed else value.reshape(2, width)
    assert value.data_ptr() % 16 == 2
    base = NvidiaSm90Target().triton_implementation_model
    catalog = replace(
        base,
        implementations=tuple(
            replace(
                item,
                parameters={
                    **item.parameters,
                    "block_size": 2048,
                    "compute_num_warps": 8,
                    "reduction_layout": "contiguous_8",
                },
            )
            if item.family == "norm_stats"
            else item
            for item in base.implementations
        ),
    )
    compiler = Compiler()
    compiler.target = NvidiaSm90Target(triton_implementation_model=catalog)
    module = compiler.compile(_module(width, packed)).module
    restored = fm.load_module(fm.emit_module(module, tmp_path / "selected.py"))
    assert restored.semantic_hash == module.semantic_hash
    artifact = write_artifact(restored, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    try:
        output = torch.empty_like(expected)
        runtime.prepare(value, output=output)
        runtime.run_into(output, value)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()

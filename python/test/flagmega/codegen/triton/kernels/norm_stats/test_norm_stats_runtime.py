# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load


def _module(shape, *, axis, use_mean):
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(
                dialect="high_level", stage="imported", entry="main"
            )

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", shape))
            stats = fm.F.nn.norm_stats(
                value, axis=axis, use_mean=use_mean, name="stats"
            )
            self.function("main", (value,), (stats,))

    return Graph().build()


@pytest.mark.parametrize(
    "shape,axis,use_mean",
    (
        ((2, 3, 5), 1, True),
        ((2, 3, 16), -1, False),
    ),
)
def test_generic_norm_stats_compiles_without_a_collective_private_workspace(
    tmp_path, shape, axis, use_mean,
):
    module = Compiler().compile(
        _module(shape, axis=axis, use_mean=use_mean)
    ).module
    artifact = write_artifact(
        module,
        tmp_path / f"axis-{axis}-mean-{use_mean}",
        target="nvidia-sm90",
        emit_executable=True,
    )
    source = (artifact / "generated_kernels.py").read_text("utf-8")
    calls = artifact.joinpath("artifact.json").read_text("utf-8")

    assert "# flagmega-kernel: norm_stats/local platform=generic" in source
    assert "for norm_outer_index in tl.range" in source
    assert "norm_reduction_offsets" in source
    assert "partial_stats" not in source
    assert '"kind": "tir_call_graph/v1"' in calls


@pytest.mark.parametrize(
    "shape,axis,use_mean",
    (
        ((2, 3, 5), 1, True),
        ((2, 3, 16), -1, False),
    ),
)
def test_generic_norm_stats_matches_torch_on_h800(
    tmp_path, shape, axis, use_mean,
):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    module = Compiler().compile(
        _module(shape, axis=axis, use_mean=use_mean)
    ).module
    artifact = write_artifact(
        module,
        tmp_path / f"runtime-axis-{axis}-mean-{use_mean}",
        target="nvidia-sm90",
        emit_executable=True,
    )
    runtime = load(artifact, device="cuda:0")
    value = torch.randn(shape, dtype=torch.bfloat16, device="cuda:0")
    runtime.prepare(value)
    actual = runtime.run(value)
    torch.cuda.synchronize()

    normalized_axis = axis + len(shape) if axis < 0 else axis
    reduction_axes = tuple(range(normalized_axis, len(shape)))
    source = value.float()
    square_sum = source.square().sum(dim=reduction_axes, keepdim=True).unsqueeze(0)
    if use_mean:
        expected = torch.cat((
            source.sum(dim=reduction_axes, keepdim=True).unsqueeze(0),
            square_sum,
        ), dim=0)
    else:
        expected = square_sum
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)

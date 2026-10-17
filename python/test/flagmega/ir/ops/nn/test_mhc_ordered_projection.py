# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The mHC coefficient/RMS FMA order is an explicit numerical contract."""

import ctypes

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.nn._mhc_ordered_projection import _fma32, ordered_stream_projection


def module(*, tokens=1, hidden=512, hc=3, splits=2, reduction="stream_fma_128", **attrs):
    count = hc * (hc + 2)

    class Graph(fm.Module):
        def forward(self):
            inputs = tuple(self.input(name, fm.tensor_type("float32", shape), id=name)
                           for name, shape in (("streams", (tokens, hc, hidden)), ("projection", (count, hc * hidden)),
                                               ("base", (count,)), ("scale", (3,))))
            result = fm.F.nn.mhc_pre(*inputs, epsilon=1e-5, sinkhorn_iterations=3, sinkhorn_epsilon=1e-6,
                                    projection_reduction=reduction, projection_splits=splits, **attrs)
            self.function("main", inputs, fm.F.tensors.get_items(result, 0, 1, 2))

    return Graph(dialect="high_level", stage="imported", entry="main").build()


def scalar_reference(streams, weights, splits):
    """libm fmaf provides an independent, single-round scalar oracle."""
    fma = ctypes.CDLL("libm.so.6").fmaf
    fma.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
    fma.restype = ctypes.c_float
    _, hc, hidden = streams.shape
    count = weights.shape[0]
    width = hidden // splits
    values, fn = streams.float().tolist()[0], weights.float().reshape(count, hc, hidden).tolist()
    dot, square = [], []
    for split in range(splits):
        totals = []
        for row in range(count + 1):
            lanes = [0.] * 128
            for lane in range(128):
                acc = 0.
                for iteration in range(width // 128):
                    k = split * width + iteration * 128 + lane
                    for stream in range(hc):
                        x = values[stream][k]
                        acc = fma(x if row == count else fn[row][stream][k], x, acc)
                lanes[lane] = acc
            warp_totals = []
            for warp in range(4):
                part = lanes[warp * 32:(warp + 1) * 32]
                for step in (16, 8, 4, 2, 1):
                    part = [ctypes.c_float(value + part[index ^ step]).value for index, value in enumerate(part)]
                warp_totals.append(part[0])
            total = 0.
            for value in warp_totals:
                total = ctypes.c_float(total + value).value
            totals.append(total)
        dot.append(totals[:-1])
        square.append(totals[-1])
    result = torch.zeros(count)
    square_sum = 0.
    for split in range(splits):
        result += torch.tensor(dot[split])
        square_sum = ctypes.c_float(square_sum + square[split]).value
    return result[None], torch.tensor([square_sum])


def test_projection_contract_round_trips_and_retains_packed_output(tmp_path):
    source = module(output_lanes=(8,), collapsed_dtype="float32", collapse_input_dtype="bfloat16",
                    collapse_reduction="sequential_fma")
    reloaded = fm.load_module(fm.emit_module(source, tmp_path / "ordered.py"))
    assert source.semantic_hash == reloaded.semantic_hash
    node, = [node for node in reloaded.nodes if node.op == "nn.mhc_pre"]
    assert node.attrs["projection_reduction"] == "stream_fma_128"
    assert node.attrs["projection_splits"] == 2
    assert node.type.fields[2].dtype == fm.VectorType(fm.DType.FLOAT32, (8,))


@pytest.mark.parametrize("splits", [0, -1, 3, True])
def test_projection_split_count_is_a_typed_power_of_two(splits):
    with pytest.raises(IRSchemaError, match="projection_splits"):
        module(splits=splits)


@pytest.mark.parametrize("kwargs", [{"hidden": 72}, {"hidden": 128, "splits": 2},
                                    {"reduction": "unknown"}, {"reduction": "parallel", "splits": 2}])
def test_invalid_algorithm_or_incomplete_split_is_rejected(kwargs):
    with pytest.raises(IRSchemaError, match="projection_reduction|projection_splits"):
        module(**kwargs)


def test_default_contract_does_not_add_attributes_to_legacy_ir():
    source = module(splits=1, reduction="parallel")
    node, = [node for node in source.nodes if node.op == "nn.mhc_pre"]
    assert "projection_reduction" not in node.attrs
    assert "projection_splits" not in node.attrs


def test_ordered_projection_matches_scalar_fmaf_and_differs_from_parallel_dot():
    generator = torch.Generator().manual_seed(179)
    streams = torch.randn((1, 3, 512), generator=generator)
    weights = torch.randn((15, 1536), generator=generator)
    actual = ordered_stream_projection(streams, weights, 2, torch)
    expected = scalar_reference(streams, weights, 2)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    assert torch.count_nonzero(actual[0] != streams.reshape(1, -1) @ weights.T)


def test_cpu_fma_does_not_double_round_tiny_addends_at_float32_midpoints():
    a = torch.tensor([1. + 2. ** -23, 1. + 2. ** -23])
    b = torch.tensor([1.5, -1.5])
    c = torch.tensor([-2. ** -60, 2. ** -60])
    fma = ctypes.CDLL("libm.so.6").fmaf
    fma.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
    fma.restype = ctypes.c_float
    expected = torch.tensor([fma(x, y, z) for x, y, z in zip(a.tolist(), b.tolist(), c.tolist(), strict=True)])
    assert not torch.equal((a.double() * b.double() + c.double()).float(), expected)
    torch.testing.assert_close(_fma32(a, b, c, torch), expected, atol=0, rtol=0)

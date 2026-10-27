# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace
from math import prod

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load
from python.test.flagmega.ir.ops.primitive_helpers import primitive_module
from python.test.flagmega.devices import require_nvidia_sm90


@pytest.mark.parametrize("shape,axes,keep_dims", [
    ((2, 7, 3), (-1, ), True),
    ((2, 7, 3), (0, 2), False),
    ((2, 7, 3), (2, 0), True),
    ((3, 513), (1, ), False),
    ((2, 7, 3), (0, 1, 2), False),
    ((3, 0, 7), (1, ), False),
    ((0, 7), (1, ), True),
    ((2, 7), (), False),
])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_reduce_sum_local_kernel_axes_empty_domains_and_fp32_accumulation(tmp_path, shape, axes, keep_dims, dtype):
    torch = pytest.importorskip("torch")
    require_nvidia_sm90()
    module = replace(
        primitive_module(fm.get_definition("math.reduce_sum"), (fm.tensor_type(dtype, shape), ), axes=axes,
                         keep_dims=keep_dims), stage="frozen_constants")
    compiled = Compiler().compile(module).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    value = torch.arange(prod(shape), device="cuda", dtype=torch.float32).remainder(31).reshape(shape) / 16
    value = value.to(getattr(torch, dtype))
    runtime.prepare(value)
    output = runtime.run(value)
    torch.cuda.synchronize()
    expected = value.float().sum(axes, keepdim=keep_dims).to(value.dtype) if axes else value
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("reduce_split", [False, True])
@pytest.mark.parametrize("keep_dims", [False, True])
def test_reduce_sum_preserves_rows_and_materializes_partial_only_through_boxing(tmp_path, reduce_split, keep_dims):
    torch = pytest.importorskip("torch")
    require_nvidia_sm90()
    placement = fm.Placement((2, 2), "xy", "bb")
    policies = (fm.SBP.split_block_cyclic((0, ), 2), fm.SBP.split_block_cyclic(
        (1, ), 2) if reduce_split else fm.SBP.broadcast())

    class Graph(fm.Module):

        def forward(self):
            value = self.input("value", fm.DistributedType(fm.tensor_type("float32", (8, 12)), policies, placement))
            partial = fm.F.math.reduce_sum(value, axes=(1, ), keep_dims=keep_dims, name="partial")
            result = fm.F.distributed.boxing(partial, partial.type.tensor)
            self.function("main", (value, ), (result, ))

    module = Graph(dialect="high_level", stage="frozen_constants", entry="main",
                   metadata={"auto_distribution": {"placement": placement.to_data()}}).build()
    assert module.node_map["partial"].type.partial == (fm.SBP.partial((1, )) if reduce_split else None)
    compiled = Compiler().compile(module).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    value = torch.arange(96, device="cuda", dtype=torch.float32).reshape(8, 12) / 8
    runtime.prepare(value)
    output = runtime.run(value)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, value.sum(1, keepdim=keep_dims), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("lanes", [(8,), (2, 4), (1,)])
def test_reduce_sum_packed_store_matches_scalar_kernel(tmp_path, dtype, lanes):
    torch = pytest.importorskip("torch")
    require_nvidia_sm90()
    values = torch.linspace(-3, 2, 3 * 4 * 72, device="cuda").reshape(3, 4, 72).to(getattr(torch, dtype))
    results = []
    for name, output_lanes in (("scalar", ()), ("packed", lanes)):
        module = primitive_module(fm.get_definition("math.reduce_sum"), (fm.tensor_type(dtype, (3, 4, 72)),),
                                  axes=(1,), keep_dims=False, output_lanes=output_lanes)
        compiled = Compiler().compile(module).module
        path = write_artifact(compiled, tmp_path / name, target="nvidia-sm90", emit_executable=True)
        assert "# flagmega-kernel: pack/" not in (path / "generated_kernels.py").read_text()
        runtime = load(path, device="cuda:0")
        runtime.prepare(values)
        results.append(runtime.run(values).reshape(3, 72).clone())
    torch.cuda.synchronize()
    torch.testing.assert_close(results[1], results[0], rtol=0, atol=0)
    # The GPU and Torch reduction trees may differ by an FP32 ULP; changing
    # the output layout must still match the scalar GPU kernel bit for bit.
    torch.testing.assert_close(results[1], values.float().sum(1).to(values.dtype),
                               rtol=2e-7 if dtype == "float32" else 0,
                               atol=1e-6 if dtype == "float32" else 0)


@pytest.mark.parametrize("axes,keep_dims", [((), False), ((1,), True)])
def test_reduce_sum_packed_identity_and_retained_dimensions(tmp_path, axes, keep_dims):
    torch = pytest.importorskip("torch")
    require_nvidia_sm90()
    module = primitive_module(fm.get_definition("math.reduce_sum"), (fm.tensor_type("bfloat16", (3, 4, 72)),),
                              axes=axes, keep_dims=keep_dims, output_lanes=(2, 4))
    path = write_artifact(Compiler().compile(module).module, tmp_path / "retained",
                          target="nvidia-sm90", emit_executable=True)
    runtime = load(path, device="cuda:0")
    values = torch.arange(3 * 4 * 72, device="cuda").remainder(31).reshape(3, 4, 72).bfloat16() / 8
    runtime.prepare(values)
    actual = runtime.run(values)
    expected = values.float().sum(axes, keepdim=keep_dims).bfloat16() if axes else values
    torch.cuda.synchronize()
    torch.testing.assert_close(actual.reshape(expected.shape), expected, rtol=0, atol=0)


@pytest.mark.parametrize("reduce_split", [False, True])
def test_reduce_sum_packed_output_preserves_owner_coordinates_and_partial(tmp_path, reduce_split):
    torch = pytest.importorskip("torch")
    require_nvidia_sm90()
    placement = fm.Placement((2, 2), "xy", "bb")
    policies = (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0,), 2) if reduce_split else fm.SBP.broadcast(),
                fm.SBP.split_block_cyclic((1,), 16))

    class Graph(fm.Module):
        def forward(self):
            value = self.input("value", fm.DistributedType(fm.tensor_type("float32", (3, 4, 72)), policies, placement))
            partial = fm.F.math.reduce_sum(value, axes=(1,), keep_dims=False, output_lanes=(2, 4), name="partial")
            result = fm.F.distributed.boxing(partial, partial.type.tensor)
            self.function("main", (value,), (result,))

    module = Graph(dialect="high_level", stage="frozen_constants", entry="main",
                   metadata={"auto_distribution": {"placement": placement.to_data()}}).build()
    compiled = Compiler().compile(module).module
    path = write_artifact(compiled, tmp_path / "owners", target="nvidia-sm90", emit_executable=True)
    runtime = load(path, device="cuda:0")
    values = torch.arange(3 * 4 * 72, device="cuda").float().reshape(3, 4, 72) / 8
    runtime.prepare(values)
    output = runtime.run(values)
    torch.cuda.synchronize()
    torch.testing.assert_close(output.reshape(3, 72), values.sum(1), rtol=0, atol=0)

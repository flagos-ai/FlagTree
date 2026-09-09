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
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
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
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
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

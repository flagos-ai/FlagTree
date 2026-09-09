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


@pytest.mark.parametrize("shape,axis,split", [
    ((8, 37), -1, False),
    ((7, 3, 5), 0, False),
    ((2, 7, 3), 1, False),
    ((8, 513), 1, True),
    ((8, 0), 1, False),
    ((0, 7), 1, False),
])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_softmax_local_kernel_arbitrary_axis_and_owners(tmp_path, shape, axis, split, dtype):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    value_type = fm.tensor_type(dtype, shape)
    metadata = {}
    if split:
        placement = fm.Placement((2, 2), "xy", "bb")
        value_type = fm.DistributedType(value_type, (fm.SBP.split_block_cyclic((0, ), 2), fm.SBP.broadcast()),
                                        placement)
        metadata = {"auto_distribution": {"placement": placement.to_data()}}
    module = replace(primitive_module(fm.get_definition("nn.softmax"), (value_type, ), axis=axis),
                     stage="frozen_constants", metadata=metadata)
    compiled = Compiler().compile(module).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    value = ((torch.arange(prod(shape), device="cuda", dtype=torch.float32).reshape(shape) * 0.37).sin() * 10)
    value = value.to(getattr(torch, dtype))
    runtime.prepare(value)
    output = runtime.run(value)
    torch.cuda.synchronize()
    torch.testing.assert_close(output,
                               value.float().softmax(axis).to(value.dtype), rtol=2e-6 if dtype == "float32" else 8e-3,
                               atol=1e-7 if dtype == "float32" else 1e-5)

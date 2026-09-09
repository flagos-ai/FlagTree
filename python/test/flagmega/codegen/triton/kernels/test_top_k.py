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


@pytest.mark.parametrize("shape,axis,k,split", [
    ((3, 13), -1, 13, False),
    ((7, 3, 5), 0, 4, False),
    ((2, 7, 3), 1, 5, False),
    ((8, 513), 1, 8, True),
    ((8, 0), 1, 0, False),
    ((0, 7), 1, 3, False),
])
@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "int32", "int64"])
def test_top_k_local_kernel_stable_ties_and_exact_integer_values(tmp_path, shape, axis, k, split, largest, dtype):
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
    index_dtype = "int64" if largest else "int32"
    module = replace(
        primitive_module(fm.get_definition("tensors.top_k"), (value_type, ), k=k, axis=axis, largest=largest,
                         sorted=largest, index_dtype=index_dtype), stage="frozen_constants", metadata=metadata)
    compiled = Compiler().compile(module).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    value = torch.arange(prod(shape), device="cuda", dtype=torch.int64).remainder(11).reshape(shape)
    if dtype in {"int32", "int64"}:
        value += 2**(54 if dtype == "int64" else 25)
        value = value.to(getattr(torch, dtype))
        if value.numel():
            flat = value.view(-1)
            flat[::11], flat[1::11] = torch.iinfo(value.dtype).min, torch.iinfo(value.dtype).max
    else:
        value = value.float()
        if value.numel():
            flat = value.view(-1)
            flat[::11], flat[1::11], flat[2::11] = float("inf"), -float("inf"), float("nan")
            flat[3::11], flat[4::11] = -0., 0.
        value = value.to(getattr(torch, dtype))
    order = torch.argsort(value, dim=axis, descending=largest, stable=True).narrow(axis, 0, k)
    expected = value.gather(axis, order)
    output = torch.empty_like(expected)
    indices = torch.empty_like(order, dtype=getattr(torch, index_dtype))
    binding = runtime.buffer_plan.function_map[compiled.entry]
    buffers = {binding.parameters[0][1][0]: value}
    assert len(binding.outputs) == 1 and len(binding.outputs[0][1]) == 2
    buffers.update(zip(binding.outputs[0][1], (output, indices), strict=True))
    arguments = tuple(buffers[item["buffer"]] for item in runtime.external_arguments)
    runtime.prepare(*arguments)
    runtime.run_into(*arguments)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(indices, order.to(indices.dtype), rtol=0, atol=0)
    if value.is_floating_point():
        torch.testing.assert_close(torch.signbit(output), torch.signbit(expected), rtol=0, atol=0)

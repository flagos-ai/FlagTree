# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace
from math import prod

import pytest

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load
from python.test.flagmega.ir.ops.tensors.test_concat_local_contract import concat_module


@pytest.mark.parametrize("shapes,axis", [(((8, 3), (8, 5)), -1), (((8, 0), (8, 3), (8, 7)), 1), (((3, 5), (4, 5)), 0)])
@pytest.mark.parametrize("vector", [False, True])
def test_concat_generic_kernel_copies_variadic_inputs_and_vector_lanes_exactly(tmp_path, shapes, axis, vector):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    module = concat_module(shapes, axis=axis, vector=vector)
    compiled = Compiler().compile(module).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    lanes = (2, 4) if vector else ()
    inputs = tuple(
        torch.arange(prod((*shape, *lanes)), device="cuda").reshape(*shape, *lanes).remainder(193).to(
            torch.bfloat16 if vector else torch.float32) for shape in shapes)
    runtime.prepare(*inputs)
    output = runtime.run(*inputs)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.cat(inputs, dim=axis % len(shapes[0])), rtol=0, atol=0)


@pytest.mark.parametrize("vector", [False, True])
def test_concat_kernel_preserves_non_concat_axis_owner_coordinates(tmp_path, vector):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    module = concat_module(((8, 3), (8, 7)), split=True, vector=vector)
    placement = module.node_map["output"].type.placement
    module = replace(module, metadata={"auto_distribution": {"placement": placement.to_data()}})
    compiled = Compiler().compile(module).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    shapes = ((8, 3, 2, 4), (8, 7, 2, 4)) if vector else ((8, 3), (8, 7))
    inputs = tuple(
        torch.arange(prod(shape), device="cuda").reshape(shape).remainder(193).to(
            torch.bfloat16 if vector else torch.float32) for shape in shapes)
    runtime.prepare(*inputs)
    output = runtime.run(*inputs)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.cat(inputs, dim=1), rtol=0, atol=0)

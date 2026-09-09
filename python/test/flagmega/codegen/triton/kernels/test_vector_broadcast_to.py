# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace
from math import prod

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule
from python.test.flagmega.ir.ops.primitive_helpers import primitive_module


@pytest.mark.parametrize("shape,lanes,target,output_lanes", [
    ((), (), (4, 3), (2, 4)),
    ((4, 1), (), (4, 5), (8, )),
    ((1, 5), (4, ), (4, 5), (2, 4)),
    ((4, 1), (4, 1), (4, 3), (4, 2)),
    ((4, 1), (2, 1, 4), (4, 3), (2, 8, 4)),
])
@pytest.mark.parametrize("split", [False, True])
def test_vector_element_broadcast_uses_source_lane_coordinates(tmp_path, shape, lanes, target, output_lanes, split):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    # Rank-zero by-value floating parameters use the runtime's FP32 ABI.
    element = "bfloat16" if shape else "float32"
    dtype = fm.vector_type(element, lanes) if lanes else element
    typ = fm.tensor_type(dtype, shape)
    metadata = {}
    if split:
        placement = fm.Placement((2, 2), "yx", "bb")
        policies = tuple(
            fm.SBP.split_block_cyclic((0, ), 2) if axis == 0 and extent == 4 else fm.SBP.broadcast()
            for axis, extent in enumerate(shape))
        typ = fm.DistributedType(typ, policies, placement)
        metadata = {"auto_distribution": {"placement": placement.to_data()}}
    module = replace(
        primitive_module(fm.get_definition("tensors.broadcast_to"), (typ, ), shape=target, output_lanes=output_lanes),
        stage="frozen_constants", metadata=metadata)
    compiled = Compiler().compile(module).module
    runtime = load(write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True),
                   device="cuda:0")
    value = (torch.arange(prod((*shape, *lanes)), device="cuda").reshape(
        (*shape, *lanes)) + .25).to(getattr(torch, element))
    output = torch.full((*target, *output_lanes), float("nan"), device="cuda", dtype=getattr(torch, element))
    arguments = tuple((value if shape else float(value)) if item["role"] == "parameter" else output
                      for item in runtime.external_arguments)
    GeneratedTirCallGraphModule.prepare(runtime, *arguments)
    GeneratedTirCallGraphModule.run_into(runtime, *arguments)
    torch.cuda.synchronize()
    padded_shape = (1, ) * (len(target) - len(shape)) + shape + (1, ) * (len(output_lanes) - len(lanes)) + lanes
    torch.testing.assert_close(output, value.reshape(padded_shape).expand_as(output), rtol=0, atol=0)

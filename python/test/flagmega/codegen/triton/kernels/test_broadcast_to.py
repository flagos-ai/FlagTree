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


@pytest.mark.parametrize("shape,output_shape,vector,split", [
    ((8, 1), (8, 5), False, False),
    ((7, ), (3, 7), True, False),
    ((8, 1), (8, 7), True, True),
    ((8, 1), (8, 5), False, True),
    ((), (3, 7), False, False),
    ((1, ), (11, ), False, False),
    ((1, 1), (1, 8), False, False),
    ((1, 1), (2, 3), True, False),
    ((), (), False, False),
])
def test_broadcast_to_local_kernel_preserves_shape_lanes_and_owners(tmp_path, shape, output_shape, vector, split):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    lanes = (2, 4) if vector else ()
    dtype = fm.vector_type("bfloat16", lanes) if vector else "float32"
    value_type = fm.tensor_type(dtype, shape)
    metadata = {}
    if split:
        placement = fm.Placement((2, 2), "xy", "bb")
        value_type = fm.DistributedType(value_type, (fm.SBP.split_block_cyclic((0, ), 2), fm.SBP.broadcast()),
                                        placement)
        metadata = {"auto_distribution": {"placement": placement.to_data()}}
    module = replace(primitive_module(fm.get_definition("tensors.broadcast_to"), (value_type, ), shape=output_shape),
                     stage="frozen_constants", metadata=metadata)
    compiled = Compiler().compile(module).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    value = (torch.arange(prod((*shape, *lanes)), device="cuda").reshape(
        (*shape, *lanes)).to(torch.bfloat16 if vector else torch.float32) + 1.25 if shape else 1.25)
    output = torch.full((*output_shape, *lanes), float("nan"), device="cuda",
                        dtype=torch.bfloat16 if vector else torch.float32)
    arguments = tuple(value if item["role"] == "parameter" else output for item in runtime.external_arguments)
    GeneratedTirCallGraphModule.prepare(runtime, *arguments)
    GeneratedTirCallGraphModule.run_into(runtime, *arguments)
    torch.cuda.synchronize()
    expected = (value if shape else torch.tensor(value, device="cuda")).expand_as(output)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)

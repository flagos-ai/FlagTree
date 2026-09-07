# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""TensorLoad/TensorStore boundary kernels, without a model or graph optimizer."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load


@pytest.mark.parametrize("extent,lanes,cyclic", [(3, 1, False), (257, 1, True), (6144, 1, False), (259, 8, True)])
def test_tensor_boundary_transfer_preserves_scalar_and_vector_tails(tmp_path, extent, lanes, cyclic):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    dtype = fm.DType.BFLOAT16 if lanes == 1 else fm.vector_type("bfloat16", (lanes,))
    value_type = fm.tensor_type(dtype, (2, extent))
    placement = fm.Placement((8, 16), "yx", "bb")
    policy = fm.SBP.split_block_cyclic((0, 1), 1) if cyclic else fm.SBP.broadcast()
    distributed = fm.DistributedType(value_type, (fm.SBP.broadcast(), policy), placement)

    class Transfer(fm.Module):
        def forward(self):
            source = self.input("source", value_type)
            local = fm.F.distributed.force_boxing(source, distributed, name="load")
            result = fm.F.distributed.force_boxing(local, value_type, name="store")
            self.function("main", (source,), (result,))

    compiler = Compiler()
    source = Transfer(
        dialect="distributed", stage="frozen_constants", entry="main",
        metadata={"auto_distribution": {"placement": placement.to_data()}},
    ).build()
    module = compiler.compile(source).module
    artifact = write_artifact(module, tmp_path / "transfer", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    shape = (2, extent) if lanes == 1 else (2, extent, lanes)
    value = (torch.arange(2 * extent * lanes, device="cuda").reshape(shape) % 251).bfloat16()
    output = torch.full_like(value, float("nan"))
    runtime.prepare(value, output=output)
    runtime.run_into(output, value)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, value, rtol=0, atol=0)
    assert runtime.resource_report["spill_bytes"] == 0

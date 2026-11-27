# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Remote Boxing must address a reusable function's frame in the root window."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


def function_frame_module(reduction: bool, depth: int):
    from triton.flagmega import ir as fm
    from triton.flagmega.compiler import Compiler

    placement = fm.Placement((4, 2, 4), "cyx", "cbb")
    tensor = fm.tensor_type("float32", (3, 4) if reduction else (1, 64))
    split = fm.DistributedType(tensor, (
        fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 1 if reduction else 16),
    ), placement)

    class Graph(fm.Module):
        def forward(self):
            value = self.input("worker_value", tensor, id="worker_value")
            local = fm.F.distributed.force_boxing(value, split, name="load")
            computed = (fm.F.nn.norm_stats(local, axis=1, use_mean=False, name="stats")
                        if reduction else fm.F.math.silu(local, name="compute"))
            result_type = fm.logical_type(computed.type)
            result = fm.F.distributed.force_boxing(computed, result_type, name="gather")
            self.function("worker", (value,), (result,), attrs={"reusable": True, "noinline": True})
            callee = "worker"
            for index in range(depth - 1):
                name = f"outer_{index}"
                parameter = self.input(f"{name}_value", tensor, id=f"{name}_value")
                doubled = fm.F.math.add(parameter, parameter, name=f"{name}_doubled")
                nested = fm.F.builtin.call(doubled, result_type=result_type, callee=callee, name=f"{name}_call")
                self.function(name, (parameter,), (nested,), attrs={"reusable": True, "noinline": True})
                callee = name
            source = self.input("source", tensor, id="source")
            prepared = fm.F.math.add(source, source, name="prepared")
            first = fm.F.builtin.call(prepared, result_type=result_type, callee=callee, name="first")
            second = fm.F.builtin.call(source, result_type=result_type, callee=callee, name="second")
            combined = fm.F.math.add(first, second, name="combined")
            self.function("main", (source,), (combined,))

    return Compiler().compile(Graph(
        dialect="distributed", stage="frozen_constants", entry="main",
        metadata={"auto_distribution": {"placement": placement.to_data()}},
    ).build()).module


def _worker(artifact, reduction, depth):
    import torch
    from triton.flagmega.runtime import load

    device = f"cuda:{os.environ['LOCAL_RANK']}"
    torch.cuda.set_device(device)
    runtime = load(artifact, device=device)
    try:
        shape = (3, 4) if reduction else (1, 64)
        source = ((torch.arange(shape[0] * shape[1], device=device) % 11 - 5) / 4).reshape(shape)

        def expected(value):
            value = value * 2 ** (depth - 1)
            if reduction:
                return (5 * value.square().sum(-1, keepdim=True)).unsqueeze(0)
            return torch.nn.functional.silu(value * 2) + torch.nn.functional.silu(value)

        output = torch.empty_like(expected(source))
        runtime.prepare(source, output=output)
        for step in range(4):
            output.fill_(float("nan"))
            scaled = source * (step + 1)
            runtime.run_into(output, scaled)
            torch.testing.assert_close(output, expected(scaled), rtol=1e-6, atol=1e-6)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("reduction", [False, True])
def test_cross_chip_boxing_in_nonzero_function_frame(tmp_path, reduction, depth):
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < 4 or any(
        torch.cuda.get_device_capability(rank) != (9, 0) for rank in range(4)
    ):
        pytest.skip("Four SM90 CUDA devices are required")
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.ir import verify_buffer_plan

    module = function_frame_module(reduction, depth)
    plan = verify_buffer_plan(module)
    assert any(frame.offset > 0 for call in plan.call_map.values()
               for frame in call.memory_pool_map.values())
    artifact = write_artifact(
        module, tmp_path / "artifact",
        target="nvidia-sm90", emit_executable=True,
    )
    result = subprocess.run([
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
        str(Path(__file__).resolve()), str(artifact), str(int(reduction)), str(depth),
    ], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-8000:]


if __name__ == "__main__":
    _worker(sys.argv[1], bool(int(sys.argv[2])), int(sys.argv[3]))

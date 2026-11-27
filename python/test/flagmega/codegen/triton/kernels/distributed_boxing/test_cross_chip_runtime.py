# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Boxing whose source owners live on other chips of a c-axis placement.

Every chip owns a separate memory instance.  A Boxing leaf that splits or
partially reduces its source over a chip axis, but not its destination, must
read peer instances; a chip-local lowering silently leaves peer elements
unwritten.
"""

import os
from math import prod
from pathlib import Path
import subprocess
import sys

import pytest


_CASES = {
    # split on c -> plain tensor (TensorStore from a chip-split source)
    "split_c": ((0,), 64),
    # split on c and one block axis -> plain tensor
    "split_cy": ((0, 1), 64),
    # All axes distinguish compact owner components in one chip-scoped pool.
    "split_cyx": ((0, 1, 2), 64),
    # partial Sum over c -> plain tensor
    "partial_c": ((0,), None),
    # partial Sum over c and all block axes -> plain tensor
    "partial_cyx": ((0, 1, 2), None),
    "routed_partial_c": ((0,), None),
}


def cross_chip_module(case: str):
    from triton.flagmega import ir as fm
    from triton.flagmega.compiler import Compiler

    axes, elements = _CASES[case]
    placement = fm.Placement((4, 2, 4), "cyx", "cbb")
    if case == "routed_partial_c":
        tensor = fm.tensor_type("float32", (4, 8, 4096))
        split = fm.DistributedType(tensor, (
            fm.SBP.split_contiguous((0,), 1),
            fm.SBP.split_contiguous((1,), 1),
            fm.SBP.split_contiguous((2,), 256),
        ), placement=fm.Placement((4, 8, 16), "cyx", "cbb"))
        placement = split.placement
        result_type = fm.DistributedType(
            fm.tensor_type("float32", (1, 8, 4096)),
            (fm.SBP.broadcast(), fm.SBP.broadcast(), fm.SBP.split_contiguous((1, 2), 32)), placement,
        )

        class Graph(fm.Module):
            def forward(self):
                source = self.input("source", tensor)
                local = fm.F.distributed.force_boxing(source, split, name="load")
                partial = fm.F.math.reduce_sum(local, axes=(0,), keep_dims=True, name="partial")
                routed = fm.F.distributed.force_boxing(partial, result_type, name="route_partials")
                squared = fm.F.math.mul(routed, routed, name="consume_routed")
                result = fm.F.distributed.force_boxing(squared, fm.logical_type(squared.type), name="gather")
                self.function("main", (source,), (result,))
    elif elements is not None:
        tensor = fm.tensor_type("float32", (1, elements))
        owners = 1
        for axis in axes:
            owners *= placement.hierarchy[axis]
        split = fm.DistributedType(tensor, (
            fm.SBP.broadcast(), fm.SBP.split_contiguous(axes, elements // owners),
        ), placement)

        class Graph(fm.Module):
            def forward(self):
                source = self.input("source", tensor)
                local = fm.F.distributed.force_boxing(source, split, name="load")
                value = fm.F.math.silu(local, name="compute")
                result = fm.F.distributed.force_boxing(value, tensor, name="gather")
                self.function("main", (source,), (result,))
    else:
        owners = 1
        for axis in axes:
            owners *= placement.hierarchy[axis]
        tensor = fm.tensor_type("float32", (3, owners))
        split = fm.DistributedType(tensor, (
            fm.SBP.broadcast(), fm.SBP.split_contiguous(axes, 1),
        ), placement)

        class Graph(fm.Module):
            def forward(self):
                source = self.input("source", tensor)
                local = fm.F.distributed.force_boxing(source, split, name="load")
                stats = fm.F.nn.norm_stats(local, axis=1, use_mean=False, name="stats")
                result = fm.F.distributed.force_boxing(
                    stats, fm.logical_type(stats.type), name="reduce",
                )
                self.function("main", (source,), (result,))

    return Compiler().compile(Graph(
        dialect="distributed", stage="frozen_constants", entry="main",
        metadata={"auto_distribution": {"placement": placement.to_data()}},
    ).build()).module


def _expected(torch, case, source):
    if case == "routed_partial_c":
        return source.sum(0, keepdim=True).square()
    if _CASES[case][1] is not None:
        return torch.nn.functional.silu(source)
    # The source is split granularity-1 over the reduction axis, so every rank
    # holds one column and the reduced result needs all of them.
    return source.square().sum(-1, keepdim=True).unsqueeze(0)


def _worker(artifact, case):
    import torch
    from triton.flagmega.runtime import load

    device = f"cuda:{os.environ['LOCAL_RANK']}"
    torch.cuda.set_device(device)
    runtime = load(artifact, device=device)
    try:
        axes, elements = _CASES[case]
        shape = ((4, 8, 4096) if case == "routed_partial_c" else
                 (1, elements) if elements is not None else (3, 32 if len(axes) == 3 else 4))
        source = ((torch.arange(prod(shape), device=device) % 11 - 5) / 4).reshape(shape).float()
        # Every rank receives the whole logical tensor: the first Boxing reads
        # its own slice from it, and the second one must reproduce it in full.
        output = torch.empty_like(_expected(torch, case, source))
        runtime.prepare(source, output=output)
        # Repeated launches exercise the WAR edge between one step's remote
        # readers and the next step's producer of the same workspace bytes.
        for step in range(4):
            output.fill_(float("nan"))
            scaled = source * (step + 1)
            runtime.run_into(output, scaled)
            torch.testing.assert_close(output, _expected(torch, case, scaled), rtol=1e-6, atol=1e-6)
    finally:
        runtime.close()


@pytest.mark.parametrize("case", sorted(_CASES))
def test_cross_chip_boxing_four_ranks(tmp_path, case):
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < 4 or any(
        torch.cuda.get_device_capability(rank) != (9, 0) for rank in range(4)
    ):
        pytest.skip("Four SM90 CUDA devices are required")
    from triton.flagmega.artifacts import write_artifact

    artifact = write_artifact(
        cross_chip_module(case), tmp_path / "artifact", target="nvidia-sm90", emit_executable=True,
    )
    result = subprocess.run([
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
        str(Path(__file__).resolve()), str(artifact), case,
    ], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-8000:]


if __name__ == "__main__":
    _worker(sys.argv[1], sys.argv[2])

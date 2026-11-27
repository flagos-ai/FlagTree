# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Check declared grid costs against real PTX without running on a GPU."""
from dataclasses import replace
import re
import pytest
import triton
import triton.language as tl

tle = pytest.importorskip("triton.experimental.tle.language")


@triton.jit
def _grid_protocol_witness(buf, out, MESH: tl.constexpr):
    pid = tl.program_id(0) * 4 + tl.program_id(1)
    tl.store(buf + pid, pid)
    tle.distributed_barrier(MESH)
    neighbor = ((tl.program_id(0) + 1) % 2) * 4 + tl.program_id(1)
    tl.store(out + pid, tl.load(buf + neighbor))


@pytest.mark.parametrize("axis_group", (False, True))
def test_cooperative_grid_protocol_matches_lowered_cta_barriers(monkeypatch, tmp_path, axis_group):
    from triton import _flagtree_backend
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.flagmega import ir as fm
    from triton.flagmega.targets.nvidia.machine import NvidiaSm90Machine

    if not hasattr(ir.builder, "create_distributed_barrier"):
        pytest.skip("TLE distributed-barrier compiler support is unavailable")
    # Offline compilation has an explicit target; never detect a visible GPU.
    monkeypatch.setattr(_flagtree_backend, "FLAGTREE_BACKEND", "nvidia")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "cache"))
    mesh = tle.device_mesh({"block": [("block_y", 2), ("block_x", 4)]})
    if axis_group:
        mesh = mesh.axis_group(("block_y",))
    compiled = triton.compile(
        ASTSource(fn=_grid_protocol_witness, signature={"buf": "*i32", "out": "*i32"},
                  constexprs={"MESH": mesh}),
        target=GPUTarget("cuda", 90, 32), options={"num_warps": 4},
    )
    if hasattr(compiled, "result"):
        compiled = compiled.result()
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    value = builder.var("value", fm.tensor_type("float32", (1,)), id="value")
    builder.function("main", (value,), (value,))
    module = replace(builder.build(entry="main"), metadata={
        "codegen_package_plan": {"launch": {"cooperative_grid": True}},
    })
    protocol = NvidiaSm90Machine().memory_synchronization_protocol(module)
    actual = len(re.findall(r"bar\.sync\s+0\s*;", compiled.asm["ptx"]))
    assert actual == protocol.grid_cta_barriers == 2

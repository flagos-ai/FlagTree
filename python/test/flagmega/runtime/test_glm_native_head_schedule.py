# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The native head policy covers all compute-warp N fragments per owner."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.selection import apply_plan, override_plan
from triton.flagmega.targets import NvidiaSm90Target


def head_graph(outputs, mesh):
    class Graph(fm.Module):
        def forward(self):
            b = fm.SBP.broadcast()
            x = self.input("x", fm.DistributedType(fm.tensor_type("bfloat16", (1, 4096)), (b, b), mesh), id="x")
            w = self.input(
                "w",
                fm.DistributedType(
                    fm.tensor_type(fm.vector_type("bfloat16", (8, 2, 8)), (256, outputs // 8)),
                    (b, fm.SBP.split_block_cyclic((0, 1), 1)),
                    mesh,
                ),
                id="w",
            )
            no = fm.F.builtin.none()
            result = fm.F.ntt.packed_matmul(
                x,
                w,
                no,
                no,
                output_data_type="bfloat16",
                name="projection",
                metadata={
                    "selected_vectorization": "vectorization.matmul.n",
                    "selected_vector_axes": (1,),
                    "selected_vector_lanes": (8,),
                },
            )
            self.function("main", (x, w), (result,))

    return Graph(
        dialect="distributed",
        stage="frozen_constants",
        entry="main",
        metadata={"auto_distribution": {"placement": mesh.to_data()}},
    ).build()


@pytest.fixture(scope="module")
def head_policy():
    path = (
        Path(__file__).resolve().parents[3]
        / "tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800/local_optimizations/head_schedule.py"
    )
    spec = importlib.util.spec_from_file_location("_flagmega_native_head_policy_test", path)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    return policy


@pytest.mark.parametrize("outputs,expected_tile", [(4096, 16), (8192, 64), (38720, 64)])
def test_head_tile_fills_n_warps_when_local_capacity_allows(head_policy, outputs, expected_tile):
    identity = "tir.dense_matmul.packed_k_major_mma"

    class Target(NvidiaSm90Target):
        def propose_tir(self, module):
            proposal = super().propose_tir(module)
            return apply_plan(
                proposal,
                override_plan(
                    proposal,
                    [("tir.projection", identity)],
                    policy="test.native_head_fragment",
                    rationale="Exercise the native eight-warp packed head.",
                ),
            )

    mesh = fm.Placement((8, 16), "yx", "bb")
    base = NvidiaSm90Target().triton_implementation_model
    catalog = replace(
        base,
        implementations=tuple(
            replace(value, parameters={**value.parameters, "compute_num_warps": 8}) if value.id == identity else value
            for value in base.implementations
        ),
    )
    compiler = Compiler(CompileOptions(placement=mesh))
    compiler.target = Target(triton_implementation_model=catalog)
    module = compiler.compile(head_graph(outputs, mesh)).module
    (before,) = (d for d in module.kernel_definitions if d.dispatch.microkernel.family == "dense_matmul")
    rewritten = head_policy.rewrite(module)
    after = next(d for d in rewritten if d.name == before.name)
    assert after.dispatch.microkernel.parameters["tile_n"] == expected_tile
    assert after.dispatch.microkernel.parameters["block_k"] == (128 if expected_tile == 64 else 32)
    if expected_tile == 16:
        assert after == before
    fm.verify_module(replace(module, kernel_definitions=rewritten))

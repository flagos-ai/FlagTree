# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The normal GLM target selects the validated packed router schedule."""

from dataclasses import replace
import importlib.util
from pathlib import Path

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions


IMPLEMENTATION = "tir.dense_matmul.packed_k_major_threaded_fma_register8_n8_s2"
ORIGINAL = "tir.dense_matmul.packed_k_major_threaded_fma"


@pytest.fixture
def native(monkeypatch):
    tutorial = Path(__file__).parents[3] / "tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800"
    monkeypatch.syspath_prepend(str(tutorial))
    spec = importlib.util.spec_from_file_location("_glm_router_target", tutorial / "local_optimizations/native_tp4.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compile(native, hierarchy=(8, 16), *, m=1, n=288, k=4096, output="float32"):
    mesh = fm.Placement(hierarchy, "yx", "bb")
    broadcast = fm.SBP.broadcast()

    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.DistributedType(
                fm.tensor_type("bfloat16", (m, k)), (broadcast, broadcast), mesh), id="x")
            weight = self.input("weight", fm.DistributedType(
                fm.tensor_type(fm.vector_type("bfloat16", (8, 2, 8)), (k // 16, n // 8)),
                (broadcast, fm.SBP.split_block_cyclic((1,), 1)), mesh), id="weight")
            none = fm.F.builtin.none()
            value = fm.F.ntt.packed_matmul(
                x, weight, none, none, output_data_type=output, name="projection",
                metadata={"selected_vectorization": "vectorization.matmul.n",
                          "selected_vector_axes": (1,), "selected_vector_lanes": (8,)})
            self.function("main", (x, weight), (value,))

    source = Graph(dialect="distributed", stage="frozen_constants", entry="main",
                   metadata={"auto_distribution": {"placement": mesh.to_data()}}).build()
    catalog = native.native_implementation_model()
    compiler = Compiler(CompileOptions(placement=mesh))
    compiler.target = native.NativeTp4DistributionTarget(
        triton_implementation_model=catalog).with_placement(mesh)
    return compiler.compile(source).module


@pytest.mark.parametrize("n,k", [(288, 4096), (264, 1024), (280, 2048)])
def test_fresh_target_selects_register_load_before_bufferization(native, n, k):
    module = _compile(native, n=n, k=k)
    record = module.selection_map["tir.projection"]
    assert record.candidate_id == IMPLEMENTATION
    definitions = [d for d in module.kernel_definitions if d.dispatch.microkernel.family == "dense_matmul"]
    assert len(definitions) == 1
    micro = definitions[0].dispatch.microkernel
    assert micro.implementation == record.candidate_id
    assert micro.parameters["weight_load_layout"] == "register8"
    assert micro.parameters["tile_n"] == 8
    assert micro.parameters["scalar_store_subtile_n"] == 2
    assert fm.verify_buffer_plan(module)


@pytest.mark.parametrize("hierarchy,m,n,output", [
    ((2, 64), 1, 2048, "float32"),
    ((1, 128), 1, 2048, "float32"),
    ((8, 16), 1, 128, "float32"),
    ((8, 16), 2, 288, "float32"),
    ((8, 16), 1, 288, "bfloat16"),
])
def test_selection_respects_replica_and_measured_domain(native, hierarchy, m, n, output):
    module = _compile(native, hierarchy, m=m, n=n, output=output)
    assert module.selection_map["tir.projection"].candidate_id != IMPLEMENTATION
    assert fm.verify_buffer_plan(module)


def test_catalog_preserves_the_generic_scalar_schedule(native):
    catalog = native.native_implementation_model()
    assert catalog.implementation(ORIGINAL).parameters.get("weight_load_layout", "scalar") == "scalar"
    implementation = catalog.implementation(IMPLEMENTATION)
    assert implementation is not None
    assert implementation.variant == catalog.implementation(ORIGINAL).variant


def test_profile_uses_the_validated_completion_lifetime_budget(native):
    target = native.NativeTp4DistributionTarget(
        triton_implementation_model=native.native_implementation_model())
    assert target.bufferization_policy.options.barrier_bytes_budget == 65536
    assert target.bufferization_policy.options.barrier_fixpoint_rounds == 1
    assert target.bufferization_policy.options.rdata_chip_sharding


def test_explicit_bufferization_policy_is_preserved(native):
    from triton.flagmega.targets import NvidiaSm90Target
    policy = NvidiaSm90Target().bufferization_policy
    policy = policy.with_options(replace(policy.options, barrier_bytes_budget=128))
    target = native.NativeTp4DistributionTarget(bufferization_policy=policy)
    assert target.bufferization_policy is policy

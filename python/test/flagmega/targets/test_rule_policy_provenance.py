# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Rule choices and machine choices have distinct editable provenance."""

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.implementation import TritonImplementationModel
from triton.flagmega.compiler import Compiler
from triton.flagmega.passes.tir.bufferize import BufferizationOptions
from triton.flagmega.passes.auto_distributed import (
    DistributedOperationCostModel,
    DistributedReshardCostModel,
)
from triton.flagmega.targets import NvidiaSm90Target, NttTargetOptions, PyNttTarget
from triton.flagmega.targets.selection import CapabilitySelectionPolicy


class _AllCapabilities:
    def supports(self, requirements) -> bool:
        return True


class _PortableTestMachine:
    """Non-NVIDIA machine proving that PyNTT graph rules are replaceable."""

    name = "portable-test-machine"
    policy_version = "portable-test-machine/v1"
    codegen_platform = "portable"
    codegen_architecture = "scalar"
    capability = _AllCapabilities()

    def default_ntt_options(self) -> NttTargetOptions:
        return NttTargetOptions(
            placements=(fm.Placement((2, 4), "xy", "bb"),),
            vector_lane_bytes=8,
            vector_max_axes=2,
            packing_vector_bytes=8,
            packing_k_pack=4,
        )

    def triton_implementation_model(self):
        return TritonImplementationModel(name="portable-test/v1")

    def bufferization_options(self):
        return BufferizationOptions.generic(alignment=64)

    def distributed_reshard_cost_model(self):
        return DistributedReshardCostModel(grid_synchronization_cost=1)

    def distributed_operation_cost_model(self):
        return DistributedOperationCostModel()

    def selection_policy(self, options):
        return CapabilitySelectionPolicy()

    def annotate_workspaces(self, node, candidates, module, *, mesh_hierarchy):
        return tuple(candidates)

    def plan_launch(self, module, kernel_nodes):
        return {"grid": (1,)}

    def plan_codegen_package(self, module, kernel_nodes, options):
        return {"profile": self.name}

    def verify(self, module, **kwargs) -> None:
        return None


def _binary_module(stage: str) -> fm.IRModule:
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage=stage, entry="main")

        def forward(self):
            value_type = fm.tensor_type("bfloat16", (1, 16))
            lhs = self.input("lhs", value_type, id="lhs")
            rhs = self.input("rhs", value_type, id="rhs")
            output = fm.F.math.add(lhs, rhs, name="output")
            self.function("main", (lhs, rhs), (output,))

    return Graph().build()


def _dense_module(stage: str) -> fm.IRModule:
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage=stage, entry="main")

        def forward(self):
            value = self.input(
                "value", fm.tensor_type("bfloat16", (1, 128)), id="value"
            )
            weight = self.weight(
                "weight",
                fm.tensor_type("bfloat16", (128, 128)),
                source="memory",
                key="weight",
                id="weight",
            )
            output = fm.F.math.matmul(
                value, weight, transpose_b=True, name="output"
            )
            self.function("main", (value,), (output,))

    return Graph().build()


def test_vectorization_selection_records_ntt_rule_set_not_machine_generation():
    target = NvidiaSm90Target()
    proposed = target.propose_vectorization(_binary_module("decomposed"))
    record = proposed.selection_map["vectorization.output"]

    assert record.policy == target.vectorization_policy.identity
    assert record.policy.startswith("ntt-auto-vectorize/")
    assert "sm90" not in record.policy.lower()
    assert "nvidia" not in record.policy.lower()


def test_distribution_materialization_records_pyntt_split_policy_provenance():
    target = NvidiaSm90Target()
    distributed = target.auto_distribute(_binary_module("stats_threaded"))
    records = tuple(
        record for record in distributed.selections
        if record.point_id.startswith("distribution.")
    )

    assert records
    assert {record.policy for record in records} == {
        target.distribution_policy.identity
    }
    assert all("pyntt" in record.policy for record in records)
    assert all("nvidia" not in record.policy.lower() for record in records)
    assert all("sm90" not in record.policy.lower() for record in records)


def test_packing_selection_records_pyntt_rule_set_not_machine_generation():
    target = NvidiaSm90Target()
    proposed = target.propose_packing(_dense_module("vectorized"))
    record = proposed.selection_map["packing.output"]

    assert record.policy == target.packing_policy.identity
    assert record.policy.startswith("pyntt-auto-packing/")
    assert "nvidia" not in record.policy.lower()
    assert "sm90" not in record.policy.lower()


def test_backend_and_machine_identities_are_separate_objects():
    target = NvidiaSm90Target()

    assert target.backend_name == "pyntt"
    assert target.machine_name == "nvidia-sm90"
    assert target.machine_policy_version.startswith("nvidia-sm90-machine/")
    assert target.vectorization_policy.identity != target.machine_policy_version
    assert target.distribution_policy.identity != target.machine_policy_version


def test_selected_tir_serializes_backend_and_machine_as_separate_fields():
    compiled = Compiler().compile(_binary_module("imported"), stop_after="lower-tir").module

    assert compiled.metadata["target_backend"] == "pyntt"
    assert compiled.metadata["target_machine"] == "nvidia-sm90"
    assert compiled.metadata["target_machine_policy"].startswith(
        "nvidia-sm90-machine/"
    )


def test_pyntt_rules_run_with_a_non_nvidia_machine_and_its_options():
    target = PyNttTarget(_PortableTestMachine())

    vectorized = target.propose_vectorization(_binary_module("decomposed"))
    distributed = target.auto_distribute(_binary_module("stats_threaded"))

    assert target.name == "pyntt:portable-test-machine"
    assert target.options.vector_lane_bytes == 8
    assert target.options.placements == (fm.Placement((2, 4), "xy", "bb"),)
    assert "nvidia" not in target.distribution_policy.identity.lower()
    assert any(point.id == "vectorization.output" for point in vectorized.selection_points)
    assert any(
        record.point_id.startswith("distribution.")
        for record in distributed.selections
    )
    assert all(
        "nvidia" not in record.policy.lower() and "sm90" not in record.policy.lower()
        for record in vectorized.selections + distributed.selections
        if record.point_id.startswith(("vectorization.", "distribution."))
    )


def test_portable_pyntt_packing_default_is_not_reinterpreted_as_sm90_policy():
    target = PyNttTarget(_PortableTestMachine())
    proposed = target.propose_packing(_dense_module("vectorized"))
    point = next(
        point for point in proposed.selection_points if point.id == "packing.output"
    )
    record = proposed.selection_map[point.id]

    assert point.default_candidate == "packing.k_major_n4_k16"
    assert record.candidate_id == point.default_candidate
    assert record.policy.startswith("pyntt-auto-packing/")
    assert "nvidia" not in record.policy.lower()
    assert "sm90" not in record.policy.lower()

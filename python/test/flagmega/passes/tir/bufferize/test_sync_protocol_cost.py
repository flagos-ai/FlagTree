# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Memory-plan comparisons honor the target's declared barrier expansion."""
from dataclasses import replace
import pytest
from triton.flagmega.ir.bufferization import MemorySynchronizationPlan, SynchronizationEvent
from triton.flagmega.passes.tir.bufferize import BufferPlanner, synchronization
from triton.flagmega.passes.tir.bufferize.reuse_preferences import dominates_memory_schedule
from .test_barrier_aware_placement import _case


def declared_protocol(grid_cta_barriers=2):
    return {"schema": "flagmega.memory-synchronization-protocol/v1", "identity": "nvidia-sm90/tle-grid/v1",
            "grid_cta_barriers": grid_cta_barriers, "device_grid_barriers": 2}


@pytest.mark.parametrize("after_blocks,grid_cta_barriers,accepted", ((5,2,True),(6,2,False),(5,1,False)))
def test_grid_expansion_pays_only_for_its_actual_cta_phases(monkeypatch, after_blocks, grid_cta_barriers, accepted):
    module,options=_case()
    module=replace(module,metadata={"memory_synchronization_protocol":declared_protocol(grid_cta_barriers)})
    baseline=BufferPlanner(module,options).run()
    candidate=replace(baseline)

    def events(scopes):
        return MemorySynchronizationPlan(tuple(SynchronizationEvent(
            "main","produced","output",scope,(),())for scope in scopes))

    before=("grid",)*4+("block",)*3
    after=("grid",)*3+("block",)*after_blocks
    monkeypatch.setattr(synchronization,"plan_memory_synchronization",
                        lambda mod,plan:events(after if plan is candidate else before))
    assert dominates_memory_schedule(module,candidate,baseline) is accepted


def test_unknown_target_retains_the_conservative_opaque_event_comparison(monkeypatch):
    module,options=_case()
    baseline=BufferPlanner(module,options).run()
    candidate=replace(baseline)

    def events(scopes):
        return MemorySynchronizationPlan(tuple(SynchronizationEvent(
            "main","produced","output",scope,(),())for scope in scopes))

    monkeypatch.setattr(synchronization,"plan_memory_synchronization",lambda mod,plan:events(
        ("grid",)*3+("block",)*5 if plan is candidate else ("grid",)*4+("block",)*3))
    assert not dominates_memory_schedule(module,candidate,baseline)


@pytest.mark.parametrize("cooperative,cta_barriers", ((False,1),(True,2)))
def test_nvidia_declares_the_protocol_of_its_selected_entry(cooperative,cta_barriers):
    from triton.flagmega.targets.nvidia.machine import NvidiaSm90Machine
    from triton.flagmega.ir.bufferization.synchronization_protocol import MemorySynchronizationProtocol
    from triton.flagmega import ir as fm
    from .test_call_exit_synchronization import completed_read_graph
    module=completed_read_graph()
    module=replace(module,metadata={"codegen_package_plan":{"launch":{"cooperative_grid":cooperative}}})
    protocol=NvidiaSm90Machine().memory_synchronization_protocol(module)
    assert protocol.grid_cta_barriers==cta_barriers
    assert MemorySynchronizationProtocol.from_data(protocol.to_data())==protocol
    module=replace(module,metadata={"memory_synchronization_protocol":protocol.to_data()})
    assert fm.verify_module(fm.IRModule.from_data(module.to_data())).semantic_hash==module.semantic_hash


def test_a_derived_foreign_backend_does_not_inherit_nvidia_counts():
    from triton.flagmega.targets.iluvatar_bi_v150 import IluvatarBiV150Machine
    module,_=_case()
    module=replace(module,metadata={"codegen_package_plan":{"launch":{"cooperative_grid":True}}})
    assert IluvatarBiV150Machine().memory_synchronization_protocol(module) is None


def test_device_protocol_matches_the_emitted_entry_fences():
    from triton.flagmega.targets.nvidia.machine import NvidiaSm90Machine
    from .test_barrier_aware_placement import _emitted_barriers
    module,_=_case()
    module=replace(module,metadata={"codegen_package_plan":{"launch":{"cooperative_grid":True}}})
    protocol=NvidiaSm90Machine().memory_synchronization_protocol(module)
    emitted=_emitted_barriers(("device",))
    assert emitted["grid"]==protocol.device_grid_barriers
    assert emitted["device"]==1


@pytest.mark.parametrize("field,value", (("grid_cta_barriers",True),("grid_cta_barriers",-1),
                                         ("device_grid_barriers",1.5),("schema","wrong")))
def test_ir_verification_rejects_malformed_protocol(field,value):
    from triton.flagmega import ir as fm
    from triton.flagmega.errors import IRVerificationError
    module,_=_case()
    data=declared_protocol();data[field]=value
    module=replace(module,metadata={"memory_synchronization_protocol":data})
    with pytest.raises(IRVerificationError,match="[Pp]rotocol"):
        fm.verify_module(module)


def test_policy_preserves_provider_through_option_copies_and_replaces_stale_target_data():
    from triton.flagmega.passes.tir.bufferize.policy import NttBufferizationPolicy
    from triton.flagmega.ir.bufferization.synchronization_protocol import MemorySynchronizationProtocol
    module,options=_case()
    protocol=MemorySynchronizationProtocol.from_data(declared_protocol())
    policy=NttBufferizationPolicy(options,synchronization_protocol_provider=lambda _:protocol)
    copied=policy.with_optimization_level("fast").with_options(options)
    assert copied._with_protocol(module).metadata["memory_synchronization_protocol"]==protocol.to_data()
    foreign=NttBufferizationPolicy(options,synchronization_protocol_provider=lambda _:None)
    assert "memory_synchronization_protocol" not in foreign._with_protocol(copied._with_protocol(module)).metadata

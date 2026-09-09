# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

from triton.flagmega.compiler import Compiler
from triton.flagmega.passes import FunctionalPass, PassManager
from triton.flagmega.passes.auto_distributed import AutoDistributedPass
from triton.flagmega.passes.auto_distributed.proposal_analysis import PROPOSAL_ANALYSIS
from triton.flagmega.targets import NvidiaSm90Target

from .helpers import packed_matmul_module


def _count_builds(monkeypatch):
    builds = []
    original = AutoDistributedPass._build_graph

    def build(module, target):
        builds.append(module)
        return original(module, target)

    monkeypatch.setattr(AutoDistributedPass, "_build_graph", staticmethod(build))
    return builds


def test_standalone_run_and_unbroken_compiler_pipeline_build_once(monkeypatch):
    builds = _count_builds(monkeypatch)
    AutoDistributedPass.run(packed_matmul_module(), NvidiaSm90Target())
    assert len(builds) == 1
    module = replace(packed_matmul_module(), stage="unused_functions_removed")
    Compiler().compile(module, stop_after="auto-distributed")
    assert len(builds) == 2


def test_pause_resume_starts_fresh_analysis(monkeypatch):
    builds = _count_builds(monkeypatch)
    module = replace(packed_matmul_module(), stage="unused_functions_removed")
    proposal = Compiler().compile(module, stop_after="propose-distribution").module
    Compiler().compile(proposal, stop_after="auto-distributed")
    assert len(builds) == 2


def test_agent_edit_is_checked_even_if_custom_pass_claims_to_preserve(monkeypatch):
    builds = _count_builds(monkeypatch)
    target = NvidiaSm90Target()
    preserve = frozenset({PROPOSAL_ANALYSIS})
    manager = PassManager("unit")
    manager.add(FunctionalPass("propose", lambda m: AutoDistributedPass.propose(m, target), preserves=preserve))
    manager.add(
        FunctionalPass("edit", lambda m: replace(m, metadata={**m.metadata, "agent_trial": 2}), preserves=preserve))
    manager.add(FunctionalPass("apply", lambda m: AutoDistributedPass.apply(m, target)))
    manager.run(packed_matmul_module())
    assert len(builds) == 2

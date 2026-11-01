# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A peer-reading program closes its device epoch before successor reuse."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton import render_triton_package
from triton.flagmega.codegen.triton.tir_package import describe_tir_package
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import ArtifactError
from triton.flagmega.options import CompileOptions
from triton.flagmega.runtime.kernel_programs import verified_kernel_programs
from triton.flagmega.targets import NvidiaSm90Target
from python.test.flagmega.codegen.triton.test_kernel_programs import import_source


def compile_peer_programs():
    mesh = fm.Placement((4, 2, 2), "cyx", "cbb")
    kind = fm.DistributedType(fm.tensor_type("float32", (4, 16)),
                             (fm.SBP.split_contiguous((0,), 1), fm.SBP.broadcast()), mesh)
    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", kind, id="x")
            old = fm.F.math.reduce_sum(x, axes=(0,), keep_dims=True, name="old")
            published = fm.F.distributed.boxing(old, fm.logical_type(old.type), name="published")
            after = fm.F.math.add(published, published, name="after")
            self.function("main", (x,), (published, after))

    class Target(NvidiaSm90Target):
        def propose_tir(self, module):
            module = super().propose_tir(module)
            return replace(module, metadata={**module.metadata, "kernel_program_partition": tuple(
                dict(symbol="program_"+name, calls=(name,), num_warps=4,
                     depends_on=("program_"+parent,) if parent else ())
                for name, parent in (("old", None), ("published", "old"), ("after", "published")))})
    compiler = Compiler(CompileOptions(placement=mesh))
    compiler.target = Target()
    return compiler.compile(Graph(dialect="distributed", stage="frozen_constants", entry="main",
        metadata={"auto_distribution": {"placement": mesh.to_data()}}).build()).module


def test_peer_program_has_a_typed_terminal_device_completion_and_residency_contract(tmp_path):
    module = compile_peer_programs()
    descriptor = describe_tir_package(module)
    specs = descriptor["kernels"]
    assert [s["device_completion"] for s in specs] == [False, True, False]
    assert specs[1]["requires_grid_wide_sync"]
    events = descriptor["programs"][1]["entry_events"]
    assert events[-1] == {"kind": "barrier", "scope": "device", "reason": "program_device_completion"}
    assert all(not p["entry_events"][-1].get("reason") for p in (descriptor["programs"][0], descriptor["programs"][2]))
    package = render_triton_package(module, tmp_path)
    source = import_source(tmp_path / "generated_kernels.py")
    kernels = tuple(getattr(source, s["symbol"]) for s in specs)
    assert len(verified_kernel_programs(package, kernels, module)) == 3
    forged = deepcopy(package)
    forged["kernels"][1]["device_completion"] = False
    source.FLAGMEGA_KERNEL_PROGRAMS = forged["kernels"]
    with pytest.raises(ArtifactError, match="device completion"):
        verified_kernel_programs(forged, kernels, module)


def test_peer_completion_cannot_claim_noncooperative_residency(tmp_path):
    module = compile_peer_programs()
    package = render_triton_package(module, tmp_path)
    source = import_source(tmp_path / "generated_kernels.py")
    kernels = tuple(getattr(source, s["symbol"]) for s in package["kernels"])
    package["kernels"][1]["requires_grid_wide_sync"] = False
    source.FLAGMEGA_KERNEL_PROGRAMS = package["kernels"]
    with pytest.raises(ArtifactError, match="device completion"):
        verified_kernel_programs(package, kernels, module)


@pytest.mark.parametrize("case,removed", [("completed", True), ("new_writer", False),
                                         ("unknown_range", False), ("explicit", False)])
def test_completion_elision_requires_global_coverage_of_every_range(case, removed):
    from triton.flagmega.codegen.triton.program_dependencies import elide_completed_device_barriers

    programs = (dict(symbol="producer", depends_on=()), dict(symbol="peer", depends_on=("producer",)),
                dict(symbol="new", depends_on=("peer",)), dict(symbol="reader", depends_on=("new",)))
    access = SimpleNamespace(physical_id="workspace", offset=0, nbytes=64)
    accesses = dict(producer=(access,), peer=(access,), new=(access,) if case == "new_writer" else (), reader=())
    completion = dict(producer=False, peer=True, new=False, reader=False)
    event = dict(kind="barrier", scope="device", before="reader", hazards=("READ->WRITE",),
                 ranges=[dict(physical_id="workspace", offset=0, nbytes=64)])
    if case == "unknown_range":
        event["ranges"].append(dict(physical_id="external", offset=0, nbytes=64))
    elif case == "explicit":
        event["ranges"] = []
    result, elided = elide_completed_device_barriers([event, dict(kind="function_call")], "reader",
                                                    programs, accesses, completion)
    assert bool(elided) is removed
    assert (result[0]["kind"] == "function_call") is removed

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Canonical coordinates do not imply that physical allocations are shared."""
import pytest

from triton.flagmega.codegen.triton.kernel_call_renderers import _canonical_writer_active
from python.test.flagmega.codegen.triton.distribution.test_rank_generic_mesh import _distributed_abi


@pytest.mark.parametrize("levels", ["bbb", "cbb", "bcb"])
@pytest.mark.parametrize("split_axes", [(), (1,)])
def test_every_private_canonical_component_is_populated(levels, split_axes):
    abi = _distributed_abi(hierarchy=(4, 8, 16), split_axes=split_axes)
    abi["distributed_type"]["placement"]["hierarchy_levels"] = levels
    abi["memory_sharing_scope"] = "block"
    abi.update(pooled=True, pool_scope_count=128, pool_scope_stride_bytes=4096, pool_scope_index="block_index")
    assert _canonical_writer_active(abi) == "True"


def test_boxing_writer_uses_destination_allocation_and_iteration_domain():
    destination = _distributed_abi(hierarchy=(4, 8, 16))
    destination["distributed_type"]["placement"]["hierarchy_levels"] = "cbb"
    iteration = _distributed_abi(hierarchy=(4, 8, 16), split_axes=(1,))
    iteration["distributed_type"]["placement"]["hierarchy_levels"] = "cbb"
    iteration["memory_sharing_scope"] = "block"
    # A shared destination still elects one writer per disjoint source shard,
    # even if that source is private. An input's memory scope is not the
    # destination's allocation scope (also true for cache/indexer publication).
    destination["memory_sharing_scope"] = "chip"
    assert _canonical_writer_active(destination, iteration_abi=iteration) == "(shard_coord2 == 0)"
    destination["memory_sharing_scope"] = "block"
    assert _canonical_writer_active(destination, iteration_abi=iteration) == "True"


def test_canonical_private_copies_are_read_by_each_physical_owner(tmp_path):
    """A compact replica may consume a private canonical intermediate later."""
    from dataclasses import replace
    import torch
    from triton.flagmega import ir as fm
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.options import CompileOptions
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Run
    from triton.flagmega.targets import NvidiaSm90Target

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    mesh = fm.Placement((2, 2), "yx", "bb")
    kind = fm.tensor_type("float32", (128,))
    broadcast = fm.DistributedType(kind, (fm.SBP.broadcast(),), mesh)
    split = fm.DistributedType(kind, (fm.SBP.split_contiguous((0,), 64),), mesh)

    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", kind, id="x")
            copied = fm.F.distributed.force_boxing(x, broadcast)
            first = fm.F.math.add(copied, copied)
            owned = fm.F.distributed.force_boxing(first, split)
            second = fm.F.math.add(owned, owned)
            result = fm.F.distributed.force_boxing(second, kind)
            self.function("main", (x,), (result,))

    source = Graph(dialect="distributed", stage="frozen_constants", entry="main",
                   metadata={"auto_distribution": {"placement": mesh.to_data()}}).build()
    compiler = Compiler(CompileOptions(placement=mesh))
    base = compiler.target
    # Canonical arguments are a normal kernel capability. This reproduces the
    # private canonical allocation without relying on MLA or a local override.
    catalog = replace(base.triton_implementation_model, implementations=tuple(
        replace(i, facts={**i.facts, "canonical_global_arguments": ("lhs", "rhs")})
        if i.family == "elementwise" and i.variant == "add" else i
        for i in base.triton_implementation_model.implementations))
    compiler.target = NvidiaSm90Target(options=base.options, triton_implementation_model=catalog)
    module = compiler.compile(source).module
    plan = fm.verify_buffer_plan(module)
    assert any(b.distributed_storage_kind.value == "canonical_global"
               and plan.memory_space_map[b.mem_span.buffer.memory_space].sharing_scope.value == "block"
               for b in plan.buffers)
    runtime = load(write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True),
                   device="cuda:0")
    x = torch.empty(128, dtype=torch.float32, device="cuda")
    storage = torch.full((160,), 123., device="cuda")
    output = storage[16:-16]
    args = tuple(output if s["role"] == "result" else x for s in runtime.external_arguments)
    try:
        Run.prepare(runtime, *args)
        for mutation in range(5):
            x.copy_((torch.arange(128, device="cuda").remainder(17).float() - 8 + mutation) / 16)
            output.fill_(-17)
            Run.run_into(runtime, *args)
            torch.testing.assert_close(output, x * 4, rtol=0, atol=0)
            assert bool((storage[:16] == 123).all()) and bool((storage[-16:] == 123).all())
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()

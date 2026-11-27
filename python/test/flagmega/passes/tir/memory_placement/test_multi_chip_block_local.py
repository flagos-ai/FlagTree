# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Private block frames belong to one physical chip, independently of SBP."""

from dataclasses import replace
from itertools import product
from math import prod

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.runtime_binding import describe_function_runtime_binding
from triton.flagmega.passes.tir import (
    MEMORY_SPACE_METADATA,
    bind_prim_function_buffers,
    materialize_kernel_prim_functions,
    plan_function_memory,
    specialize_prim_functions_for_buffer_layouts,
)

from python.test.flagmega.passes.tir.memory_placement.test_plan_function_memory import _kernel_attrs, _options

MESHES = (
    fm.Placement((4, 2, 4), "cyx", "cbb"),
    fm.Placement((2, 4, 4), "ycx", "bcb"),
    fm.Placement((4, 2, 4), "dyx", "dbb"),
)


def _chain(mesh, *, narrow=False, full_replica=False):
    chip = next(i for i, level in enumerate(mesh.hierarchy_levels) if level != "b")
    blocks = tuple(i for i, level in enumerate(mesh.hierarchy_levels) if level == "b")
    tensor = fm.tensor_type("float32", (16, 32))
    rows = fm.SBP.broadcast() if full_replica else fm.SBP.split_contiguous((chip,), 4)
    source_type = fm.DistributedType(tensor, (rows, fm.SBP.broadcast()), mesh)
    target_type = fm.DistributedType(tensor, (
        rows, fm.SBP.split_contiguous(blocks, 4) if narrow else fm.SBP.broadcast(),
    ), mesh)
    builder = fm.IRBuilder(dialect="semantic_tir", stage="packaged_tir")
    value = builder.var("value", source_type, id="value")
    produced = builder.call("tir.kernel", (value,), source_type, id="producer", attrs=_kernel_attrs())
    viewed = (builder.call("distributed.sharded_view", (produced,), target_type, id="view",
                           attrs={"new_type": target_type}) if narrow else produced)
    output = builder.call("tir.kernel", (viewed,), target_type, id="consumer", attrs=_kernel_attrs())
    builder.function("main", (value,), (output,))
    module = materialize_kernel_prim_functions(builder.build(entry="main"))
    return fm.verify_module(replace(module, metadata={**module.metadata, "launch_contract": {
        "kind": "single_prepared_entry", "entry": "flagmega_main", "cooperative_grid": True,
        "grid_mesh": mesh.to_data(),
    }}))


@pytest.mark.parametrize("mesh", MESHES, ids=("cbb", "bcb", "dbb"))
@pytest.mark.parametrize("narrow,full_replica", ((False, False), (True, False), (True, True)))
def test_owner_local_edges_do_not_require_an_all_block_mesh(tmp_path, mesh, narrow, full_replica):
    module = _chain(mesh, narrow=narrow, full_replica=full_replica)
    placed = plan_function_memory(module, _options())
    assert placed.node_map["producer"].metadata.get(MEMORY_SPACE_METADATA) == "block_local_data"
    plan = fm.make_buffer_plan(placed, options=_options())
    producer = plan.buffer_map["producer"]
    assert producer.storage == "block_local_data"
    assert producer.distributed_storage_kind is fm.DistributedBufferStorageKind.COMPACT_LOCAL
    if narrow:
        view = plan.buffer_map["view"]
        assert view.storage == producer.storage
        assert view.mem_span.must_alias(producer.mem_span)
        assert view.nbytes == producer.nbytes
        if full_replica:
            assert view.distributed_storage_kind is fm.DistributedBufferStorageKind.REPLICATED_LOCAL
    specialized = specialize_prim_functions_for_buffer_layouts(placed, plan)
    plan = fm.make_buffer_plan(specialized, options=_options())
    bound = bind_prim_function_buffers(specialized, plan=plan)
    bound = fm.verify_module(replace(bound, stage="bufferized_tir", dialect="bufferized_tir",
        metadata={**bound.metadata, "buffer_plan": plan.to_data()}))
    assert fm.load_module(fm.emit_module(bound, tmp_path / "selected.py")).semantic_hash == bound.semantic_hash


@pytest.mark.parametrize("mesh", MESHES, ids=("cbb", "bcb", "dbb"))
def test_block_pool_index_is_bounded_with_every_chip_coordinate(mesh):
    module = _chain(mesh)
    module = replace(module, nodes=tuple(
        replace(n, metadata={**n.metadata, MEMORY_SPACE_METADATA: "block_local_data"})
        if n.id == "producer" else n for n in module.nodes
    ))
    plan = fm.make_buffer_plan(module, options=_options())
    module = specialize_prim_functions_for_buffer_layouts(module, plan)
    plan = fm.make_buffer_plan(module, options=_options())
    bound = bind_prim_function_buffers(module, plan=plan)
    bound = fm.verify_module(replace(bound, stage="bufferized_tir", dialect="bufferized_tir",
        metadata={**bound.metadata, "buffer_plan": plan.to_data()}))
    binding = describe_function_runtime_binding(bound)
    pool, = (p for p in binding["pools"] if p["storage"] == "block_local_data")
    abi = next(c for c in binding["call_abi"]["kernel_calls"] if c["call"] == "producer")["outputs"][0]["buffers"][0]["abi"]
    count = prod(size for size, level in zip(mesh.hierarchy, mesh.hierarchy_levels) if level == "b")
    assert pool["scope_count"] == count
    for coordinates in product(*(range(size) for size in mesh.hierarchy)):
        shard_index = 0
        block_index = 0
        for coordinate, size, level in zip(coordinates, mesh.hierarchy, mesh.hierarchy_levels):
            shard_index = shard_index * size + coordinate
            if level == "b":
                block_index = block_index * size + coordinate
        index = {"block_index": block_index, "shard_index": shard_index}[abi["pool_scope_index"]]
        assert 0 <= index < count, (coordinates, index, count)


def private_view_stage(world, layout, full_replica):
    """A real private producer with a proper contained narrowing consumer."""
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.options import CompileOptions

    mesh = (fm.Placement((world, 8, 16), "cyx", "cbb") if layout == "cbb"
            else fm.Placement((8, world, 16), "ycx", "bcb"))
    chip, = (i for i, level in enumerate(mesh.hierarchy_levels) if level == "c")
    blocks = tuple(i for i, level in enumerate(mesh.hierarchy_levels) if level == "b")

    class Graph(fm.Module):
        def forward(self):
            source = self.input("value", fm.tensor_type("float32", (512, 2)), id="value")
            parent = fm.DistributedType(source.type, (
                fm.SBP.broadcast() if full_replica else fm.SBP.split_contiguous((chip,)),
                fm.SBP.broadcast(),
            ), mesh)
            viewed = fm.F.distributed.sharded_view(source, parent)
            stats = fm.F.nn.norm_stats(viewed, axis=1, use_mean=False, name="private_stats")
            narrowed = fm.DistributedType(stats.type.tensor, (
                fm.SBP.broadcast(), fm.SBP.split_contiguous((chip, *blocks)), fm.SBP.broadcast(),
            ), mesh)
            view = fm.F.distributed.sharded_view(stats, narrowed, name="private_narrow")
            doubled = fm.F.math.add(view, view)
            result = fm.F.distributed.force_boxing(doubled, stats.type.tensor)
            self.function("main", (source,), (result,))

    source = Graph(dialect="distributed", stage="frozen_constants", entry="main",
                   metadata={"auto_distribution": {"placement": mesh.to_data()}}).build()
    module = Compiler(CompileOptions(placement=mesh)).compile(source).module
    plan = fm.verify_buffer_plan(module)
    assert plan.buffer_map["private_stats"].storage == "block_local_data"
    assert plan.buffer_map["private_narrow"].storage == "block_local_data"
    assert plan.buffer_map["private_narrow"].distributed_storage_kind is (
        fm.DistributedBufferStorageKind.REPLICATED_LOCAL if full_replica
        else fm.DistributedBufferStorageKind.COMPACT_LOCAL
    )
    return module


@pytest.mark.parametrize("world", (2, 4))
@pytest.mark.parametrize("layout", ("cbb", "bcb"))
@pytest.mark.parametrize("full_replica", (False, True))
def test_real_private_producer_and_contained_view(world, layout, full_replica):
    private_view_stage(world, layout, full_replica)

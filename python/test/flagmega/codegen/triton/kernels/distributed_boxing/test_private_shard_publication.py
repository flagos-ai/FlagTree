# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A full private backing image does not enlarge its semantic shard's ownership."""

from collections import Counter
from itertools import product
from types import SimpleNamespace

import pytest
from triton.flagmega import ir as fm

from .test_local_narrowing import local_abi
from .test_partial_reduce_local_abi import _abi, _prepare


@pytest.mark.parametrize("coordinate_space", ("local", "canonical_global", "parent_shard_local"))
@pytest.mark.parametrize("lanes", (1, 4))
@pytest.mark.parametrize("split_kind", ("contiguous", "cyclic"))
def test_publication_visits_each_source_owner_only(coordinate_space, lanes, split_kind):
    mesh = fm.Placement((2, 4), "yx", "bb")
    dtype = "float32" if lanes == 1 else fm.vector_type("float32", lanes)
    width = 24 if split_kind == "contiguous" else 19
    tensor = fm.tensor_type(dtype, (2, width))
    split = (fm.SBP.split_contiguous((0, 1), 3) if split_kind == "contiguous" else
             fm.SBP.split_block_cyclic((0, 1), 2))
    source_type = fm.DistributedType(tensor, (fm.SBP.broadcast(), split), mesh)
    source = local_abi(source_type)
    source.update(coordinate_space=coordinate_space, memory_sharing_scope="block")
    if coordinate_space != "local":
        source.update(scalar_storage_strides=(width * lanes, lanes),
                      storage_coordinate_expressions=source["logical_coordinate_expressions"])
    result = _abi((2, width), lane_count=lanes)
    result.update(distributed_type=None, memory_sharing_scope="chip")
    leaf = _prepare((source,), (result,), ("tensor_store",))["leaves"][0]

    published = Counter()
    for y, x in product(range(2), range(4)):
        shard = fm.local_shard_descriptor(source_type, (y, x))
        shape = tuple(d.fixed_value for d in shard.local_capacity_shape)
        expected = set()
        for row in range(shard.active_shape[0].fixed_value):
            for column in range(shard.active_shape[1].fixed_value):
                global_row = shard.axes[0].map_local_to_global(row).fixed_value
                global_column = shard.axes[1].map_local_to_global(column).fixed_value
                for lane in range(lanes):
                    global_offset = (global_row * width + global_column) * lanes + lane
                    source_offset = ((row * shape[1] + column) * lanes + lane
                                     if coordinate_space == "local" else global_offset)
                    expected.add((source_offset, global_offset))
        actual = set()
        scope = {"shard_y": y, "shard_x": x, "shard_coord_0": y, "shard_coord_1": x,
                 "tl": SimpleNamespace(minimum=min, maximum=max)}
        if eval(leaf["writer_active"], {"__builtins__": {}}, scope):
            for offset in range(leaf["capacity"]):
                scope["boxing_offsets"] = offset
                if eval(leaf["active"], {"__builtins__": {}}, scope):
                    src = eval(leaf["source_offset"], {"__builtins__": {}}, scope)
                    dst = eval(leaf["result_offset"], {"__builtins__": {}}, scope)
                    actual.add((src, dst))
                    published[dst] += 1
        assert actual == expected, (y, x, actual - expected, expected - actual)
    assert published == Counter({offset: 1 for offset in range(2 * width * lanes)})


@pytest.mark.parametrize("lanes", (1, 4))
def test_private_reduction_then_narrow_consumer_publishes_only_current_values(tmp_path, lanes):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Runtime

    mesh = fm.Placement((8, 16), "yx", "bb")
    tensor = fm.tensor_type("float32", (128, 1, 4096))
    distributed = fm.DistributedType(tensor, (fm.SBP.split_contiguous((0, 1), 1),
        fm.SBP.broadcast(), fm.SBP.broadcast()), mesh)

    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", tensor, id="x")
            local = fm.F.distributed.sharded_view(x, distributed)
            partial = fm.F.math.reduce_sum(local, axes=(0,), keep_dims=False,
                                          output_lanes=() if lanes == 1 else (lanes,))
            reduced = fm.F.distributed.force_boxing(partial, fm.DistributedType(
                partial.type.tensor, (fm.SBP.broadcast(),) * 2, mesh), name="allreduce")
            view = fm.F.distributed.sharded_view(reduced, fm.DistributedType(
                reduced.type.tensor, (fm.SBP.broadcast(),
                    fm.SBP.split_contiguous((0, 1), 32 // lanes)), mesh))
            consumed = (fm.F.math.add(view, view) if lanes == 1 else
                        fm.F.math.vectorized_binary(view, view, binary_op="add"))
            result = fm.F.distributed.force_boxing(consumed, consumed.type.tensor)
            self.function("main", (x,), (result,))

    module = Compiler().compile(Graph(dialect="distributed", stage="frozen_constants", entry="main",
        metadata={"auto_distribution": {"placement": mesh.to_data()}}).build()).module
    assert fm.verify_buffer_plan(module).buffer_map["allreduce"].storage == "block_local_data"
    runtime = load(write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True),
                   device="cuda:0")
    try:
        x = torch.empty((128, 1, 4096), device="cuda")
        storage = torch.full((4096 + 64,), -123., device="cuda")
        output = storage[32:-32].reshape((1, 4096 // lanes, *(() if lanes == 1 else (lanes,))))
        arguments = tuple(output if spec["role"] == "result" else x for spec in runtime.external_arguments)
        Runtime.prepare(runtime, *arguments)
        indices = torch.arange(x.numel(), device="cuda", dtype=torch.int64).reshape(x.shape)
        for mutation in range(3):
            x.copy_(((indices * 3 + mutation * 5) % 31 - 15).float() / 8)
            output.fill_(float("nan"))
            Runtime.run_into(runtime, *arguments)
            torch.cuda.synchronize()
            expected = (x.sum(0) * 2).reshape(output.shape)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            assert bool((storage[:32] == -123.).all() & (storage[-32:] == -123.).all())
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Bounded pinning and independent byte-transfer proofs for readonly sources."""
import mmap
from dataclasses import replace
from math import prod

import numpy as np
import pytest
import torch
from safetensors.numpy import save_file
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import pack_rdata, verify_rdata
from triton.flagmega.importer import DirectoryCheckpoint
from triton.flagmega.passes.constants import freeze_constant_islands
from triton.flagmega.passes.tir.bufferize.planner import BufferizationOptions
from triton.flagmega.passes.tir.bufferize.policy import NttBufferizationPolicy


def fixture(tmp_path, rows=4, cols=6):
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "config.json").write_text("{}")
    values = {"a": np.arange(rows * cols, dtype=np.float32).reshape(rows, cols),
              "b": np.arange(2 * rows * cols, dtype=np.float32).reshape(2 * rows, cols) + 100,
              "unrelated": np.zeros(512 * 1024, dtype=np.float32)}
    save_file(values, root / "model.safetensors")
    checkpoint = DirectoryCheckpoint(root)
    placement = fm.Placement((2, 2), "cx", "cb")
    b = fm.IRBuilder(dialect="tir", stage="selected_tir",
                     metadata={"launch_contract": {"grid_mesh": placement.to_data()}})
    args = [b.weight(key, fm.tensor_type("float32", values[key].shape),
                     source="checkpoint", key=key, id=key) for key in ("a", "b")]
    joined = b.call("tensors.concat", args, fm.tensor_type("float32", (3 * rows, cols)), attrs={"axis": 0}, id="joined")
    ty = fm.DistributedType(fm.tensor_type("float32", (3 * rows, cols)),
                            (fm.SBP.split_contiguous((0,)), fm.SBP.split_block_cyclic((1,), block_size=2)), placement)
    output = b.call("distributed.sharded_view", (joined,), ty, attrs={"new_type": ty}, id="output")
    b.function("main", (), (output,))
    module = freeze_constant_islands(b.build(entry="main"))
    module = NttBufferizationPolicy(replace(BufferizationOptions.generic(alignment=4),
                                            rdata_chip_sharding=True)).bufferize(module)
    return replace(module, stage="bufferized_tir"), checkpoint


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA readonly mappings")
@pytest.mark.parametrize("rows,cols", [(4, 6), (32, 66)])
def test_loader_respects_page_budget_and_cleans_up(tmp_path, monkeypatch, rows, cols):
    import triton.flagmega.runtime.source_rdata as loader
    module, checkpoint = fixture(tmp_path, rows, cols)
    assets = tmp_path / "assets"
    index = pack_rdata(module, checkpoint, assets)
    verify_rdata(assets, module=module)
    budget = mmap.PAGESIZE
    assert index["sources"][0]["nbytes"] > budget
    monkeypatch.setattr(loader, "PINNED_SOURCE_LIMIT", budget, raising=False)
    original = loader._RegisteredFile
    state = {"active": 0, "peak": 0, "opens": 0}

    class Tracked(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tracked_bytes = (self.array.nbytes + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE
            state["active"] += self.tracked_bytes
            state["peak"] = max(state["peak"], state["active"])
            state["opens"] += 1
            if state["active"] > budget:
                self.close()
                pytest.fail(f"readonly source registration exceeded {budget} bytes: peak={state['peak']}")

        def close(self):
            super().close()
            state["active"] -= self.tracked_bytes

    monkeypatch.setattr(loader, "_RegisteredFile", Tracked)
    for chip in (0, 1):
        expected = loader.load_source_rdata(assets, index, module, chip=chip, device="cpu")
        actual = loader.load_source_rdata(assets, index, module, chip=chip, device="cuda").cpu()
        assert torch.equal(actual, expected)
        assert state["active"] == 0
    assert 0 < state["peak"] <= budget


def geometries():
    yield [7, 6, 2], [14, 6, 2], [14, 6, 2], 0, [3, 11], [], [
        [(7, [(owner, 0, 2, 7)])] for owner in (0, 1)]
    yield [4, 6, 2], [16, 10, 2], [16, 10, 2], 1, [3, 7], [], [
        [(4, [(block, 2, 2, 4), (chip, 0, 2, 8)]), (5, [])]
        for chip in (0, 1) for block in (0, 1)]
    yield [4, 5, 3], [4, 5, 3], [5, 4, 3], 1, [1, 3], [
        {"shape": [4, 5, 3], "source_strides": [3, 12, 1]}], [[], [(0, [])]]
    yield [2, 3, 4, 2], [2, 3, 4, 2], [4, 12], 0, [1, 3], [
        {"shape": [2, 3, 4, 2], "source_strides": [24, 4, 1, 12]},
        {"shape": [2, 2, 12], "source_strides": [12, 24, 1]}], [[]]
    yield [97], [97], [97], 0, [31, 66], [], [[]]


@pytest.mark.parametrize("geometry", list(geometries()))
@pytest.mark.parametrize("budget", [4, 8, 32, 1024])
def test_plans_cover_exact_output_and_every_independent_source_read(geometry, budget):
    from triton.flagmega.runtime.source_rdata import byte_addresses
    from triton.flagmega.runtime.source_rdata_windows import source_transfers
    shape, global_shape, concat, axis, extents, transforms, maps = geometry
    operands = [{"source": i % 2, "offset": 17 + i * 333,
                 "extent": extent, "nbytes": prod(concat) // concat[axis] * extent}
                for i, extent in enumerate(extents)]
    program = {"shape": global_shape, "concat_shape": concat, "axis": axis,
               "operands": operands, "transforms": transforms}
    sources = [{"nbytes": 8191}, {"nbytes": 8193}]
    layouts = [(owner * prod(shape), shape, owner_maps) for owner, owner_maps in enumerate(maps)]
    covered = np.zeros((len(layouts), prod(shape)), dtype=np.int32)
    transfers = list(source_transfers(program, layouts, sources, budget=budget, page=4))
    for transfer in transfers:
        assert sum((w.end - w.begin + 3) // 4 * 4 for w in transfer.windows) <= budget
        for window in transfer.windows:
            assert window.begin % 4 == 0 and 0 <= window.begin < window.end <= sources[window.source]["nbytes"]
        for owner in range(transfer.owner_begin, transfer.owner_end):
            covered[owner, transfer.start:transfer.start + transfer.count] += 1
            child, address, valid = byte_addresses(program, shape, layouts[owner][2], transfer.start, transfer.count)
            for part, byte in zip(child[valid], address[valid], strict=True):
                operand = operands[part]
                slot = transfer.window_indices[owner - transfer.owner_begin][part]
                assert slot is not None
                window = transfer.windows[slot]
                assert window.source == operand["source"]
                assert window.begin <= operand["offset"] + byte < window.end
    np.testing.assert_array_equal(covered, np.ones_like(covered))


def test_pin_budget_cannot_be_smaller_than_one_host_page():
    from triton.flagmega.errors import ArtifactError
    from triton.flagmega.runtime.source_rdata_windows import source_transfers
    with pytest.raises(ArtifactError, match="one host page"):
        list(source_transfers({}, [], [], budget=4095, page=4096))


def test_strided_owners_share_registered_pages_instead_of_repinning_the_matrix():
    from triton.flagmega.runtime.source_rdata_windows import source_transfers
    shape, global_shape = (64, 128), (64, 4096)
    size = prod(global_shape)
    program = {"shape": global_shape, "concat_shape": global_shape, "axis": 0, "transforms": [],
               "operands": [{"source": 0, "offset": 0, "extent": 64, "nbytes": size}]}
    layouts = [(i * prod(shape), shape, [(64, []), (128, [(i, 4, 32, 128)])]) for i in range(32)]
    transfers = list(source_transfers(program, layouts, [{"nbytes": size}], budget=16384))
    assert sum(w.end - w.begin for transfer in transfers for w in transfer.windows) <= size * 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA readonly mappings")
@pytest.mark.parametrize("failure", ["registration", "after_launch"])
def test_exception_unwinds_only_after_queued_reads_finish(tmp_path, monkeypatch, failure):
    import triton.flagmega.runtime.source_rdata as loader
    import triton.flagmega.runtime.source_rdata_kernel as kernel
    module, checkpoint = fixture(tmp_path, rows=max(32, mmap.PAGESIZE // 256), cols=66)
    assets = tmp_path / "assets"
    index = pack_rdata(module, checkpoint, assets)
    monkeypatch.setattr(loader, "PINNED_SOURCE_LIMIT", mmap.PAGESIZE)
    original = loader._RegisteredFile
    synchronize = torch.cuda.synchronize
    state = {"active": 0, "attempts": 0, "queued": False}

    class Tracked(original):
        def __init__(self, *args, **kwargs):
            state["attempts"] += 1
            if failure == "registration" and state["attempts"] == 2:
                raise RuntimeError("injected registration failure")
            super().__init__(*args, **kwargs)
            state["active"] += 1

        def close(self):
            assert not state["queued"], "unregistered memory still in use by queued GPU reads"
            super().close()
            state["active"] -= 1

    def synchronized(*args, **kwargs):
        result = synchronize(*args, **kwargs)
        state["queued"] = False
        return result

    actual = kernel.gather_source_bytes

    class Launch:
        def __getitem__(self, grid):
            def run(*args, **kwargs):
                result = actual[grid](*args, **kwargs)
                state["queued"] = True
                if failure == "after_launch":
                    raise RuntimeError("injected post-launch failure")
                return result
            return run

    monkeypatch.setattr(loader, "_RegisteredFile", Tracked)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronized)
    monkeypatch.setattr(kernel, "gather_source_bytes", Launch())
    with pytest.raises(RuntimeError, match="injected"):
        loader.load_source_rdata(assets, index, module, chip=0, device="cuda")
    assert state["active"] == 0 and not state["queued"]

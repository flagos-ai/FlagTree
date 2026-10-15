# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Checkpoint tensor views share a shard mapping without retaining stale files."""

import json
from concurrent.futures import ThreadPoolExecutor

import torch
from safetensors.torch import save_file

from triton.flagmega.importer import DirectoryCheckpoint


def _checkpoint(root, count=64):
    root.mkdir()
    (root / "config.json").write_text("{}")
    values = {f"w{i:03d}": torch.arange(32, dtype=torch.float32) + i for i in range(count)}
    save_file(values, root / "model.safetensors")
    return DirectoryCheckpoint(root), values


def test_tensor_views_share_one_shard_mapping(tmp_path):
    checkpoint, values = _checkpoint(tmp_path / "checkpoint")
    tensors = {key: checkpoint.load_tensor(key) for key in values}
    first = next(iter(values))
    first_range = checkpoint.tensor_byte_range(first)
    first_pointer = tensors[first].data_ptr()
    for key, expected in values.items():
        torch.testing.assert_close(tensors[key], expected, rtol=0, atol=0)
        # Distances within one mapping equal byte distances in the source
        # shard. Opening a new mapping for every weight cannot satisfy this.
        assert tensors[key].data_ptr() - first_pointer == checkpoint.tensor_byte_range(key).offset - first_range.offset


def test_mapping_refreshes_after_atomic_source_replacement(tmp_path):
    checkpoint, values = _checkpoint(tmp_path / "checkpoint", count=1)
    old = checkpoint.load_tensor("w000")
    replacement = tmp_path / "replacement.safetensors"
    save_file({"w000": values["w000"] + 100}, replacement)
    replacement.replace(checkpoint.root / "model.safetensors")
    current = checkpoint.load_tensor("w000")
    torch.testing.assert_close(old, values["w000"], rtol=0, atol=0)
    torch.testing.assert_close(current, values["w000"] + 100, rtol=0, atol=0)


def test_parallel_tensor_reads_keep_the_shared_mapping(tmp_path):
    checkpoint, values = _checkpoint(tmp_path / "checkpoint")
    keys = tuple(values)
    with ThreadPoolExecutor(max_workers=8) as workers:
        tensors = tuple(workers.map(checkpoint.load_tensor, keys))
    base = checkpoint.tensor_byte_range(keys[0]).offset
    for key, tensor in zip(keys, tensors):
        torch.testing.assert_close(tensor, values[key], rtol=0, atol=0)
        assert tensor.data_ptr() - tensors[0].data_ptr() == checkpoint.tensor_byte_range(key).offset - base


def test_evicted_shard_reader_does_not_invalidate_live_tensors(tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "config.json").write_text("{}")
    weight_map = {}
    for i in range(130):
        filename = f"shard-{i}.safetensors"
        key = f"w{i}"
        save_file({key: torch.tensor([i], dtype=torch.float32)}, root / filename)
        weight_map[key] = filename
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    checkpoint = DirectoryCheckpoint(root)
    tensors = [checkpoint.load_tensor(key) for key in weight_map]
    for i, tensor in enumerate(tensors):
        assert tensor.item() == i

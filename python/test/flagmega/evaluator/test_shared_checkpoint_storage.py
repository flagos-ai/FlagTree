# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import numpy as np

from triton.flagmega import ir as fm
from triton.flagmega.evaluator.numpy_backend import iter_numpy_materialized_constant_assets, NumpyStorageReader
from triton.flagmega.importer import MemoryCheckpoint, TensorByteRange, TensorInfo
from triton.flagmega.passes.constants import freeze_constant_islands


def test_closed_recipes_share_one_readonly_mapping_per_checkpoint_file(tmp_path, monkeypatch):
    values = np.arange(64 * 8, dtype=np.float32).reshape(64, 8)
    backing = tmp_path / "storage.bin"
    backing.write_bytes(b"prefix!" + values.tobytes())
    builder = fm.IRBuilder(dialect="ntt", stage="canonical_constants")
    weights = tuple(builder.weight(f"w{i}", fm.tensor_type("float32", (8,)),
                                    source="storage", key=f"w{i}", id=f"w{i}") for i in range(64))
    builder.function("main", (), weights)
    module = freeze_constant_islands(builder.build(entry="main"))

    class StorageCheckpoint(MemoryCheckpoint):
        def tensor_byte_range(self, key):
            return TensorByteRange(backing, 7 + int(key[1:]) * 32, 32)

        def load_tensor(self, *args, **kwargs):
            raise AssertionError("storage materialization must not load Torch tensors")

    checkpoint = StorageCheckpoint({}, {f"w{i}":TensorInfo(f"w{i}", fm.DType.FLOAT32, (8,), "storage")
                                         for i in range(64)})
    original = np.memmap
    mappings = []

    def observe(*args, **kwargs):
        mappings.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "memmap", observe)
    outputs = dict(iter_numpy_materialized_constant_assets(module, checkpoint))
    assert mappings == [backing]
    for i in range(64):
        np.testing.assert_array_equal(outputs[f"w{i}"], values[i])
        assert not outputs[f"w{i}"].flags.writeable


def test_mapping_eviction_keeps_live_array_storage(tmp_path):
    reader = NumpyStorageReader()
    arrays = []
    for i in range(130):
        path = tmp_path / f"{i}.bin"
        path.write_bytes(np.array([i], dtype=np.int32).tobytes())
        arrays.append(reader.array(TensorByteRange(path, 0, 4), np.int32, (1,)))
    assert len(reader.mappings) == 128
    assert [int(array[0]) for array in arrays] == list(range(130))

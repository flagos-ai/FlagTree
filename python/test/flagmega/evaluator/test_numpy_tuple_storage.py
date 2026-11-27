# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import EvaluationError
from triton.flagmega.evaluator.numpy_backend import (
    NumpyMaterializationContext, _validate_storage_value, materialize_numpy_recipe,
)
from triton.flagmega.importer.checkpoint import MemoryCheckpoint, TensorByteRange, TensorInfo
from triton.flagmega.ir.distributed_inference import broadcast_ir_type
from triton.flagmega.passes.constants import freeze_constant_islands


def test_numpy_nested_tuple_boxing_retains_original_storage_without_tensor_loading(tmp_path):
    raw = np.arange(272, dtype=np.uint8).reshape(2, 136)
    scale = np.array([0x3f80, 0x8000], dtype=np.uint16)
    path = tmp_path / "weights.bin"
    path.write_bytes(raw.tobytes() + scale.tobytes())
    raw_type = fm.tensor_type("uint8", raw.shape)
    scale_type = fm.tensor_type("bfloat16", scale.shape)
    infos = {"raw": TensorInfo("raw", fm.DType.UINT8, raw.shape, str(path)),
             "scale": TensorInfo("scale", fm.DType.BFLOAT16, scale.shape, str(path))}

    class StoredCheckpoint(MemoryCheckpoint):
        def tensor_byte_range(self, key):
            return TensorByteRange(path, 0, raw.nbytes) if key == "raw" else TensorByteRange(path, raw.nbytes, scale.nbytes)

        def load_tensor(self, key, *, device="cpu"):
            raise AssertionError("Readonly tuple recipes must preserve stored bytes.")

    builder = fm.IRBuilder(dialect="nn", stage="canonical_constants")
    a = builder.weight("raw", raw_type, source=str(path), key="raw", id="raw")
    b = builder.weight("scale", scale_type, source=str(path), key="scale", id="scale")
    pair_type = fm.TupleType((raw_type, scale_type))
    pair = builder.call("builtin.tuple", (a, b), pair_type, id="pair")
    nested_type = fm.TupleType((pair_type, raw_type))
    nested = builder.call("builtin.tuple", (pair, a), nested_type, id="nested")
    target = broadcast_ir_type(nested_type, fm.Placement((1,), "x", "c"))
    boxed = builder.call("distributed.boxing", (nested,), target, attrs={"new_type": target}, id="boxed")
    projected = builder.call("builtin.get_item", (boxed,), target.fields[0], attrs={"index": 0}, id="projected")
    builder.function("main", (), (projected,))
    module = freeze_constant_islands(fm.verify_module(builder.build(entry="main")))
    recipe, = module.constant_recipes
    result, = materialize_numpy_recipe(module, recipe, StoredCheckpoint({}, infos))
    assert result[0].dtype == np.uint8 and result[0].tobytes() == raw.tobytes()
    assert result[1].dtype == np.uint16 and result[1].tobytes() == scale.tobytes()


@pytest.mark.parametrize("bad", [(), (np.zeros(2, dtype=np.uint8),),
                                  ((np.zeros(2, dtype=np.float32),),)])
def test_numpy_tuple_validation_checks_nested_storage_fields(bad):
    value_type = fm.TupleType((fm.TupleType((fm.tensor_type("uint8", (2,)),)),))
    node = fm.Node("nested", "builtin.tuple", (), value_type)
    builder = fm.IRBuilder(dialect="nn", stage="imported")
    builder.function("main", (), ())
    module = builder.build(entry="main")
    context = NumpyMaterializationContext(module, MemoryCheckpoint({}, {}))
    with pytest.raises(EvaluationError):
        _validate_storage_value(node, bad, context)

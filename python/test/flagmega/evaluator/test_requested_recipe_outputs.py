# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import numpy as np
import pytest
import torch
from safetensors.numpy import save_file
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import pack_rdata, verify_rdata
from triton.flagmega.evaluator.numpy_backend import iter_numpy_materialized_constant_assets
from triton.flagmega.evaluator.torch_backend import CheckpointWeightResolver, iter_materialized_constant_assets
from triton.flagmega.importer import DirectoryCheckpoint, MemoryCheckpoint, TensorInfo
from triton.flagmega.passes.tir.bufferize.planner import BufferizationOptions
from triton.flagmega.passes.tir.bufferize.policy import NttBufferizationPolicy


def mixed_recipe_module():
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    large = builder.weight("large", fm.tensor_type("float32", (4, 8)), source="checkpoint", key="large", id="large")
    packed = builder.call("tensors.pack", (large,), fm.tensor_type(fm.vector_type("float32", 2), (4, 4)),
                          attrs={"lanes": (2,), "axes": (1,)}, id="packed")
    small = builder.weight("small", fm.tensor_type("float32", (4,)), source="checkpoint", key="small", id="small")
    numeric = builder.call("math.silu", (small,), small.type, id="numeric")
    recipe = fm.ConstantRecipe("mixed", (large, packed, small, numeric), (packed.id, numeric.id))
    assets = tuple(fm.Node(node.id, "builtin.const_asset", (), node.type,
                           attrs={"recipe": recipe.id, "output": node.id}) for node in (packed, numeric))
    return fm.verify_module(fm.IRModule(
        "tir", "constants_frozen", assets, (fm.Function("main", (), tuple(node.id for node in assets)),), "main",
        metadata={"constant_phase": "frozen"}, constant_recipes=(recipe,),
    ))


@pytest.mark.parametrize("backend", ("numpy", "torch"))
def test_streaming_requested_numeric_output_never_loads_unrequested_weight(backend):
    module = mixed_recipe_module()
    calls = []
    class Checkpoint(MemoryCheckpoint):
        def load_tensor(self, key, *, device="cpu"):
            calls.append(key)
            assert key != "large", "Unrequested address-only output loaded a weight numerically"
            return super().load_tensor(key, device=device)
    values = {"small": torch.tensor([-2., -1., 0., 1.]), "large": torch.arange(32.).reshape(4, 8)}
    checkpoint = Checkpoint({}, {key:TensorInfo(key, fm.DType.FLOAT32, tuple(value.shape), "memory")
                                 for key, value in values.items()}, values)
    if backend == "numpy":
        result = dict(iter_numpy_materialized_constant_assets(module, checkpoint, outputs=("numeric",)))
    else:
        result = dict(iter_materialized_constant_assets(module, CheckpointWeightResolver(checkpoint, cache=False),
                                                        outputs=("numeric",)))
    assert list(result) == ["numeric"]
    torch.testing.assert_close(result["numeric"], torch.nn.functional.silu(values["small"]), rtol=0, atol=0)
    assert calls == ["small"]


def test_unrequested_numeric_branch_does_not_force_numpy_to_torch(monkeypatch):
    import triton.flagmega.evaluator.torch_backend as semantic
    module = mixed_recipe_module()
    values = {"small": torch.tensor([-2., -1., 0., 1.]), "large": torch.arange(32.).reshape(4, 8)}
    checkpoint = MemoryCheckpoint({}, {key:TensorInfo(key, fm.DType.FLOAT32, tuple(value.shape), "memory")
                                       for key, value in values.items()}, values)
    def forbidden(*args, **kwargs):
        pytest.fail("Unrequested numeric branch forced the representation-only branch into Torch")
    monkeypatch.setattr(semantic, "materialize_constant_recipe", forbidden)
    result = dict(iter_numpy_materialized_constant_assets(module, checkpoint, outputs=("packed",)))
    np.testing.assert_array_equal(result["packed"], values["large"].numpy().reshape(4, 4, 2))


def test_source_publisher_materializes_only_numeric_outputs_of_a_mixed_recipe(tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "config.json").write_text("{}")
    values = {"large": np.arange(32, dtype=np.float32).reshape(4, 8),
              "small": np.array([-2., -1., 0., 1.], dtype=np.float32)}
    save_file(values, root / "model.safetensors")
    calls = []
    class Checkpoint(DirectoryCheckpoint):
        def load_tensor(self, key, *, device="cpu"):
            calls.append(key)
            assert key != "large", "Virtual source output must never materialize numerically"
            return super().load_tensor(key, device=device)
    module = mixed_recipe_module()
    module = NttBufferizationPolicy(BufferizationOptions.generic(alignment=4)).bufferize(module)
    module = replace(module, stage="bufferized_tir")
    assets = tmp_path / "assets"
    index = pack_rdata(module, Checkpoint(root), assets)
    assert index["programs"] and len(index["programs"]) == 1
    assert index["programs"][0]["buffer"] == "packed"
    verify_rdata(assets, module=module)
    buffer = fm.verify_buffer_plan(module).buffer_map["numeric"]
    raw = (assets / "rdata.bin").read_bytes()
    actual = np.frombuffer(raw, dtype=np.float32, count=4, offset=buffer.offset)
    np.testing.assert_array_equal(actual, torch.nn.functional.silu(torch.from_numpy(values["small"])).numpy())
    assert calls == ["small"]

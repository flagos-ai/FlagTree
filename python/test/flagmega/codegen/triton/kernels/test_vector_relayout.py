# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace
from math import prod

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.runtime import load
from python.test.flagmega.ir.ops.primitive_helpers import primitive_module


@pytest.mark.parametrize("op", ["pack", "unpack"])
@pytest.mark.parametrize("axes,lanes", [((0, ), (4, )), ((1, 0), (2, 4)), ((0, 0), (2, 2))])
@pytest.mark.parametrize("old_lanes,split", [((), False), ((2, 2), False), ((2, 2), True)])
def test_local_vector_relayout_preserves_lane_order_and_owner_coordinates(tmp_path, op, axes, lanes, old_lanes, split):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    dtype = fm.vector_type("bfloat16", old_lanes) if old_lanes else "bfloat16"
    logical = fm.tensor_type(dtype, (16, 6))
    metadata = {}
    if split:
        placement = fm.Placement((2, 2), "xy", "bb")
        logical = fm.DistributedType(logical, (fm.SBP.split_contiguous((0, )), fm.SBP.broadcast()), placement)
        metadata = {"auto_distribution": {"placement": placement.to_data()}}
    value_type = logical
    attrs = {"axes": axes, "lanes": lanes}
    if op == "unpack":
        value_type = fm.get_definition("tensors.pack").infer_type((fm.Node("input", "builtin.var", (), logical), ),
                                                                  attrs)
        attrs = {"axes": axes}
    module = replace(primitive_module(fm.get_definition(f"tensors.{op}"), (value_type, ), **attrs),
                     stage="frozen_constants", metadata=metadata)
    tensor = value_type.tensor if isinstance(value_type, fm.DistributedType) else value_type
    shape = (*[extent.fixed_value for extent in tensor.shape], *getattr(tensor.dtype, "lanes", ()))
    values = torch.arange(prod(shape), device="cuda").reshape(shape).remainder(197).bfloat16()
    expected = TorchEvaluator(DictWeightResolver({})).run(module, {"value": values})[0]
    compiled = Compiler().compile(module).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    runtime.prepare(values)
    actual = runtime.run(values)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Six independent rounding boundaries, field layouts, and chip head ownership."""

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn.kda_input_projection import KdaInputProjection


def graph(*, lanes=((),) * 6, input_lanes=(), k=512, heads=8, width=16, rank_f=64, rank_g=128,
          placement=None, head=None, **changes):
    shapes = ((1, k), *((heads, width, k),) * 3, (heads, k), (rank_f, k), (rank_g, k))
    names = ("x", "q", "k", "v", "b", "f", "g")

    class Graph(fm.Module):
        def forward(self):
            inputs, values = [], []
            for i, (name, shape) in enumerate(zip(names, shapes, strict=True)):
                dtype = fm.vector_type("bfloat16", input_lanes) if i == 0 and input_lanes else "bfloat16"
                if i == 0 and input_lanes:
                    shape = (1, k // dtype.lane_count)
                kind = changes.get(name, fm.tensor_type(dtype, shape))
                x = self.input(name, kind, id=name)
                inputs.append(x)
                if placement is not None:
                    policies = [fm.SBP.broadcast()] * kind.rank
                    if 1 <= i <= 4:
                        policies[0] = head or fm.SBP.broadcast()
                    x = fm.F.distributed.sharded_view(x, fm.DistributedType(kind, tuple(policies), placement))
                values.append(x)
            result = fm.F.nn.kda_input_projection(*values, output_lanes=lanes, name="projection")
            self.function("main", inputs, fm.F.tensors.get_items(result, *range(6)))

    return Graph(dialect="distributed" if placement else "high_level",
                 stage="frozen_constants" if placement else "imported", entry="main",
                 metadata={"auto_distribution": {"placement": placement.to_data()}} if placement else {}).build()


@pytest.mark.parametrize("packed", [False, True])
def test_independent_rounding_and_edit_resume(tmp_path, packed):
    lanes = ((2, 4), (4,), (), (), (8,), (2, 4)) if packed else ((),) * 6
    module = graph(lanes=lanes, input_lanes=(2, 4))
    restored = fm.load_module(fm.emit_module(module, tmp_path / "projection.py"))
    assert restored.semantic_hash == module.semantic_hash
    generator = torch.Generator().manual_seed(951)
    values = {node.id: torch.randn(tuple(d.fixed_value for d in node.type.shape) +
                                  tuple(getattr(node.type.dtype, "lanes", ())), generator=generator).bfloat16()
              for node in module.nodes if node.op == "builtin.var"}
    actual = TorchEvaluator(DictWeightResolver({})).run(restored, values)
    x = values["x"].reshape(1, -1).float()
    for result, name in zip(actual, ("q", "k", "v", "b", "f", "g"), strict=True):
        expected = (x @ values[name].reshape(-1, 512).float().T).bfloat16()
        torch.testing.assert_close(result.reshape(expected.shape), expected, rtol=0, atol=0)
    node = module.node_map["projection"]
    assert node.effect == fm.effect("pure")
    assert KdaInputProjection.cost(node).bytes_written == 2 * (3 * 8 * 16 + 8 + 64 + 128)


@pytest.mark.parametrize("change", [
    {"lanes": ((),) * 5}, {"lanes": ((), (), (), (), (True,), ())},
    {"lanes": ((), (), (), (), (3,), ())}, {"x": fm.tensor_type("bfloat16", (2, 512))},
    {"q": fm.tensor_type("bfloat16", (128, 512))}, {"v": fm.tensor_type("bfloat16", (8, 32, 512))},
    {"b": fm.tensor_type("bfloat16", (16, 512))}, {"f": fm.tensor_type("float32", (64, 512))},
    {"g": fm.tensor_type("bfloat16", (128, 256))},
])
def test_invalid_contracts_are_rejected(change):
    with pytest.raises(IRSchemaError, match="KdaInputProjection"):
        graph(**change)


def test_per_field_distribution_and_inverse():
    placement = fm.Placement((4, 8, 16), "cyx", "cbb")
    module = graph(placement=placement, head=fm.SBP.split_contiguous((0,)))
    node = module.node_map["projection"]
    for i, field in enumerate(node.type.fields):
        assert field.axis_policies[1] == (fm.SBP.split_contiguous((0,)) if i < 4 else fm.SBP.broadcast())
    inputs = tuple(module.node_map[name].type for name in node.inputs)
    logical = tuple(t.tensor for t in inputs)
    assert KdaInputProjection.infer_distributed_input_types(node.type, logical, node.attrs) == (inputs,)
    assert tuple(KdaInputProjection.distributed_input_type_tuples(tuple((t,) for t in inputs), node.attrs)) == (inputs,)


@pytest.mark.parametrize("head", [fm.SBP.split_contiguous((1,)), fm.SBP.split_block_cyclic((0,), 1)])
def test_unsupported_head_ownership_is_rejected(head):
    with pytest.raises(IRSchemaError, match="chip head ownership"):
        graph(placement=fm.Placement((4, 2), "cy", "cb"), head=head)


def test_normal_compilation_supports_six_typed_results():
    from triton.flagmega.compiler import Compiler
    compiled = Compiler().compile(graph()).module
    dispatch, = (function.dispatch for function in compiled.kernel_definitions
                 if function.dispatch.semantic_op == "nn.kda_input_projection")
    assert len(dispatch.outputs) == 6

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Material P-axis FP8 projections preserve each partition's BF16 boundary."""

from dataclasses import replace
from math import prod

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn.partitioned_block_scaled_linear import PartitionedBlockScaledLinear as Op


class Graph(fm.Module):
    def __init__(self, *, tokens=2, k=1024, n=80, parts=4, input_lanes=(), output_lanes=(),
                 explicit=False, quantization="dynamic_fp8_sm90", kinds=None, **kwargs):
        super().__init__(dialect="high_level", stage="imported", entry="main", **kwargs)
        self.tokens, self.k, self.n, self.parts = tokens, k, n, parts
        self.input_lanes, self.output_lanes = input_lanes, output_lanes
        self.explicit, self.quantization, self.kinds = explicit, quantization, kinds

    def forward(self):
        types = self.kinds or (
            fm.tensor_type(fm.vector_type("bfloat16", self.input_lanes) if self.input_lanes else "bfloat16",
                           (self.tokens, self.k // prod(self.input_lanes))),
            fm.tensor_type("float8_e4m3fn", (self.n, self.k)),
            fm.tensor_type("float32", ((self.n + 127) // 128, self.k // 128)),
        )
        x, w, scales = (self.input(name, kind, id=name) for name, kind in zip(("x", "weight", "scales"), types))
        if self.explicit:
            x = fm.F.tensors.unpack(x, axes=(-1,) * len(self.input_lanes)) if self.input_lanes else x
            span, values = self.k // self.parts, []
            for p in range(self.parts):
                def sliced(value, start, end):
                    return fm.F.tensors.slice(value, starts=(start,), ends=(end,), axes=(1,))
                local_x, local_w, local_scale = (sliced(x, p * span, (p + 1) * span),
                    sliced(w, p * span, (p + 1) * span), sliced(scales, p * span // 128, (p + 1) * span // 128))
                # The existing device BlockScaledMatMul renderer implements
                # one row. Use independent rows as the multi-token reference;
                # the new producer still executes all tokens in one operation.
                rows = []
                for token in range(self.tokens):
                    row = fm.F.tensors.slice(local_x, starts=(token,), ends=(token + 1,), axes=(0,))
                    rows.append(fm.F.math.block_scaled_matmul(row, local_w, local_scale,
                        weight_block_n=128, weight_block_k=128, activation_quantization=self.quantization))
                result = rows[0] if len(rows) == 1 else fm.F.tensors.concat(*rows, axis=0)
                values.append(fm.F.tensors.reshape(result, (1, self.tokens, self.n)))
            result = fm.F.tensors.concat(*values, axis=0)
            if self.output_lanes:
                result = fm.F.tensors.pack(result, self.output_lanes, axes=(-1,) * len(self.output_lanes))
        else:
            result = fm.F.nn.partitioned_block_scaled_linear(x, w, scales, partitions=self.parts,
                activation_quantization=self.quantization, output_lanes=self.output_lanes, name="projection")
        self.function("main", ("x", "weight", "scales"), (result,))


@pytest.mark.parametrize("input_lanes,output_lanes", [((), ()), ((2, 4), ()), ((), (2, 4)), ((2, 4), (2, 4))])
@pytest.mark.parametrize("quantization", ("dynamic_fp8", "dynamic_fp8_sm90"))
def test_matches_explicit_rounded_parts_and_python_resume(tmp_path, input_lanes, output_lanes, quantization):
    options = {"input_lanes": input_lanes, "output_lanes": output_lanes, "quantization": quantization}
    module, reference = Graph(**options).build(), Graph(explicit=True, **options).build()
    generator = torch.Generator().manual_seed(1762)
    inputs = {"x": torch.randn(2, 1024, generator=generator).bfloat16().reshape(2, 1024 // prod(input_lanes), *input_lanes),
              "weight": torch.randn(80, 1024, generator=generator).to(torch.float8_e4m3fn),
              "scales": torch.linspace(.01, .08, 8).reshape(1, 8)}
    evaluator = TorchEvaluator(DictWeightResolver({}))
    actual, = evaluator.run(module, inputs)
    expected, = evaluator.run(reference, inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (4, 2, 80 // prod(output_lanes), *output_lanes)
    restored = fm.load_module(fm.emit_module(module, tmp_path / "projection.py"))
    assert restored.semantic_hash == module.semantic_hash
    assert module.node_map["projection"].effect.is_pure


@pytest.mark.parametrize("axis", ("partition", "rows", "broadcast"))
@pytest.mark.parametrize("lanes", ((), (2, 4)))
def test_forward_inverse_distribution_preserves_material_partition(axis, lanes):
    module = Graph(tokens=1, k=2048, n=128, input_lanes=lanes, output_lanes=lanes).build()
    placement, b = fm.Placement((4, 8, 16), "cyx", "cbb"), fm.SBP.broadcast()
    p = fm.SBP.split_contiguous((0,)) if axis == "partition" else b
    n = fm.SBP.split_contiguous((0,)) if axis == "rows" else b
    attrs = module.node_map["projection"].attrs
    logical = tuple(module.node_map[name].type for name in ("x", "weight", "scales"))
    kinds = (fm.DistributedType(logical[0], (b, p), placement), fm.DistributedType(logical[1], (n, p), placement),
             fm.DistributedType(logical[2], (b, b), placement))
    operands = [replace(module.node_map[name], type=kind) for name, kind in zip(("x", "weight", "scales"), kinds)]
    output = Op.infer_type(operands, attrs)
    assert output.axis_policies == (p, b, n) and output.partial is None
    assert Op.infer_distributed_input_types(output, logical, attrs) == (kinds,)
    if axis == "partition":
        assert tuple(d.fixed_value for d in fm.local_tensor_type(output).shape) == (1, 1, 128 // prod(lanes))


@pytest.mark.parametrize("options", ({"k": 768}, {"k": 1024, "parts": 3}, {"n": 81, "output_lanes": (8,)}))
def test_rejects_incomplete_quantization_parts_or_lanes(options):
    with pytest.raises(IRSchemaError):
        Graph(**options).build()


@pytest.mark.parametrize("name,value", [("partitions", 0), ("partitions", True), ("weight_block_k", -1),
    ("weight_block_n", []), ("activation_quantization", "none"), ("activation_quantization", []), ("output_lanes", (0,))])
def test_rejects_invalid_attributes(name, value):
    with pytest.raises(IRSchemaError):
        Op.normalize_attrs({"partitions": 4, name: value})


def test_rejects_splitting_a_semantic_partition_or_a_block_owner():
    module = Graph(tokens=1).build()
    for mesh, axes in ((fm.Placement((2, 8, 16), "cyx", "cbb"), (0,)),
                       (fm.Placement((4, 8, 16), "cyx", "cbb"), (1,))):
        b, split = fm.SBP.broadcast(), fm.SBP.split_contiguous(axes)
        policies = ((b, split), (b, split), (b, b))
        inputs = [replace(module.node_map[name], type=fm.DistributedType(module.node_map[name].type, policy, mesh))
                  for name, policy in zip(("x", "weight", "scales"), policies)]
        with pytest.raises(IRSchemaError):
            Op.infer_type(inputs, module.node_map["projection"].attrs)

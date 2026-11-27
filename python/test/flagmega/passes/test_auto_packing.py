# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator


def _fp8_module():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    activation_type = fm.tensor_type("bfloat16", [1, 128])
    weight_type = fm.tensor_type("float8_e4m3fn", [128, 128])
    scale_type = fm.tensor_type("float32", [1, 1])
    source = builder.var("source", activation_type, id="source")
    weight = builder.weight("weight", weight_type, source="memory", key="weight", id="weight")
    scale = builder.weight("scale", scale_type, source="memory", key="scale", id="scale")
    output = builder.call(
        "math.block_scaled_matmul", [source, weight, scale], activation_type, id="output",
        attrs={"weight_block_n": 128, "weight_block_k": 128},
    )
    builder.function("main", [source], [output])
    return builder.build(entry="main")


def test_auto_packing_inserts_an_explicit_pack_and_preserves_fp8_semantics(tmp_path):
    module = _fp8_module()
    packed = Compiler().compile(module, stop_after="apply-packing").module
    output = packed.node_map["output"]
    packed_weight = packed.node_map[output.inputs[1]]
    assert output.op == "math.packed_block_scaled_matmul"
    assert packed_weight.op == "tensors.pack"
    assert packed_weight.inputs == ("weight",)
    assert packed_weight.type == fm.tensor_type(fm.vector_type("float8_e4m3fn", (2, 16)), (128, 4))
    assert packed.node_map["weight"].op == "builtin.weight"

    source = torch.randn(1, 128, dtype=torch.bfloat16)
    weight = torch.randn(128, 128, dtype=torch.float32).to(torch.float8_e4m3fn)
    resolver = DictWeightResolver({"weight": weight, "scale": torch.ones(1, 1)})
    evaluator = TorchEvaluator(resolver)
    torch.testing.assert_close(evaluator.run(packed, {"source": source})[0], evaluator.run(module, {"source": source})[0])
    path = fm.emit_module(packed, tmp_path / "packed.py")
    assert fm.load_module(path) == packed

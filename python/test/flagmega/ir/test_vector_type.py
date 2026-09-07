# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator


def test_repeated_axis_pack_is_backward_compatible():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            source = self.input("source", fm.tensor_type("bfloat16", [8, 3]), id="source")
            packed = fm.F.tensors.pack(source, (2, 2), axis=0, name="packed")
            output = fm.F.tensors.unpack(packed, axis=0, name="output")
            self.function("main", [source], [output])

    module = Graph().build()
    packed_type = module.node_map["packed"].type
    assert packed_type.shape == (fm.dim(2), fm.dim(3))
    assert packed_type.dtype == fm.vector_type("bfloat16", (2, 2))

    source = torch.arange(24, dtype=torch.bfloat16).reshape(8, 3)
    outputs, trace = TorchEvaluator(DictWeightResolver({})).run_with_trace(module, {"source": source})
    assert trace["packed"].shape == (2, 3, 2, 2)
    torch.testing.assert_close(outputs[0], source)


def test_multi_axis_pack_and_partial_unpack_follow_nncase_lane_order():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            source = self.input("source", fm.tensor_type("bfloat16", [4, 6]), id="source")
            packed = fm.F.tensors.pack(source, (2, 3), axes=(0, 1), name="packed")
            partial = fm.F.tensors.unpack(packed, axes=(0,), name="partial")
            output = fm.F.tensors.unpack(partial, axes=(1,), name="output")
            self.function("main", [source], [output])

    module = Graph().build()
    assert module.node_map["packed"].type == fm.tensor_type(fm.vector_type("bfloat16", (2, 3)), (2, 2))
    assert module.node_map["partial"].type == fm.tensor_type(fm.vector_type("bfloat16", (3,)), (4, 2))
    source = torch.arange(24, dtype=torch.bfloat16).reshape(4, 6)
    outputs, trace = TorchEvaluator(DictWeightResolver({})).run_with_trace(module, {"source": source})
    assert trace["packed"].shape == (2, 2, 2, 3)
    assert trace["partial"].shape == (4, 2, 3)
    torch.testing.assert_close(outputs[0], source)

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from collections import Counter

from triton.flagmega import ir as fm
from triton.flagmega.ir.ops.tensors.reshape import Reshape
from triton.flagmega.passes.auto_distributed import AutoDistributedPass
from triton.flagmega.targets import NvidiaSm90Target


def test_distinct_weight_transforms_share_type_analysis_not_graph_identity(monkeypatch):
    builder = fm.IRBuilder(dialect="high_level", stage="packed")
    tensor = fm.tensor_type("bfloat16", (256, 256))
    outputs = []
    for name in ("first", "second"):
        value = builder.weight(name, tensor, source="unit.safetensors", key=name, id=name)
        outputs.append(
            builder.call("tensors.reshape", (value, ), tensor, attrs={"shape": (256, 256)}, id=name + ".reshape"))
    builder.function("main", (), outputs)
    module = builder.build(entry="main")
    calls = Counter()
    original = Reshape.infer_type

    def counted(cls, inputs, attrs):
        calls[(tuple(v.type for v in inputs), tuple(attrs["shape"]))] += 1
        return original(inputs, attrs)

    monkeypatch.setattr(Reshape, "infer_type", classmethod(counted))
    graph = AutoDistributedPass._build_graph(module, NvidiaSm90Target())
    assert calls and max(calls.values()) == 1
    first, second = graph.bucket_map["first.reshape"], graph.bucket_map["second.reshape"]
    assert len(first.candidates) == len(second.candidates) > 1
    assert {c.id for c in first.candidates}.isdisjoint(c.id for c in second.candidates)
    assert {s.producer_id for s in graph.reshard_sites} >= {"first", "second"}

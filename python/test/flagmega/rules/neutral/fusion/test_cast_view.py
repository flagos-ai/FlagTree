# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.fusion import fusion_rules
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.rewriter import DataflowPass


def graph(kind, *, exported=False):
    mesh = fm.Placement((2, 2), "yx", "bb")
    narrow = fm.DistributedType(fm.tensor_type(fm.vector_type("bfloat16", (8,)), (2, 4)),
                                (fm.SBP.broadcast(), fm.SBP.broadcast()), mesh)

    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", narrow, id="x")
            wide = fm.F.ntt.vectorized_cast(x, fm.vector_type("float32", (2, 4)), (1,), name="wide")
            if kind == "sharded":
                result_type = fm.DistributedType(wide.type.tensor, (fm.SBP.split_contiguous((0,)), fm.SBP.broadcast()), mesh)
                view = fm.F.distributed.sharded_view(wide, result_type, name="view")
            else:
                view = fm.F.tensors.bitcast(wide, fm.vector_type("float32", (8,)), name="view")
            other = self.input("other", view.type, id="other")
            output = fm.F.math.vectorized_binary(other, view, binary_op="add", name="output", metadata={
                "selected_vectorization": "test.cast_view", "selected_vector_axes": (1,) * len(view.type.tensor.dtype.lanes),
                "selected_vector_lanes": view.type.tensor.dtype.lanes,
            })
            self.function("main", (x, other), (output, wide) if exported else (output,))

    return Graph(dialect="ntt", stage="frozen_constants", entry="main",
                 metadata={"auto_distribution": {"placement": mesh.to_data()}}).build()


@pytest.mark.parametrize("kind", ["sharded", "bitcast"])
def test_private_cast_through_view_fuses_without_changing_values(kind, tmp_path):
    source = graph(kind)
    fused = DataflowPass("Fuse", fusion_rules(), rewrite_constants=False).run(source)
    assert not any(node.op in {"tensors.cast", "ntt.vectorized_cast"} for node in fused.nodes)
    assert "pre_ops" in fused.node_map["output"].attrs
    assert fm.load_module(fm.emit_module(fused, tmp_path / "fused.py")) == fused
    x = torch.arange(64).reshape(2, 4, 8).bfloat16() / 16
    lanes = (2, 4) if kind == "sharded" else (8,)
    other = torch.linspace(-1, 1, 64).reshape(2, 4, *lanes)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(fused, {"x": x, "other": other}),
                               evaluator.run(source, {"x": x, "other": other}), rtol=0, atol=0)


def test_exported_cast_is_not_removed_through_its_view():
    source = graph("sharded", exported=True)
    fused = DataflowPass("Fuse", fusion_rules(), rewrite_constants=False).run(source)
    assert fused.node_map["wide"] == source.node_map["wide"]


def test_numeric_bit_reinterpretation_is_not_commuted():
    from triton.flagmega.rules.neutral.commute_cast_view import commute_cast_view_rules

    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type("bfloat16", (8,)), id="x")
            wide = fm.F.tensors.cast(x, "float32", name="wide")
            bits = fm.F.tensors.bitcast(wide, "int32", name="bits")
            self.function("main", (x,), (bits,))
    source = Graph(dialect="nn", stage="frozen_constants", entry="main").build()
    assert DataflowPass("Commute", commute_cast_view_rules(), rewrite_constants=False).run(source) == source


@pytest.mark.parametrize("kind", ["shared", "nontrailing"])
def test_cast_with_shared_uses_or_nontrailing_lane_axes_is_not_commuted(kind):
    from triton.flagmega.rules.neutral.commute_cast_view import commute_cast_view_rules

    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type(fm.vector_type("bfloat16", (8,)), (4, 4)), id="x")
            wide = fm.F.ntt.vectorized_cast(x, fm.vector_type("float32", (4,)),
                                          (0,) if kind == "nontrailing" else (1,), name="wide")
            view = fm.F.tensors.bitcast(wide, fm.vector_type("float32", (8,)), name="view")
            if kind == "shared":
                other = fm.F.tensors.bitcast(wide, "float32", name="other")
                self.function("main", (x,), (view, other))
            else:
                self.function("main", (x,), (view,))
    source = Graph(dialect="ntt", stage="frozen_constants", entry="main").build()
    assert DataflowPass("Commute", commute_cast_view_rules(), rewrite_constants=False).run(source) == source


@pytest.mark.parametrize("source_dtype,target_dtype", [("float32", "bfloat16"), ("bfloat16", "float32")])
def test_scalar_numeric_cast_commutes_with_lane_grouping_exactly(source_dtype, target_dtype):
    from triton.flagmega.rules.neutral.commute_cast_view import commute_cast_view_rules

    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type(source_dtype, (16,)), id="x")
            cast = fm.F.tensors.cast(x, target_dtype, name="cast")
            view = fm.F.tensors.bitcast(cast, fm.vector_type(target_dtype, (4,)), name="view")
            self.function("main", (x,), (view,))
    source = Graph(dialect="ntt", stage="frozen_constants", entry="main").build()
    result = DataflowPass("Commute", commute_cast_view_rules(), rewrite_constants=False).run(source)
    assert result.node_map["view"].op == "ntt.vectorized_cast"
    values = {"x": torch.linspace(-7, 7, 16).to(getattr(torch, source_dtype))}
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(result, values), evaluator.run(source, values), rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["sharded", "bitcast"])
def test_cast_view_fusion_executes_exact_gpu_values(kind, tmp_path):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.runtime import load

    source = graph(kind)
    compiled = Compiler().compile(source).module
    dispatches = [fm.kernel_dispatch_for_call(compiled, node) for node in compiled.nodes]
    assert not any(dispatch is not None and dispatch.semantic_op in {"tensors.cast", "ntt.vectorized_cast"}
                   for dispatch in dispatches)
    runtime = load(write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True),
                   device="cuda:0")
    x = torch.arange(64, device="cuda").reshape(2, 4, 8).bfloat16() / 16
    lanes = (2, 4) if kind == "sharded" else (8,)
    other = torch.linspace(-1, 1, 64, device="cuda").reshape(2, 4, *lanes)
    expected = TorchEvaluator(DictWeightResolver({})).run(source, {"x": x, "other": other})[0]
    runtime.prepare(x, other)
    torch.testing.assert_close(runtime.run(x, other), expected, rtol=0, atol=0)

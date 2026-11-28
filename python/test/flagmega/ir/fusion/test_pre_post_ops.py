# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.nn.softmax import Softmax


def make_graph():

    @fm.fusion(fm.tensor_type("bfloat16", (2, 16)))
    def widen(x):
        return fm.F.tensors.cast(x, "float32", name="wide")

    @fm.fusion(fm.tensor_type("float32", (2, 16)))
    def narrow(x):
        return fm.F.tensors.cast(x, "bfloat16", name="narrow")

    class Graph(fm.Module):

        def forward(self):
            x = self.input("x", widen.input_type, id="x")
            y = fm.F.with_ops(fm.F.nn.softmax, x, axis=-1, pre_ops={Softmax.value: widen}, post_ops=(narrow, ),
                              name="y")
            self.function("main", (x, ), (y, ))

    return Graph(dialect="high_level", stage="imported", entry="main").build()


def test_pre_base_post_inference_and_evaluation():
    module = fm.verify_module(make_graph())
    assert module.node_map["y"].type == fm.tensor_type("bfloat16", (2, 16))
    value = torch.randn(2, 16).bfloat16()
    result, = TorchEvaluator(DictWeightResolver({})).run(module, {"x": value})
    torch.testing.assert_close(result, value.float().softmax(-1).bfloat16(), rtol=0, atol=0)


def test_python_checkpoint_constructs_editable_fusion_bodies(tmp_path):
    module = make_graph()
    path = fm.emit_module(module, tmp_path / "ir.py")
    source = path.read_text()
    assert "@fm.fusion(" in source
    assert "F.with_ops(" in source
    assert "F.tensors.cast(" in source
    assert fm.load_module(path).semantic_hash == module.semantic_hash
    assert fm.IRModule.from_data(module.to_data()).semantic_hash == module.semantic_hash


def test_python_checkpoint_preserves_parameter_label_and_metadata(tmp_path):
    from dataclasses import replace
    module = make_graph()
    node = module.node_map["y"]
    body = node.attrs["pre_ops"]["value"]
    parameter = replace(body.parameter, attrs={"name": "storage_value"}, metadata={"role": "pre_input"})
    body = fm.Fusion(body.name, (parameter, *body.nodes[1:]), body.output)
    node = replace(node, attrs={**node.attrs, "pre_ops": {"value": body}})
    module = replace(module, nodes=tuple(node if value.id == node.id else value for value in module.nodes))
    assert fm.load_module(fm.emit_module(module, tmp_path / "parameter.py")).to_data() == module.to_data()


def test_fusion_rejects_free_values():
    external = fm.Node("external", "builtin.var", (), fm.tensor_type("float32", (2, 16)), attrs={"name": "external"})
    with pytest.raises(IRSchemaError, match="free|unknown|topological"):

        @fm.fusion(external.type)
        def invalid(x):
            return fm.F.math.add(x, external)


def test_pre_ops_rejects_parameter_from_another_op():
    from triton.flagmega.ir.ops.math.silu import Silu

    @fm.fusion(fm.tensor_type("float32", (2, 16)))
    def identity(x):
        return x

    class Graph(fm.Module):

        def forward(self):
            x = self.input("x", identity.input_type)
            fm.F.with_ops(fm.F.nn.softmax, x, pre_ops={Silu.value: identity})

    with pytest.raises(IRSchemaError, match="parameter|Parameter"):
        Graph(dialect="high_level", stage="imported", entry="main").build()

# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega import pattern_match as pm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.errors import EvaluationError, IRSchemaError, IRVerificationError
from triton.flagmega.ir.ops.tir.ref_slice import RefSlice
from python.test.flagmega.ir.ops.primitive_helpers import primitive_module


def reference_type(size=4):
    return fm.RefType("records", (("key", fm.tensor_type("int32", (size, 3))),
                                  ("value", fm.tensor_type(fm.vector_type("bfloat16", 8), (size, 2)))))


def reference_module(*, length=1, value_type=None, index_type=None):
    return primitive_module(RefSlice, (value_type or reference_type(), index_type or fm.tensor_type("int32", ())),
                            length=length)


@pytest.mark.parametrize("index,length", [(0, 1), (3, 1), (1, 2), (0, 4)])
def test_reference_slice_evaluator_aliases_each_typed_field(index, length):
    module = reference_module(length=length)
    values = {
        "key": torch.arange(12, dtype=torch.int32).reshape(4, 3), "value": torch.arange(64).reshape(4, 2, 8).bfloat16()
    }
    result = TorchEvaluator(DictWeightResolver({})).run(
        module, {"value": values, "index": torch.tensor(index, dtype=torch.int32)})[0]
    assert module.node_map["output"].type == reference_type(length)
    for name, field in values.items():
        assert result[name].untyped_storage().data_ptr() == field.untyped_storage().data_ptr()
        torch.testing.assert_close(result[name], field[index:index + length], rtol=0, atol=0)
        result[name].fill_(7)
        assert bool((field[index:index + length] == 7).all())


@pytest.mark.parametrize("length", [0, -1, 5, True, 1.5])
def test_reference_slice_rejects_invalid_length(length):
    with pytest.raises(IRSchemaError):
        reference_module(length=length)


@pytest.mark.parametrize("fields", [(), (("x", fm.tensor_type("int32", ())), ),
                                    (("x", fm.tensor_type("int32", (2, ))), ("y", fm.tensor_type("int32", (3, )))),
                                    (("x", fm.tensor_type("int32", (fm.dim("n", minimum=1, maximum=5), ))), )])
def test_reference_slice_rejects_opaque_scalar_mismatched_or_dynamic_leading_fields(fields):
    with pytest.raises(IRSchemaError):
        reference_module(value_type=fm.RefType("records", fields))


@pytest.mark.parametrize("index_type", [fm.tensor_type("int32", (1, )), fm.tensor_type("float32", ())])
def test_reference_slice_parameter_info_requires_scalar_integer(index_type):
    with pytest.raises(IRSchemaError):
        reference_module(index_type=index_type)


@pytest.mark.parametrize("index", [-1, 4])
def test_reference_slice_evaluator_rejects_out_of_bounds(index):
    module = reference_module()
    values = {"key": torch.zeros((4, 3), dtype=torch.int32), "value": torch.zeros((4, 2, 8), dtype=torch.bfloat16)}
    with pytest.raises(EvaluationError, match="outside"):
        TorchEvaluator(DictWeightResolver({})).run(module,
                                                   {"value": values, "index": torch.tensor(index, dtype=torch.int32)})


@pytest.mark.parametrize("boxed", [False, True])
def test_function_result_abi_does_not_silently_export_an_unbound_subspan(boxed):
    module = reference_module()
    if boxed:
        result = module.node_map["output"]
        box = fm.Node("box", "builtin.tuple", (result.id, ), fm.TupleType((result.type, )))
        module = replace(module, nodes=(*module.nodes, box), functions=(replace(module.functions[0],
                                                                                outputs=(box.id, )), ))
    plan = fm.make_buffer_plan(module)
    buffered = replace(module, dialect="bufferized_tir", stage="bufferized_tir",
                       metadata={"buffer_plan": plan.to_data()})
    with pytest.raises(IRVerificationError, match="reference subspan result"):
        fm.verify_buffer_plan(buffered)


@pytest.mark.parametrize("index", [-1, 4])
def test_reference_slice_rejects_invalid_constant_index_during_construction(index):

    class Graph(fm.Module):

        def forward(self):
            value = self.input("value", reference_type())
            position = fm.F.builtin.scalar_const(fm.tensor_type("int32", ()), index)
            result = fm.F.tir.ref_slice(value, position)
            self.function("main", (value, ), (result, ))

    with pytest.raises(IRSchemaError, match="outside"):
        Graph(dialect="tir", stage="imported", entry="main").build()


def test_reference_slice_exposes_static_functional_pattern_with_named_length():
    module = reference_module(length=2)
    node = module.node_map["output"]
    assert pm.try_match_root(node, pm.F.tir.is_ref_slice(length=2), module) is not None
    assert pm.try_match_root(node, pm.F.tir.is_ref_slice(length=1), module) is None

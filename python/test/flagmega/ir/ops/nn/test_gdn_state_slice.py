# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import EvaluationError, IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn._gdn_state import GatedDeltaNetStateConfig, create_gdn_state
from triton.flagmega.ir.ops.nn.gdn_state_slice import GatedDeltaNetStateSlice
from python.test.flagmega.ir.ops.primitive_helpers import primitive_module


def config():
    return GatedDeltaNetStateConfig(3, 1, 2, 4, 4, 4, 16)


@pytest.mark.parametrize("layer", [0, 1, 2])
def test_state_slice_is_an_alias_of_exactly_one_layer(layer):
    c = config()
    module = primitive_module(GatedDeltaNetStateSlice, (c.ref_type, fm.tensor_type("int32", ())))
    state = create_gdn_state(c)
    view = TorchEvaluator(DictWeightResolver({})).run(
        module, {"state": state, "layer_id": torch.tensor(layer, dtype=torch.int32)})[0]
    assert view.config == replace(c, num_layers=1)
    assert view.convolution.untyped_storage().data_ptr() == state.convolution.untyped_storage().data_ptr()
    assert view.recurrent.untyped_storage().data_ptr() == state.recurrent.untyped_storage().data_ptr()
    view.update_convolution_layer(torch.full((c.conv_dim, c.conv_kernel_size - 1), 2., dtype=torch.bfloat16))
    view.update_recurrent_layer(torch.full((c.num_value_heads, c.key_head_dim, c.value_head_dim), 3.))
    for index in range(c.num_layers):
        assert torch.all(state.convolution_layer(index) == (2 if index == layer else 0))
        assert torch.all(state.recurrent_layer(index) == (3 if index == layer else 0))
    assert module.node_map["output"].type == view.config.ref_type


@pytest.mark.parametrize("layer", [-1, 3])
def test_state_slice_rejects_out_of_bounds_layer(layer):
    c = config()
    module = primitive_module(GatedDeltaNetStateSlice, (c.ref_type, fm.tensor_type("int32", ())))
    with pytest.raises(EvaluationError, match="outside"):
        TorchEvaluator(DictWeightResolver({})).run(
            module, {"state": create_gdn_state(c), "layer_id": torch.tensor(layer, dtype=torch.int32)})


def test_state_slice_python_resume_preserves_reference_type():
    module = primitive_module(GatedDeltaNetStateSlice, (config().ref_type, fm.tensor_type("int32", ())))
    namespace = {}
    exec(fm.module_source(module), namespace)
    assert namespace["MODULE"].semantic_hash == module.semantic_hash


@pytest.mark.parametrize("state_type", [fm.RefType("opaque", ()), fm.RefType("qwen3_5_gated_delta_net_state", ())])
def test_state_slice_rejects_unconfigured_reference(state_type):
    with pytest.raises(IRSchemaError):
        primitive_module(GatedDeltaNetStateSlice, (state_type, fm.tensor_type("int32", ())))

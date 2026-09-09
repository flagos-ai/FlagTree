# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.ir.ops.nn._gdn_state import GatedDeltaNetStateConfig
from triton.flagmega.ir.ops.nn._paged_attention_state import PagedAttentionStateConfig
from triton.flagmega.runtime import module as runtime_module


@pytest.mark.parametrize("input_type,extra_tensor", (
    (fm.tensor_type("bfloat16", (2, 16)), False),
    (fm.tensor_type("int64", (1, )), False),
    (fm.tensor_type("int32", (1, 1)), False),
    (fm.tensor_type("int32", ("tokens", )), False),
    (fm.tensor_type("int32", (2, )), False),
    (fm.tensor_type("int32", (1, )), True),
))
@pytest.mark.parametrize("state_type", (
    GatedDeltaNetStateConfig(1, 1, 2, 4, 4, 4, 16).ref_type,
    PagedAttentionStateConfig(1, 2, 8).ref_type,
))
def test_state_fields_alone_do_not_select_single_token_convenience_adapter(monkeypatch, input_type, extra_tensor,
                                                                           state_type):
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    value = builder.var("input", input_type)
    state = builder.var("state", state_type)
    parameters = [value, state]
    if extra_tensor:
        parameters.append(builder.var("weights", fm.tensor_type("bfloat16", (16, 16))))
    builder.function("main", parameters, (value, state))
    module = fm.verify_module(builder.build(entry="main"))
    marker = object()
    monkeypatch.setattr(runtime_module, "GeneratedTirCallGraphModule", lambda *args: marker)
    for name in ("GeneratedTirGatedDeltaNetModule", "GeneratedTirPagedAttentionModelModule",
                 "GeneratedTirPagedAttentionLayerModule"):
        monkeypatch.setattr(runtime_module, name,
                            lambda *args: pytest.fail("The single-token adapter does not implement this entry ABI."))
    assert runtime_module.create_tir_runtime(None, None, module, None) is marker


@pytest.mark.parametrize("distributed", (False, True))
@pytest.mark.parametrize("state_type,adapter", (
    (GatedDeltaNetStateConfig(1, 1, 2, 4, 4, 4, 16).ref_type, "GeneratedTirGatedDeltaNetModule"),
    (PagedAttentionStateConfig(1, 2, 8).ref_type, "GeneratedTirPagedAttentionLayerModule"),
))
def test_single_token_logical_entry_keeps_convenience_adapter(monkeypatch, distributed, state_type, adapter):
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    input_type = fm.tensor_type("int32", (1, ))
    if distributed:
        input_type = fm.DistributedType(input_type, (fm.SBP.broadcast(), ), fm.Placement((2, ), "x", "b"))
    value = builder.var("input_ids", input_type)
    state = builder.var("state", state_type)
    builder.function("main", (value, state), (value, state))
    module = fm.verify_module(builder.build(entry="main"))
    marker = object()
    monkeypatch.setattr(runtime_module, adapter, lambda *args: marker)
    assert runtime_module.create_tir_runtime(None, None, module, None) is marker

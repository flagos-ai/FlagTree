# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Paired transfer capacity comes from its typed pipe, not a fixed value two."""

from copy import deepcopy

import pytest

from triton.flagmega.codegen.triton.kernel_call_renderers import _dense_matmul_glu_call
from triton.flagmega.codegen.triton.runtime_binding import describe_function_runtime_binding
from triton.flagmega.errors import CodegenError


def _raw(module, capacity):
    binding = describe_function_runtime_binding(module)
    call = deepcopy(next(value for value in binding["call_abi"]["kernel_calls"]
                         if value["semantic_op"] == "nn.packed_dense_matmul_glu"))
    call["transfer_pipeline"]["capacity"] = capacity
    for workspace in call["shared_workspaces"][:2]:
        workspace["shape"] = (capacity, *workspace["shape"][1:])
    return call


@pytest.mark.parametrize("capacity", (1, 2, 3))
def test_paired_glu_uses_exact_selected_pipe_capacity(packed_glu_paired_table_inline_descriptor_pipeline_module, capacity):
    raw = _raw(packed_glu_paired_table_inline_descriptor_pipeline_module, capacity)
    call = _dense_matmul_glu_call(raw)
    assert call["pipeline_contract"]["capacity"] == capacity
    assert all(workspace["shape"][0] == capacity for workspace in call["shared_workspaces"])


@pytest.mark.parametrize("capacity", (0, -1, True, 1.5))
def test_paired_glu_rejects_invalid_capacity(packed_glu_paired_table_inline_descriptor_pipeline_module, capacity):
    with pytest.raises(CodegenError, match="capacity"):
        _dense_matmul_glu_call(_raw(packed_glu_paired_table_inline_descriptor_pipeline_module, capacity))


def test_paired_glu_capacity_does_not_relax_workspace_shape(packed_glu_paired_table_inline_descriptor_pipeline_module):
    raw = _raw(packed_glu_paired_table_inline_descriptor_pipeline_module, 3)
    raw["shared_workspaces"][1]["shape"] = (2, *raw["shared_workspaces"][1]["shape"][1:])
    with pytest.raises(CodegenError, match="workspaces"):
        _dense_matmul_glu_call(raw)

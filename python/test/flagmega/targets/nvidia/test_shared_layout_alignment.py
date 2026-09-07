# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""NVMMA Shared storage must carry its swizzle-period alignment into SAT."""

import pytest
from dataclasses import replace

from triton.flagmega.ir import tensor_type
from triton.flagmega.ir.tir import TIRSharedWorkspaceDescriptor
from triton.flagmega.targets.nvidia.implementations import _implementation, sm90_triton_implementation_model
from triton.flagmega.targets import NvidiaSm90Target
from triton.flagmega.errors import IRVerificationError


def test_all_matrix_shared_catalog_workspaces_use_nncase_alignment():
    invalid = [(implementation.id, value.name, value.alignment_bytes)
               for implementation in sm90_triton_implementation_model().implementations
               for value in implementation.shared_workspaces
               if value.matrix_compatible and value.alignment_bytes % 1024]
    assert not invalid, invalid


@pytest.mark.parametrize("alignment, matrix, expected", [(128, True, 1024), (4096, True, 4096), (16, False, 16)])
def test_target_workspace_policy_preserves_stronger_and_nonmatrix_contracts(alignment, matrix, expected):
    workspace = TIRSharedWorkspaceDescriptor("tile", tensor_type("bfloat16", (32, 64)), alignment,
                                             matrix_compatible=matrix)
    implementation = _implementation("unit.tile", "unit", "tile", shared_workspaces=(workspace,))
    assert implementation.shared_workspaces[0].alignment_bytes == expected


def test_qkv_schedule_does_not_compensate_misaligned_shared_rows():
    for implementation in sm90_triton_implementation_model().implementations:
        assert "mma_logical_row_xor" not in implementation.parameters


def test_injected_catalog_cannot_weaken_nvmma_alignment():
    model = sm90_triton_implementation_model()
    implementation = next(value for value in model.implementations
                          if any(workspace.matrix_compatible for workspace in value.shared_workspaces))
    edited = replace(implementation, shared_workspaces=tuple(
        replace(value, alignment_bytes=128) if value.matrix_compatible else value
        for value in implementation.shared_workspaces))
    model = replace(model, implementations=tuple(edited if value.id == edited.id else value
                                                 for value in model.implementations))
    with pytest.raises(IRVerificationError, match="Shared.*1024"):
        NvidiaSm90Target(triton_implementation_model=model)

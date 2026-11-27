# Copyright 2026 FlagOS Contributors; SPDX-License-Identifier: MIT
"""Reject stale credit/staging contracts rather than silently accepting them."""
from copy import deepcopy

import pytest

from triton.flagmega.codegen.triton.iq4_q8_persistent_renderer import implementations
from triton.flagmega.codegen.triton.iq4_q8_pipeline_renderer import validate_transfer_contract
from triton.flagmega.errors import CodegenError


def exported_contract(implementation):
    pipeline = implementation.transfer_pipeline
    return dict(transfer_pipeline=dict(capacity=pipeline.capacity,
        producer_read_argument_indices=pipeline.producer_read_argument_indices,
        consumer_shared_workspace_indices=pipeline.consumer_shared_workspace_indices,
        auxiliary_consumer=None,
        channels=tuple(dict(name=c.name, source_argument_indices=c.source_argument_indices,
            shared_workspace_indices=c.shared_workspace_indices,
            source_alignment_bytes=c.source_alignment_bytes) for c in pipeline.channels)),
        shared_workspaces=tuple(dict(shape=tuple(d.fixed_value for d in workspace.type.shape),
            dtype=workspace.type.dtype.value, matrix_compatible=workspace.matrix_compatible)
            for workspace in implementation.shared_workspaces))


@pytest.mark.parametrize('rows', [1, 2, 5])
def test_persistent_rejects_wrong_shared_dtype_extent_and_channel_source(rows):
    implementation = implementations(rows)[0]
    raw = exported_contract(implementation)
    validate_transfer_contract(raw, implementation)
    for key, value in [('dtype', 'float32'), ('shape', (2, 2)),
                       ('matrix_compatible', not raw['shared_workspaces'][0]['matrix_compatible'])]:
        invalid = deepcopy(raw)
        invalid['shared_workspaces'][0][key] = value
        with pytest.raises(CodegenError, match='staging ABI mismatch'):
            validate_transfer_contract(invalid, implementation)
    invalid = deepcopy(raw)
    invalid['transfer_pipeline']['channels'][0]['source_argument_indices'] = (3,)
    with pytest.raises(CodegenError, match='transfer contract mismatch'):
        validate_transfer_contract(invalid, implementation)
    # A byte-staging ABI cannot validate as a prefetch-credit queue or vice versa.
    other = implementations(5 if rows == 1 else 1)[0]
    with pytest.raises(CodegenError, match='staging ABI mismatch'):
        validate_transfer_contract(raw, other)

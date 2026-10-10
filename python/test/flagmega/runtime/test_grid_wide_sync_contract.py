# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Backends without CUDA cooperative-launch admission (e.g. BI-V150) can only
run a grid-wide barrier kernel safely when the launch grid does not exceed
the physical SM count -- without admission there is no guarantee every CTA
is resident at once, and an under-scheduled CTA can deadlock the barrier."""

import pytest

import triton.flagmega.runtime.prepared as prepared_runtime
from triton.flagmega.errors import RuntimeContractError
from triton.flagmega.runtime import ResourceContract
from triton.flagmega.runtime.module import _requires_grid_wide_sync


class _IRModuleStub:
    def __init__(self, metadata):
        self.metadata = metadata


def test_requires_grid_wide_sync_reads_launch_contract_flag():
    assert _requires_grid_wide_sync(_IRModuleStub({})) is False
    assert _requires_grid_wide_sync(_IRModuleStub({"launch_contract": {}})) is False
    assert _requires_grid_wide_sync(
        _IRModuleStub({"launch_contract": {"cooperative_grid": False}})
    ) is False
    assert _requires_grid_wide_sync(
        _IRModuleStub({"launch_contract": {"cooperative_grid": True}})
    ) is True


class _Metadata:
    num_warps = 1
    num_ctas = 1
    shared = 0
    ptxas_stack_frame_bytes = 0
    ptxas_spill_store_bytes = 0
    ptxas_spill_load_bytes = 0


class _Compiled:
    metadata = _Metadata()
    n_regs = 8
    n_spills = 0

    @property
    def run(self):
        return lambda *_args: None


def _contract(**overrides):
    fields = dict(compute_num_warps=1, resident_blocks_per_sm=1)
    fields.update(overrides)
    return ResourceContract(**fields)


def test_no_admission_without_grid_wide_sync_is_unaffected():
    contract = _contract(cooperative_launch_admission=False)
    prepared_runtime._validate_resources(_Compiled(), contract, (16,))


def test_grid_wide_sync_with_admission_is_unaffected():
    contract = _contract(requires_grid_wide_sync=True, cooperative_launch_admission=True)
    prepared_runtime._validate_resources(_Compiled(), contract, (1024,))


def test_grid_wide_sync_without_admission_requires_sm_count():
    contract = _contract(requires_grid_wide_sync=True, cooperative_launch_admission=False)
    with pytest.raises(RuntimeContractError, match="no available_sm_count was supplied"):
        prepared_runtime._validate_resources(_Compiled(), contract, (16,))


def test_grid_wide_sync_without_admission_rejects_oversubscribed_grid():
    contract = _contract(
        requires_grid_wide_sync=True, cooperative_launch_admission=False, available_sm_count=16,
    )
    with pytest.raises(RuntimeContractError, match="17 CTA.*only 16 SM"):
        prepared_runtime._validate_resources(_Compiled(), contract, (17,))


def test_grid_wide_sync_without_admission_accepts_grid_at_sm_count():
    contract = _contract(
        requires_grid_wide_sync=True, cooperative_launch_admission=False, available_sm_count=16,
    )
    prepared_runtime._validate_resources(_Compiled(), contract, (4, 4))


def test_grid_wide_sync_without_admission_counts_cluster_size():
    clustered_metadata = _Metadata()
    clustered_metadata.num_ctas = 2

    class _ClusteredCompiled(_Compiled):
        metadata = clustered_metadata

    contract = _contract(
        requires_grid_wide_sync=True, cooperative_launch_admission=False, available_sm_count=16,
    )
    with pytest.raises(RuntimeContractError, match="18 CTA.*only 16 SM"):
        prepared_runtime._validate_resources(_ClusteredCompiled(), contract, (9,))

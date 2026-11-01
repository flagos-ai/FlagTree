# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The device-call stack and assembler register spills are different resources."""

from types import SimpleNamespace

import pytest

from triton.flagmega.errors import RuntimeContractError
from triton.flagmega.runtime.prepared import PreparedKernel, ResourceContract, _validate_resources


def _compiled(*, local_words=10, stack=40, stores=0, loads=0):
    return SimpleNamespace(
        name="entry", n_regs=32, n_spills=local_words, run=lambda: None,
        metadata=SimpleNamespace(
            num_warps=4, shared=1024,
            ptxas_stack_frame_bytes=stack,
            ptxas_spill_store_bytes=stores,
            ptxas_spill_load_bytes=loads,
        ),
    )


def test_call_stack_is_not_a_register_spill():
    compiled = _compiled()
    contract = ResourceContract(4, 1)
    _validate_resources(compiled, contract)
    prepared = PreparedKernel(compiled, (), (), grid=(1,), contract=contract)
    report = prepared.resource_report
    assert report["spill_bytes"] == 0
    assert report["spill_store_bytes"] == report["spill_load_bytes"] == 0
    assert report["stack_frame_bytes"] == report["local_memory_bytes"] == 40


@pytest.mark.parametrize("stores,loads", [(4, 0), (0, 12), (4, 12)])
def test_spill_contract_uses_assembler_counts_even_without_driver_local_memory(stores, loads):
    with pytest.raises(RuntimeContractError, match="spill-store bytes.*spill-load bytes"):
        _validate_resources(_compiled(local_words=0, stores=stores, loads=loads), ResourceContract(4, 1))


def test_missing_assembler_evidence_is_not_silently_accepted():
    compiled = _compiled(local_words=0)
    del compiled.metadata.ptxas_spill_load_bytes
    with pytest.raises(RuntimeContractError, match="resource metadata"):
        _validate_resources(compiled, ResourceContract(4, 1))


def _compiled_without_ptxas_fields(*, n_spills=0):
    # A backend (e.g. COREX) that only reports the aggregate n_spills count,
    # with no per-function ptxas_stack_frame_bytes/spill_store_bytes/
    # spill_load_bytes breakdown at all.
    return SimpleNamespace(
        name="entry", n_regs=32, n_spills=n_spills, run=lambda: None,
        metadata=SimpleNamespace(num_warps=4, shared=1024),
    )


def test_non_ptxas_backend_attributes_n_spills_as_spills_not_stack():
    compiled = _compiled_without_ptxas_fields(n_spills=3)
    contract = ResourceContract(4, 1, reports_ptxas_resource_fields=False)
    with pytest.raises(RuntimeContractError, match="spill-store bytes.*spill-load bytes"):
        _validate_resources(compiled, contract)


def test_non_ptxas_backend_with_no_spills_is_accepted():
    compiled = _compiled_without_ptxas_fields(n_spills=0)
    contract = ResourceContract(4, 1, reports_ptxas_resource_fields=False)
    _validate_resources(compiled, contract)
    prepared = PreparedKernel(compiled, (), (), grid=(1,), contract=contract)
    report = prepared.resource_report
    assert report["spill_bytes"] == report["local_memory_bytes"] == 0
    assert report["stack_frame_bytes"] == 0


def test_non_ptxas_backend_missing_n_spills_is_not_silently_accepted():
    compiled = _compiled_without_ptxas_fields()
    del compiled.n_spills
    contract = ResourceContract(4, 1, reports_ptxas_resource_fields=False)
    with pytest.raises(RuntimeContractError, match="resource metadata"):
        _validate_resources(compiled, contract)


def test_spill_tolerance_accepts_spills_at_or_below_the_limit():
    compiled = _compiled_without_ptxas_fields(n_spills=24)
    contract = ResourceContract(
        4, 1, reports_ptxas_resource_fields=False, spill_tolerance_bytes=96,
    )
    _validate_resources(compiled, contract)


def test_spill_tolerance_still_rejects_spills_above_the_limit():
    compiled = _compiled_without_ptxas_fields(n_spills=25)
    contract = ResourceContract(
        4, 1, reports_ptxas_resource_fields=False, spill_tolerance_bytes=96,
    )
    with pytest.raises(RuntimeContractError, match="tolerated byte"):
        _validate_resources(compiled, contract)


def test_spill_tolerance_defaults_to_zero_and_does_not_relax_nvidia():
    compiled = _compiled(local_words=0, stores=1, loads=0)
    contract = ResourceContract(4, 1)
    assert contract.spill_tolerance_bytes == 0
    with pytest.raises(RuntimeContractError, match="tolerated byte"):
        _validate_resources(compiled, contract)


@pytest.mark.parametrize("cached_error_launcher", [False, True])
def test_oversized_shared_is_rejected_before_loading_or_reading_driver_registers(cached_error_launcher):
    class Unloaded:
        metadata = SimpleNamespace(num_warps=4, shared=288908)
        run_reads = 0

        @property
        def run(self):
            self.run_reads += 1
            if cached_error_launcher:
                # A previous load failure caches a raising launcher without
                # initializing n_regs/n_spills. Reading it is not admission.
                return lambda: None
            raise AssertionError("Oversized shared storage must be rejected before loading")

    compiled = Unloaded()
    for _ in range(2):
        with pytest.raises(RuntimeContractError, match="288908 shared-memory bytes; limit is 227328"):
            _validate_resources(compiled, ResourceContract(4, 1))
    assert compiled.run_reads == 0


def test_resident_shared_budget_is_checked_before_loading():
    class Unloaded:
        metadata = SimpleNamespace(num_warps=4, shared=120000)

        @property
        def run(self):
            raise AssertionError("Resident shared budget must be checked before loading")

    with pytest.raises(RuntimeContractError, match="240000 shared-memory bytes for 2 resident block"):
        _validate_resources(Unloaded(), ResourceContract(4, 2))

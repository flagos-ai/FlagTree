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

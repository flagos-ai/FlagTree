# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

import triton.flagmega.runtime.prepared as prepared_runtime
from triton.flagmega.errors import RuntimeContractError
from triton.flagmega.runtime import PreparedKernel, ResourceContract


class _Metadata:
    num_warps = 1
    shared = 0
    ptxas_stack_frame_bytes = 0
    ptxas_spill_store_bytes = 0
    ptxas_spill_load_bytes = 0


class _Compiled:
    metadata = _Metadata()
    n_regs = 8
    n_spills = 0
    function = object()
    packed_metadata = object()
    name = "stream-contract"

    def launch_metadata(self, grid, stream, *arguments):
        return None

    @property
    def run(self):
        return lambda *_args: None


class _Scratch:
    nbytes = 128

    def __call__(self, size, alignment, stream):
        return object()


def test_prepared_global_scratch_binds_to_first_stream(monkeypatch):
    monkeypatch.setattr(
        prepared_runtime,
        "_prepare_global_scratch",
        lambda _compiled, _arguments, _grid: _Scratch(),
    )
    prepared = PreparedKernel(
        _Compiled(),
        (),
        (),
        grid=(1,),
        contract=ResourceContract(1, 1),
    )

    prepared.launch(stream=101)
    prepared.launch(stream=101)
    with pytest.raises(RuntimeContractError, match="bound to its first launch stream"):
        prepared.launch(stream=202)

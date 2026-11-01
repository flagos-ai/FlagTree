# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Prepared launches must retain the pointer facts used by compilation."""

from types import SimpleNamespace

import pytest

from triton.flagmega.errors import RuntimeContractError
from triton.flagmega.runtime import PreparedKernel, ResourceContract
from python.test.flagmega.runtime.test_prepared_stream_contract import _Compiled


class Pointer:
    def __init__(self, address):
        self.address = address

    def data_ptr(self):
        return self.address


def prepared(attributes):
    compiled = _Compiled()
    compiled.src = SimpleNamespace(attrs=attributes)
    return PreparedKernel(compiled, (Pointer(256), 16), (0, 1), grid=(1,), contract=ResourceContract(1, 1))


def test_rebinding_cannot_violate_a_compiled_pointer_alignment():
    kernel = prepared({(0,): [["tt.divisibility", 16]]})
    with pytest.raises(RuntimeContractError, match="argument 0.*16-byte"):
        kernel.launch(Pointer(260), 16, stream=101)


def test_aligned_rebinding_retains_pointer_specialization():
    prepared({(0,): [["tt.divisibility", 16]]}).launch(Pointer(512), 16, stream=101)


def test_unspecialized_pointer_has_no_invented_alignment_requirement():
    prepared({}).launch(Pointer(260), 16, stream=101)


def test_integer_divisibility_is_not_a_pointer_contract():
    prepared({(1,): [["tt.divisibility", 16]]}).launch(Pointer(260), 17, stream=101)

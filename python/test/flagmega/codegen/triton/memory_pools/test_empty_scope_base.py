# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
from triton.flagmega.codegen.triton.pool_abi import emit_pool_scope_base
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir.bufferization.memory_pool import FunctionMemoryPool


def test_used_empty_pool_is_distinct_from_absent_pool():
    assert not FunctionMemoryPool("workspace", 0, 16).requires_binding
    assert FunctionMemoryPool("workspace", 0, 16, ("empty",)).requires_binding
    assert FunctionMemoryPool("workspace", 256, 16).requires_binding


def test_empty_block_scopes_share_the_undereferenced_base():
    pool = {"scope_count": 4, "scope_nbytes": 0, "scope": "block", "scope_index": "program_id_x"}
    assert emit_pool_scope_base(pool, "arena") == "arena"
    assert emit_pool_scope_base({**pool, "scope_nbytes": 256}, "arena") == "_flagmega_block_scope_base(arena, 256)"


@pytest.mark.parametrize("change", [
    {"scope_nbytes": -1}, {"scope_index": ""}, {"scope_index": "unknown"}, {"scope": "die"},
])
def test_empty_pool_does_not_weaken_scope_validation(change):
    pool = {"scope_count": 4, "scope_nbytes": 0, "scope": "block", "scope_index": "program_id_x"}
    with pytest.raises(CodegenError):
        emit_pool_scope_base({**pool, **change}, "arena")


def test_missing_scope_size_is_not_an_explicit_empty_pool():
    with pytest.raises(CodegenError):
        emit_pool_scope_base({"scope_count": 4, "scope": "block", "scope_index": "program_id_x"}, "arena")

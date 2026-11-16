# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Reject malformed or missing census ABI before a device launch."""
import re
import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

@triton.jit
def _invalid_census(data, CASE: tl.constexpr):
    if CASE == 0:
        tle.gpu.census_mark(0, 0)
    elif CASE == 1:
        tle.gpu.census_mark(0.0, 0)
    elif CASE == 2:
        tle.gpu.census_mark(0, 4)
    elif CASE == 3:
        tle.gpu.census_mapping(2, 0)
    elif CASE == 4:
        tle.gpu.census_configure(4, 128)
        tle.gpu.census_configure(5, 128)
    elif CASE == 5:
        tle.gpu.census_mark(tl.full((), 0, tl.uint32), 0)
    elif CASE == 6:
        tle.gpu.census_configure(0, 128)
    elif CASE == 7:
        tle.gpu.census_configure(4, 128)
        tle.gpu.census_mark(0, 0)
    tl.store(data, 1)

@pytest.mark.parametrize('case,message', [
    (0, 'configured WS profile-scratch ABI|enabled, configured WS profile-scratch ABI'),
    (1, 'scalar signed i32'), (2, 'point must be a constant integer'),
    (3, 'role must be a constant integer'), (4, 'conflicting census'),
    (5, 'scalar signed i32'), (6, 'occurrences must be a constant integer'),
    (7, 'configured WS profile-scratch ABI'),
])
def test_census_rejects_missing_or_malformed_abi(case, message, capfd):
    with pytest.raises(Exception) as failure:
        triton.compile(ASTSource(_invalid_census, {'data': '*i32'}, constexprs={'CASE': case}),
                       target=GPUTarget('cuda', 90, 32),
                       options={'num_warps': 8, 'ptx_version': 87})
    # MLIR diagnostics are written to stderr, while Python receives a pass failure.
    assert re.search(message, str(failure.value) + "\n" + capfd.readouterr().err)

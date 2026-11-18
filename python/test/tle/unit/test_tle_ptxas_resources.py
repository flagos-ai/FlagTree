# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest


def test_ptxas_resources_include_every_callee_and_separate_stack():
    from triton.experimental.tle.compiler_resources import parse_ptxas_resource_usage

    result = parse_ptxas_resource_usage("""
ptxas info    : Function properties for entry
    40 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 168 registers, 40 bytes cumulative stack size
ptxas info    : Function properties for worker
    0 bytes stack frame, 8 bytes spill stores, 28 bytes spill loads
ptxas info    : Function properties for helper
    8 bytes stack frame, 4 bytes spill stores, 0 bytes spill loads
""")
    assert result == {
        "ptxas_stack_frame_bytes": 40,
        "ptxas_spill_store_bytes": 12,
        "ptxas_spill_load_bytes": 28,
    }


@pytest.mark.parametrize("log", ["", "ptxas info : Used 32 registers"])
def test_missing_ptxas_function_properties_is_an_error(log):
    from triton.experimental.tle.compiler_resources import parse_ptxas_resource_usage

    with pytest.raises(RuntimeError, match="per-function resource"):
        parse_ptxas_resource_usage(log)

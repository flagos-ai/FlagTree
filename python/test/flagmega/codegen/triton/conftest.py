# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import ast

import pytest


@pytest.fixture
def assert_entry_owned_phase_schedule():
    def check(source, symbol):
        functions = {
            value.name: value for value in ast.parse(source).body
            if isinstance(value, ast.FunctionDef)
        }
        schedule = functions[symbol]
        assert ast.unparse(schedule.decorator_list[0]) == "triton.jit"
        assert len(schedule.body) == 3
        callees = [ast.unparse(value.value.func) for value in schedule.body]
        assert callees == [
            f"{symbol}__partials", "tle.distributed_barrier", f"{symbol}__finalize",
        ]
        for suffix in ("partials", "finalize"):
            phase = functions[f"{symbol}__{suffix}"]
            assert ast.unparse(phase.decorator_list[0]) == "triton.jit(noinline=True)"
            assert "tle.distributed_barrier" not in ast.unparse(phase)
    return check

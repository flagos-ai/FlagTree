# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""FlagMega-on-Radeon Target T1: the amd-gfx1100/gfx1201 machines register and
reach --stop-after propose-tir on a bf16 model using the (NV-reused) planner
services. All NVIDIA-identity / Sm90Capability verifier logic is gated to the
selected_tir stage and later, so it does not fire before propose-tir; the only
pre-TIR gate is the AMD capability filter over the implementation catalog, which
keeps the portable bf16 candidates and drops the fp8/tma/mma_v3 ones."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.targets import get_target, target_names

AMD_TARGETS = ("amd-gfx1100", "amd-gfx1201")

# propose-tir emits the selected_tir_variants stage.
_PROPOSE_TIR_STAGE = "selected_tir_variants"


def _bf16_matmul_module() -> fm.IRModule:
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    value_type = fm.tensor_type("bfloat16", (1, 128))
    weight_type = fm.tensor_type("bfloat16", (128, 128))
    value = builder.var("value", value_type, id="value")
    weight = builder.weight(
        "weight", weight_type, source="memory", key="weight", id="weight"
    )
    projection = builder.call(
        "math.matmul",
        (value, weight),
        value_type,
        id="projection",
        attrs={"transpose_a": False, "transpose_b": True},
    )
    builder.function("main", (value,), (projection,))
    return builder.build(entry="main")


def test_amd_targets_registered():
    assert set(AMD_TARGETS) <= set(target_names())


@pytest.mark.parametrize("target_name", AMD_TARGETS)
def test_amd_target_constructs(target_name):
    target = get_target(target_name)
    assert target.name == target_name
    assert target.codegen_platform == "amd"


@pytest.mark.parametrize("target_name", AMD_TARGETS)
def test_amd_bf16_matmul_reaches_propose_tir(target_name):
    module = Compiler(CompileOptions(target=target_name)).compile(
        _bf16_matmul_module(), stop_after="propose-tir"
    ).module

    assert module.stage == _PROPOSE_TIR_STAGE

    points = list(module.selection_points)
    assert points, "propose-tir produced no selection points"
    # The AMD capability filter must not have emptied any selection point: every
    # node keeps at least one capability-supported candidate (portable bf16).
    for point in points:
        assert point.candidates, (
            f"{target_name}: selection point {point.id!r} has no "
            f"capability-supported candidate"
        )

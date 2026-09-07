# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import inspect

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.selection import TritonTirSelectionPolicy
from triton.flagmega.targets import NvidiaSm90Target


def _block_fp8_module() -> fm.IRModule:
    builder = fm.IRBuilder(dialect="high_level", stage="distributed")
    activation = fm.tensor_type("bfloat16", (1, 128))
    weight = fm.tensor_type("float8_e4m3fn", (128, 128))
    scale = fm.tensor_type("float32", (1, 1))
    lhs = builder.var("lhs", activation, id="lhs")
    rhs = builder.weight("rhs", weight, source="weights", key="rhs", id="rhs")
    rhs_scale = builder.weight(
        "rhs_scale", scale, source="weights", key="rhs_scale", id="rhs_scale"
    )
    result = builder.call(
        "math.block_scaled_matmul",
        (lhs, rhs, rhs_scale),
        activation,
        id="result",
        attrs={"weight_block_n": 128, "weight_block_k": 128},
    )
    builder.function("main", (lhs,), (result,))
    return builder.build(entry="main")


def test_triton_selector_receives_target_workspace_policy_by_injection():
    calls = []

    def annotate(node, candidates, module, *, mesh_hierarchy):
        calls.append((node.id, mesh_hierarchy))
        return candidates

    target = NvidiaSm90Target(tir_selection_policy=TritonTirSelectionPolicy(annotate))
    target.propose_tir(_block_fp8_module())

    assert calls == [("result", (8, 16))]


def test_triton_selector_has_no_nvidia_import_dependency():
    source = inspect.getsource(inspect.getmodule(TritonTirSelectionPolicy))

    assert "targets.nvidia" not in source
    assert "sm90" not in source.lower()

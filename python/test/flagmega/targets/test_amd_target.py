# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""FlagMega-on-Radeon Target T1: the amd-gfx1100/gfx1201 machines register and
reach --stop-after propose-tir on a bf16 model using the (NV-reused) planner
services. All NVIDIA-identity / Sm90Capability verifier logic is gated to the
selected_tir stage and later, so it does not fire before propose-tir; the only
pre-TIR gate is the AMD capability filter over the implementation catalog, which
keeps the portable bf16 candidates and drops the fp8/tma/mma_v3 ones."""

from pathlib import Path
from types import SimpleNamespace
import runpy
import sys

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


def test_cooperative_warp_cap_applies_to_a_supplied_implementation_model():
    """A recipe/tutorial may pass its own implementation model to the target;
    the cooperative warp cap must still apply, otherwise compute_num_warps=16
    kernels blow the RDNA register budget (~127K regs vs 65536) and deadlock the
    grid barrier."""
    from triton.flagmega.targets.amd.machine import AmdGfx1100Machine
    from triton.flagmega.targets.portable_triton_implementations import (
        portable_triton_implementation_model,
    )

    cap = AmdGfx1100Machine._COOPERATIVE_MAX_WARPS
    supplied = portable_triton_implementation_model()
    # Sanity: the portable model carries kernels above the cap, so the cap is
    # actually exercised below.
    assert any(
        (dict(impl.parameters).get("compute_num_warps") or 0) > cap
        for impl in supplied.implementations
    )

    capped = AmdGfx1100Machine(implementation_model=supplied).triton_implementation_model()
    for impl in capped.implementations:
        warps = dict(impl.parameters).get("compute_num_warps")
        assert warps is None or warps <= cap


@pytest.mark.parametrize("target_cls_name", ("AmdGfx1100Target", "AmdGfx1201Target"))
def test_amd_target_resolves_a_capped_model_from_a_supplied_one(target_cls_name):
    """The recipe path supplies an implementation model THROUGH the target, not
    the machine. Verify the resolved model is still capped end-to-end -- this
    exercises the PyNttTarget -> machine resolution, so it fails if either the
    target routing or the machine cap regresses (the machine-only test above
    would not catch a PyNttTarget that uses the supplied model directly)."""
    from triton.flagmega.targets.amd.machine import AmdGfx1100Machine
    from triton.flagmega.targets.amd_gfx1100 import AmdGfx1100Target
    from triton.flagmega.targets.amd_gfx1201 import AmdGfx1201Target
    from triton.flagmega.targets.portable_triton_implementations import (
        portable_triton_implementation_model,
    )

    cap = AmdGfx1100Machine._COOPERATIVE_MAX_WARPS
    target_cls = {"AmdGfx1100Target": AmdGfx1100Target, "AmdGfx1201Target": AmdGfx1201Target}[target_cls_name]
    target = target_cls(triton_implementation_model=portable_triton_implementation_model())

    resolved = target.triton_implementation_model
    if callable(resolved):
        resolved = resolved()
    for impl in resolved.implementations:
        warps = dict(impl.parameters).get("compute_num_warps")
        assert warps is None or warps <= cap


def _model_with_injected_impl():
    """A portable model whose first implementation carries an explicitly chosen
    non-cap knob (tile_n=32) alongside an over-cap compute_num_warps, so a test
    can tell whether a directly-supplied model is preserved and/or capped."""
    from dataclasses import replace

    from triton.flagmega.targets.portable_triton_implementations import (
        portable_triton_implementation_model,
    )

    model = portable_triton_implementation_model()
    first = model.implementations[0]
    injected = replace(
        first,
        parameters={**dict(first.parameters), "compute_num_warps": 16, "tile_n": 32},
    )
    model = replace(model, implementations=(injected,) + model.implementations[1:])
    return model, injected.id


def _resolved_impl(target, candidate_id):
    resolved = target.triton_implementation_model
    if callable(resolved):
        resolved = resolved()
    (impl,) = [i for i in resolved.implementations if i.id == candidate_id]
    return dict(impl.parameters)


def test_pyntt_target_preserves_a_directly_supplied_model():
    """A model handed straight to the PyNttTarget constructor (not through the
    machine) must be used, not silently replaced by the machine default. On a
    machine with no extra invariants (NVIDIA) it passes through untouched --
    both the injected tile_n and the compute_num_warps survive verbatim."""
    from triton.flagmega.targets.nvidia.machine import NvidiaSm90Machine
    from triton.flagmega.targets.pyntt import PyNttTarget

    model, cid = _model_with_injected_impl()
    target = PyNttTarget(NvidiaSm90Machine(), triton_implementation_model=model)

    params = _resolved_impl(target, cid)
    assert params.get("tile_n") == 32          # explicit injection preserved
    assert params.get("compute_num_warps") == 16  # no cap on this machine


def test_pyntt_target_caps_but_preserves_a_directly_supplied_model():
    """On AMD the cooperative warp cap must apply to a directly-supplied model
    (correctness invariant), while every other explicitly injected knob -- here
    tile_n=32 -- is preserved rather than discarded."""
    from triton.flagmega.targets.amd.machine import AmdGfx1100Machine
    from triton.flagmega.targets.pyntt import PyNttTarget

    cap = AmdGfx1100Machine._COOPERATIVE_MAX_WARPS
    model, cid = _model_with_injected_impl()
    target = PyNttTarget(AmdGfx1100Machine(), triton_implementation_model=model)

    params = _resolved_impl(target, cid)
    assert params.get("tile_n") == 32              # explicit injection preserved
    assert params.get("compute_num_warps") == cap  # invariant applied


_AMD_RDNA_TUTORIAL = (Path(__file__).resolve().parents[4]
                      / "python/tutorials/flagmega/01-qwen3-1.7b-bf16/amd-rdna")
_GLU_TN64 = "tir.dense_matmul_glu.packed_k_major_gemv_tn64"


@pytest.fixture(scope="module")
def amd_rdna_tutorial():
    """Load the amd-rdna recipe and run.py once; they import the sibling
    nvidia-h800 ``local_optimizations`` package, which registers ops."""
    saved = list(sys.path)
    sys.path.insert(0, str(_AMD_RDNA_TUTORIAL))
    try:
        loaded = {name: runpy.run_path(str(_AMD_RDNA_TUTORIAL / name)) for name in ("recipe.py", "run.py")}
    finally:
        sys.path[:] = saved
    return loaded.__getitem__


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1201"])
def test_amd_rdna_recipe_retiles_only_the_gate_up_gemv(amd_rdna_tutorial, arch):
    from triton.flagmega.targets.portable_triton_implementations import (
        portable_triton_implementation_model,
    )
    recipe = amd_rdna_tutorial("recipe.py")
    model = recipe["create_target"](arch=arch).triton_implementation_model
    portable = portable_triton_implementation_model()
    assert model.implementation(_GLU_TN64).parameters["block_k"] == 32
    assert model.implementation(_GLU_TN64).parameters["tile_n"] == 64
    # The shared catalog and the other GLU tiles keep their own K step.
    assert portable.implementation(_GLU_TN64).parameters["block_k"] == 128
    tn16 = "tir.dense_matmul_glu.packed_k_major_gemv_tn16"
    assert model.implementation(tn16).parameters["block_k"] == portable.implementation(tn16).parameters["block_k"]
    assert model.implementation(recipe["_ATTENTION_PARTIAL"]) is not None
    with pytest.raises(ValueError, match="arch must be one of"):
        recipe["create_target"](arch="gfx942")


def test_amd_rdna_recipe_pins_only_points_that_offer_the_implementation(amd_rdna_tutorial):
    recipe = amd_rdna_tutorial("recipe.py")
    point = lambda point_id, *candidates: SimpleNamespace(
        id=point_id, candidates=tuple(SimpleNamespace(id=value) for value in candidates))
    module = SimpleNamespace(semantic_hash="module", selection_map={}, selection_points=(
        point("tir.gate_up", "tir.dense_matmul_glu.packed_k_major_gemv_tn16", _GLU_TN64),
        point("tir.down", "tir.dense_matmul.split_k_n_packed_k_major_gemv"),
    ))
    plan = recipe["_pin_implementation"](module, _GLU_TN64)
    assert [(record.point_id, record.candidate_id) for record in plan.records] == [("tir.gate_up", _GLU_TN64)]
    with pytest.raises(ValueError, match="not offered"):
        recipe["_pin_implementation"](module, recipe["_ATTENTION_PARTIAL"])


def test_amd_rdna_benchmark_context_lengths(amd_rdna_tutorial):
    parse = amd_rdna_tutorial("run.py")["_context_lengths"]
    assert parse("8192, 1,1024,1") == [1, 1024, 8192]
    for text in ("", "0,1024", "1k"):
        with pytest.raises(SystemExit):
            parse(text)

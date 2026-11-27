# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Expert stages over FP8 weights with per-block projection scales.

This is the GLM-5.3 serving configuration: ``float8_e4m3fn`` expert weights and
rank-3 ``*_proj_scale`` operands holding one scale per weight tile. Nothing in
the rest of the expert suite reaches it, because every other fixture uses
bf16/FP32 weights and rank-2 per-expert scales, and the generated masked load
of a narrow weight only fails to compile once the pointee is FP8.
"""

import re

import pytest
from triton.flagmega.codegen.triton import render_triton_package
from triton.flagmega.compiler import Compiler
from triton.flagmega.ir.ops.nn.sparse_experts_down import SparseExpertsDown
from triton.flagmega.ir.ops.nn.sparse_experts_gate_up import SparseExpertsGateUp

from python.test.flagmega.codegen.triton.kernels.sparse_experts.helpers import (
    execute_and_reference,
    stage_module,
)

DEFINITIONS = [SparseExpertsGateUp, SparseExpertsDown]

# The portable SIMT expert stage tiles K at 128. ``32`` puts the whole reduction
# in one weight scale group, which is the only regime a per-tile scalar group
# index can serve; ``16`` splits it in two, so the scale and the activation
# quantization it pairs with both have to vary inside a single K tile.
WEIGHT_BLOCKS = [32, 16]


def _fp8_module(definition, weight_block=32):
    return stage_module(definition, dtype="float32", weight_dtype="float8_e4m3fn", block_scaled=True,
                        weight_block=weight_block, hidden=32, intermediate=32, tokens=2)


@pytest.mark.parametrize("definition", DEFINITIONS)
def test_masked_weight_loads_use_a_floating_fill_value(tmp_path, definition):
    """``other=0`` is an int32 literal Triton cannot cast to an FP8 pointee.

    ``tl.load`` casts ``other`` to the pointee type, and that cast is rejected
    for ``float8_e4m3fn``: ``cannot cast int32 to fp8e4nv``. The fill has to be
    written as a floating literal for every weight load the stage emits.
    """

    compiled = Compiler().compile(_fp8_module(definition)).module
    render_triton_package(compiled, tmp_path / "generated")
    source = (tmp_path / "generated" / "generated_kernels.py").read_text()
    weight_loads = re.findall(r"^\s*_fm_(?:gate|up|weight) = tl\.load\(.*$", source, re.MULTILINE)
    assert weight_loads, "the stage must emit at least one masked weight load"
    for load in weight_loads:
        assert "other=0)" not in load, f"integer fill on an FP8 load: {load.strip()}"


@pytest.mark.parametrize("definition", DEFINITIONS)
def test_multi_group_k_tile_indexes_the_scale_per_lane(tmp_path, definition):
    """A K tile spanning several scale groups cannot use one group index.

    ``_fm_k_start // weight_block_k`` names the group the tile starts in. With
    ``weight_block_k=16`` under a 128-wide tile that covers eight groups, so the
    generated load must index by ``_fm_k`` and the activation abs-max must reduce
    per group instead of over the whole tile.
    """

    compiled = Compiler().compile(_fp8_module(definition, 16)).module
    render_triton_package(compiled, tmp_path / "generated")
    source = (tmp_path / "generated" / "generated_kernels.py").read_text()
    scale_loads = re.findall(r"^\s*_fm_\w*block_scale = tl\.load\(.*$", source, re.MULTILINE)
    assert scale_loads, "the block-scaled stage must emit at least one scale load"
    for load in scale_loads:
        assert "_fm_k_start // 16" not in load, f"per-tile group index on a multi-group tile: {load.strip()}"
        assert re.search(r"_fm_k\[None, :\]\)*\s*// 16", load), (
            f"scale is not indexed per K lane: {load.strip()}")
    assert "_fm_group_scale" in source, "the activation abs-max must reduce per weight scale group"


@pytest.mark.parametrize("weight_block", WEIGHT_BLOCKS)
@pytest.mark.parametrize("definition", DEFINITIONS)
def test_block_scaled_fp8_device_matches_evaluator(tmp_path, definition, weight_block):
    """A weight K block narrower than the codegen K tile still has to be honoured.

    ``weight_block_k`` is a property of the checkpoint and the K tile is a
    tuning parameter, so a tile may span several scale groups. Indexing the
    scale by ``_fm_k_start // weight_block_k`` then applies the first group's
    scale to the whole tile, and reducing the activation abs-max over the whole
    tile quantizes it against a scale no group agreed to.
    """

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    actual, expected, _ = execute_and_reference(_fp8_module(definition, weight_block), tmp_path, torch)
    # FP32 accumulation order differs between the tiled kernel and torch.matmul.
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("definition", DEFINITIONS)
@pytest.mark.parametrize("sign", (-1, 1))
def test_activation_quantization_preserves_round_to_even_ties(tmp_path, definition, sign):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    from triton.flagmega import ir as fm

    module = stage_module(definition, dtype="bfloat16", weight_dtype="float8_e4m3fn",
                          block_scaled=True, weight_block=128, hidden=128, intermediate=128, tokens=1)
    values = {}
    activation = definition.input_parameters[0].name
    for name in module.function_map[module.entry].parameters:
        tensor = fm.logical_type(module.node_map[name].type)
        shape = tuple(dim.fixed_value for dim in tensor.shape)
        if name == "router_expert_ids":
            value = torch.zeros(shape, dtype=torch.int32)
        elif name.endswith("_scale"):
            value = torch.ones(shape)
        elif name == activation:
            value = torch.zeros(shape, dtype=torch.bfloat16)
            value[..., 0], value[..., 1] = 1.203125, sign * .81640625
        else:
            value = torch.zeros(shape)
            value[..., 1] = 1.0
            value = value.to(getattr(torch, tensor.dtype.value))
        values[name] = value
    actual, expected, runtime = execute_and_reference(module, tmp_path, torch, overrides=values)
    try:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()

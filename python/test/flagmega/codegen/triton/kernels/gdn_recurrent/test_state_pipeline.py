# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from python.test.flagmega.codegen.triton.kernels.gdn_recurrent.helpers import execute_recurrent
from python.test.flagmega.gdn.recurrent_helpers import recurrent_case, recurrent_reference


@pytest.mark.parametrize("dimension,value_dimension,mesh,tile,policy", [
    (8, 6, (1, 2), 4, None), (12, 12, (2, 4), 4, None), (16, 16, (1, 1), 4, None),
    (128, 128, (2, 4), 32, None), (256, 128, (2, 4), 32, None),
    (512, 16, (2, 4), 4, None),
    (8, 8, (4, 4), 4, fm.SBP.split_contiguous((0, 1), granularity=4)),
    (8, 6, (1, 2), 4, fm.SBP.split_block_cyclic((0, 1), 4)),
])
def test_state_pipeline_matches_repeated_recurrence(tmp_path, dimension, value_dimension, mesh, tile, policy):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    key_tile = max(32, 1 << (dimension - 1).bit_length())
    identity = f"tir.gdn_recurrent.state_smem_pipeline_k{key_tile}_v{tile}"
    compiler = Compiler()
    model = compiler.target.triton_implementation_model
    assert model.implementation(identity) is not None
    compiler.target.triton_implementation_model = replace(model, preferences={**model.preferences,
                                                       "gdn_recurrent": (identity,)})
    config, attrs, types, values = recurrent_case(dimension=dimension, value_dimension=value_dimension,
                                                 qk_norm_mode="add", qk_norm_epsilon=1e-6, round_core=True)
    values["projection_input"][:, 0] = torch.tensor([1., -2., .5]).bfloat16()
    expected, expected_states = recurrent_reference(values, attrs)
    actual, states = execute_recurrent(tmp_path, torch, config, attrs, types, values, compiler=compiler,
                                       placement=fm.Placement(mesh, "yx", "bb"), split_policy=policy)
    source = (tmp_path / "artifact/generated_kernels.py").read_text()
    assert "gdn_recurrent/state_smem_pipeline" in source
    assert "_fm_state_slot" in source and "is_async=True" in source
    baseline = Compiler()
    baseline_model = baseline.target.triton_implementation_model
    baseline_id = "tir.gdn_recurrent.persistent" + (f"_k{key_tile}" if key_tile > 128 else "")
    reference_impl = baseline_model.implementation(baseline_id)
    reference_impl = replace(reference_impl, parameters={**reference_impl.parameters, "tile_state": (key_tile, tile)})
    baseline.target.triton_implementation_model = replace(baseline_model,
        implementations=tuple(reference_impl if impl.id == baseline_id else impl for impl in baseline_model.implementations))
    gpu_expected, gpu_states = execute_recurrent(tmp_path / "consumer_reference", torch, config, attrs, types, values,
                                                  compiler=baseline, placement=fm.Placement(mesh, "yx", "bb"), split_policy=policy)
    torch.testing.assert_close(actual, gpu_expected, rtol=0, atol=0)
    for actual_state, gpu_state in zip(states, gpu_states, strict=True):
        torch.testing.assert_close(actual_state, gpu_state, rtol=0, atol=0)
    # Independent FP32 reductions may straddle a BF16 rounding midpoint.
    # The transport itself is checked bit-for-bit against the same GPU math.
    torch.testing.assert_close(actual, expected, rtol=2 ** -7, atol=1e-6)
    for actual_state, expected_state in zip(states, expected_states, strict=True):
        torch.testing.assert_close(actual_state, expected_state, rtol=2e-5, atol=1e-7)

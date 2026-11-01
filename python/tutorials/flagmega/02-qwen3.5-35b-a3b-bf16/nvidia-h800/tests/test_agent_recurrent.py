"""The paired projection override must forward native gate/norm contracts."""

from pathlib import Path
import sys

import pytest


def test_agent_cached_recurrent_forwards_fast_gate_contract(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from triton.flagmega import ir as fm
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.codegen.triton import tir_package
    from python.test.flagmega.codegen.triton.kernels.gdn_recurrent.helpers import execute_recurrent
    from python.test.flagmega.gdn.recurrent_helpers import recurrent_case, recurrent_reference
    try:
        original = tir_package.TritonTemplateRegistry
        from agent_optimizations.decode import install, create_target
        install()
        config, attrs, types, values = recurrent_case(dimension=128, value_dimension=128,
            qk_norm_mode="add", qk_norm_epsilon=1e-6, round_core=True)
        attrs = {**attrs, "gate_math_mode": "fast", "norm_math_mode": "accurate"}
        compiler = Compiler()
        compiler.target = create_target()
        expected, expected_states = recurrent_reference(values, attrs)
        actual, states = execute_recurrent(tmp_path, torch, config, attrs, types, values,
            placement=fm.Placement((2, 4), "yx", "bb"), compiler=compiler)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual_state, expected_state in zip(states, expected_states):
            torch.testing.assert_close(actual_state, expected_state, rtol=2e-5, atol=1e-7)
    finally:
        if "original" in locals():
            tir_package.TritonTemplateRegistry = original
        sys.path.remove(str(root))

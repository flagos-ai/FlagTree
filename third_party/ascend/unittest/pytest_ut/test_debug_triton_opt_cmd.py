# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""
Test that ttir_to_linalg prints the triton-opt command line when opt.debug=True.
Does not require NPU hardware.
"""

from unittest.mock import patch, MagicMock
import pytest

MOD = "triton.backends.ascend.compiler"

FAKE_TOOL = "/fake/triton-opt"
FAKE_PIPELINE = "builtin.module(some-pass,another-pass)"


def _make_metadata():
    return {
        "enable_nd2nz_on_vector": False,
        "enable_select_analysis": False,
        "compile_on_910_95": False,
        "force_simt_template": False,
        "enable_sync_block_lock": False,
        "enable_mask_fallback_conversion": False,
        "optimize_dynamic_offset": False,
        "auto_blockify_size": 1,
        "add_auto_scheduling": False,
        "enable_dynamic_cv_pipeline": False,
        "hash": "deadbeef",
    }


def _make_opt(debug):
    opt = MagicMock()
    opt.debug = debug
    return opt


def _run_ttir_to_linalg(debug, capsys):
    """
    Patch all MLIR machinery so ttir_to_linalg can run without hardware,
    then return the captured stdout.
    """
    from triton.backends.ascend.compiler import ttir_to_linalg

    pm_mock = MagicMock()
    pm_mock.get_pipeline_str.return_value = FAKE_PIPELINE

    ir_mock = MagicMock()
    ir_mock.pass_manager.return_value = pm_mock

    dump_manager_mock = MagicMock()

    with patch(f"{MOD}._get_triton_opt_path", return_value=FAKE_TOOL), \
         patch(f"{MOD}._get_triton_adapter_opt_path", return_value="/fake/triton-adapter-opt"), \
         patch(f"{MOD}._is_auto_map_parallel_blocks_enabled", return_value=False), \
         patch(f"{MOD}.ir", ir_mock), \
         patch(f"{MOD}.ascend", MagicMock()), \
         patch(f"{MOD}.passes", MagicMock()), \
         patch(f"{MOD}.get_dump_manager", return_value=dump_manager_mock):
        ttir_to_linalg(MagicMock(), _make_metadata(), _make_opt(debug))

    return capsys.readouterr().out


def test_debug_print_contains_tool_path(capsys):
    out = _run_ttir_to_linalg(debug=True, capsys=capsys)
    assert FAKE_TOOL in out


def test_debug_print_contains_pass_pipeline(capsys):
    out = _run_ttir_to_linalg(debug=True, capsys=capsys)
    assert f"--pass-pipeline={FAKE_PIPELINE}" in out


def test_debug_print_contains_debug_info_flag(capsys):
    out = _run_ttir_to_linalg(debug=True, capsys=capsys)
    assert "--mlir-print-debuginfo" in out


def test_debug_print_contains_output_flag(capsys):
    out = _run_ttir_to_linalg(debug=True, capsys=capsys)
    assert " -o " in out


def test_debug_print_has_prefix(capsys):
    out = _run_ttir_to_linalg(debug=True, capsys=capsys)
    assert "[DEBUG] cmd list:" in out


def test_no_debug_print_when_debug_is_false(capsys):
    out = _run_ttir_to_linalg(debug=False, capsys=capsys)
    assert "[DEBUG] cmd list:" not in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

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

import os
from unittest.mock import patch, MagicMock
import pytest

MOD = "triton.backends.ascend.compiler"

FAKE_TOOL = "/fake/triton-opt"
FAKE_PIPELINE = "builtin.module(some-pass,another-pass)"


def _make_metadata():
    return {
        "compile_on_910_95": False,
        "auto_blockify_size": 1,
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

    The dump manager is mocked as a test double that mirrors the real
    ``FileCacheManager(dump=True)``: its ``cache_dir`` and ``_make_path``
    are anchored at ``$TRITON_DUMP_DIR`` (or the default ``~/.triton/dump``
    when the env var is unset) joined with the base32-encoded hash.
    """
    from triton.backends.ascend.compiler import ttir_to_linalg
    from triton.runtime.cache import _base32

    pm_mock = MagicMock()
    pm_mock.get_pipeline_str.return_value = FAKE_PIPELINE

    ir_mock = MagicMock()
    ir_mock.pass_manager.return_value = pm_mock

    # Mirror get_dump_manager(hash_key) -> FileCacheManager(_base32(key), dump=True):
    #   cache_dir = knobs.cache.dump_dir / _base32(key)
    # where knobs.cache.dump_dir is $TRITON_DUMP_DIR or ~/.triton/dump.
    dump_dir_root = os.environ.get("TRITON_DUMP_DIR") \
        or os.path.join(os.path.expanduser("~"), ".triton", "dump")
    expected_cache_dir = os.path.join(dump_dir_root, _base32("deadbeef"))

    dump_manager_mock = MagicMock()
    dump_manager_mock.cache_dir = expected_cache_dir
    dump_manager_mock._make_path.side_effect = \
        lambda filename: os.path.join(expected_cache_dir, filename)

    with patch(f"{MOD}._get_triton_opt_path", return_value=FAKE_TOOL), \
         patch(f"{MOD}._get_triton_adapter_opt_path", return_value="/fake/triton-adapter-opt"), \
         patch(f"{MOD}._is_auto_map_parallel_blocks_enabled", return_value=False), \
         patch(f"{MOD}.ir", ir_mock), \
         patch(f"{MOD}.ascend", MagicMock()), \
         patch(f"{MOD}.passes", MagicMock()), \
         patch(f"{MOD}.get_dump_manager", return_value=dump_manager_mock):
        ttir_to_linalg(MagicMock(), _make_metadata(), _make_opt(debug))

    return capsys.readouterr().out


@pytest.mark.skip(reason="The case is not supported on Ascend 950, skipping for now. Will be fixed in future.")
def test_debug_print_contains_tool_path(capsys):
    out = _run_ttir_to_linalg(debug=True, capsys=capsys)
    assert FAKE_TOOL in out


@pytest.mark.skip(reason="The case is not supported on Ascend 950, skipping for now. Will be fixed in future.")
def test_debug_print_contains_pass_pipeline(capsys):
    out = _run_ttir_to_linalg(debug=True, capsys=capsys)
    assert f"--pass-pipeline={FAKE_PIPELINE}" in out


@pytest.mark.skip(reason="The case is not supported on Ascend 950, skipping for now. Will be fixed in future.")
def test_debug_print_contains_debug_info_flag(capsys):
    out = _run_ttir_to_linalg(debug=True, capsys=capsys)
    assert "--mlir-print-debuginfo" in out


@pytest.mark.skip(reason="The case is not supported on Ascend 950, skipping for now. Will be fixed in future.")
def test_debug_print_contains_output_flag(capsys):
    out = _run_ttir_to_linalg(debug=True, capsys=capsys)
    assert " -o " in out


@pytest.mark.skip(reason="The case is not supported on Ascend 950, skipping for now. Will be fixed in future.")
def test_debug_print_has_prefix(capsys):
    out = _run_ttir_to_linalg(debug=True, capsys=capsys)
    assert "[DEBUG] cmd list:" in out


@pytest.mark.skip(reason="The case is not supported on Ascend 950, skipping for now. Will be fixed in future.")
def test_no_debug_print_when_debug_is_false(capsys):
    out = _run_ttir_to_linalg(debug=False, capsys=capsys)
    assert "[DEBUG] cmd list:" not in out


@pytest.mark.skip(reason="The case is not supported on Ascend 950, skipping for now. Will be fixed in future.")
def test_debug_print_uses_dump_dir_when_set(capsys, monkeypatch):
    """When TRITON_DUMP_DIR is set, debug output should show paths under that dir."""
    from triton.runtime.cache import _base32
    monkeypatch.setenv("TRITON_DUMP_DIR", "/my/dump/dir")
    out = _run_ttir_to_linalg(debug=True, capsys=capsys)
    # The actual hash is "deadbeef" which becomes base32 "32W353Y"
    hash_dir = _base32("deadbeef")
    # Should contain the dump dir path with base32-encoded hash, not /tmp
    assert "/my/dump/dir" in out
    assert "/tmp" not in out
    # Should contain the base32-encoded hash-based subdirectory
    assert hash_dir in out
    assert "kernel.ttir.mlir" in out


@pytest.mark.skip(reason="The case is not supported on Ascend 950, skipping for now. Will be fixed in future.")
def test_debug_print_uses_default_dump_dir_when_unset(capsys, monkeypatch):
    """When TRITON_DUMP_DIR is not set, debug output should show the default
    dump dir (~/.triton/dump), not /tmp."""
    monkeypatch.delenv("TRITON_DUMP_DIR", raising=False)
    out = _run_ttir_to_linalg(debug=True, capsys=capsys)
    default_dump_dir = os.path.join(os.path.expanduser("~"), ".triton", "dump")
    # Should contain the default dump dir, not /tmp
    assert default_dump_dir in out
    assert "/tmp" not in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

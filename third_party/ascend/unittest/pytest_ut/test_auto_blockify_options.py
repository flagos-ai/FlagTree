# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: MIT

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from triton.backends.ascend import compiler

pytestmark = pytest.mark.backend("cpu")


@pytest.mark.parametrize("environment_enabled", [False, True])
@pytest.mark.parametrize("kernel_enabled", [None, False, True])
def test_a2_a3_auto_blockify_precedence(environment_enabled, kernel_enabled):
    """The compiler must use the same option precedence as the launcher."""
    opt = compiler.NPUOptions(enable_auto_blockify=kernel_enabled)
    metadata = dict(opt.__dict__)
    metadata["bitcodes"] = []
    captured = []

    def compile_binary(command, **kwargs):
        captured.append(command)
        output = Path(command[command.index("-o") + 1] + ".o")
        output.write_bytes(b"test-binary")
        return SimpleNamespace(stdout=b"", stderr=b"")

    with patch.object(compiler, "_parse_linalg_metadata", side_effect=lambda ir, meta: (ir, meta)), \
         patch.object(compiler, "_check_bishengir_api_change", return_value=True), \
         patch.object(compiler, "_check_bishengir_is_regbased", return_value=False), \
         patch.object(compiler, "NPUUtils") as utils, \
         patch.object(compiler, "get_auto_bind_sub_block_option", return_value=False), \
         patch.object(compiler, "_is_auto_map_parallel_blocks_enabled", return_value=environment_enabled), \
         patch.object(compiler, "_is_ascend_sanitizer_enabled", return_value=False), \
         patch.object(compiler, "_is_debug_line_info_disabled", return_value=True), \
         patch.object(compiler, "_enable_print_ub_bits", return_value=False), \
         patch.object(compiler, "_enable_dump_memory_info", return_value=False), \
         patch.object(compiler, "_enable_msdebug", return_value=False), \
         patch.object(compiler, "_get_npucompiler_path", return_value=("/fake/bishengir-compile", {})), \
         patch("shutil.copy2"), \
         patch.object(compiler.subprocess, "run", side_effect=compile_binary):
        utils.return_value.get_arch.return_value = "Ascend910B2"
        result = compiler.linalg_to_bin_enable_npu_compile_A2_A3("module {}", metadata, opt)

    assert result == b"test-binary"
    expected = environment_enabled if kernel_enabled is None else kernel_enabled
    assert ("--enable-auto-blockify-loop" in captured[0]) is expected

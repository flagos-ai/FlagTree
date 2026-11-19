# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
Test `ir_override` for Ascend NPU compilation stages.

Strategy (inspired by upstream test_autotuner.py):
  1. Compile a "donor" kernel (does ``×10``) with ``TRITON_KERNEL_DUMP=1``
     to obtain dumped IR files (``.ttir`` / ``.ttadapter`` / ``.bcmlir``).
  2. Replace the donor function name with the target function name in each
     dumped file and save them as override inputs.
  3. Run a "target" kernel (identity: load → store) with ``ir_override``
     pointing to the renamed IR files.
  4. Verify the target output equals ``input × 10`` — proving that the
     override actually replaced the compilation stage.
"""

import os
import pathlib

import pytest
import torch
import torch_npu

import triton
import triton.backends.ascend.runtime  # noqa: F401 – patches triton.autotune
import triton.language as tl

# ── helpers ────────────────────────────────────────────────────────────


def _find_dumped_files(dump_root: str):
    """Walk *dump_root* and return ``{ext: filepath}`` for Ascend IR stages."""
    extensions = ("ttir", "ttadapter", "bcmlir")
    found = {}
    for dirpath, _, filenames in os.walk(dump_root):
        for fn in filenames:
            for ext in extensions:
                if fn.endswith(f".{ext}"):
                    found[ext] = os.path.join(dirpath, fn)
    return found


# ── test class ─────────────────────────────────────────────────────────


class TestIrOverride:
    """
    End-to-end tests for ``ir_override`` at three Ascend compilation stages:
    ``.ttir``, ``.ttadapter``, and ``.bcmlir``.
    """

    N = 1024
    DONOR_NAME = "_donor_mul10"
    TARGET_NAME = "_target_identity"

    @pytest.fixture(scope="class")
    def donor_ir(self, tmp_path_factory):
        """Compile donor kernel once, dump its IR (shared across all tests)."""
        tmp_path = tmp_path_factory.mktemp("ir_override")
        dump_dir = tmp_path / "dump"
        dump_dir.mkdir()

        # Route dumped IR into a known, isolated directory.
        # TRITON_ALWAYS_COMPILE forces recompilation so the stage loop
        # runs and dumps every IR stage even when a cached compiled
        # kernel already exists.
        _old_dump = os.environ.get("TRITON_KERNEL_DUMP")
        _old_dump_dir = os.environ.get("TRITON_DUMP_DIR")
        _old_always = os.environ.get("TRITON_ALWAYS_COMPILE")
        os.environ["TRITON_KERNEL_DUMP"] = "1"
        os.environ["TRITON_DUMP_DIR"] = str(dump_dir)
        os.environ["TRITON_ALWAYS_COMPILE"] = "1"
        try:
            # ── donor kernel: ×10 ──
            @triton.autotune(
                configs=[triton.Config({"BLOCK_SIZE": 32}, num_warps=4, num_stages=2, num_ctas=1)],
                key=["N"],
                do_bench=False,
                hints={"auto_gen_config": False},
            )
            @triton.jit
            def _donor_mul10(x_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
                offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask = offsets < N
                x = tl.load(x_ptr + offsets, mask=mask)
                output = x * 10.0
                tl.store(output_ptr + offsets, output, mask=mask)

            x = torch.randn(self.N, device="npu")
            out = torch.empty_like(x)
            grid = lambda meta: (triton.cdiv(self.N, meta["BLOCK_SIZE"]), )
            _donor_mul10[grid](x, out, self.N)
            torch.testing.assert_close(out, x * 10.0)
        finally:
            if _old_dump is not None:
                os.environ["TRITON_KERNEL_DUMP"] = _old_dump
            else:
                os.environ.pop("TRITON_KERNEL_DUMP", None)
            if _old_dump_dir is not None:
                os.environ["TRITON_DUMP_DIR"] = _old_dump_dir
            else:
                os.environ.pop("TRITON_DUMP_DIR", None)
            if _old_always is not None:
                os.environ["TRITON_ALWAYS_COMPILE"] = _old_always
            else:
                os.environ.pop("TRITON_ALWAYS_COMPILE", None)

        # Return (dumped_files, tmp_path) for test methods.
        return {
            "dumped": _find_dumped_files(str(dump_dir)),
            "tmp_path": tmp_path,
        }

    # ── per-stage test runner ──────────────────────────────────────────

    def _run_override_test(self, donor_ir, ext: str):
        """Run target kernel (identity) with ``ir_override`` from donor IR."""
        dumped = donor_ir["dumped"]
        tmp_path = donor_ir["tmp_path"]

        src_path = dumped.get(ext)
        if src_path is None:
            pytest.skip(f"No dumped .{ext} file found "
                        f"(bytecode mode may be disabled or stage not reached)")

        # Copy donor IR and rename the function symbol inside.
        content = pathlib.Path(src_path).read_text()
        content = content.replace(self.DONOR_NAME, self.TARGET_NAME)
        override_path = tmp_path / f"override.{ext}"
        override_path.write_text(content)

        # ── target kernel: identity ──
        @triton.autotune(
            configs=[
                triton.Config(
                    {"BLOCK_SIZE": 32, "ir_override": str(override_path)},
                    num_warps=4,
                    num_stages=2,
                    num_ctas=1,
                )
            ],
            key=["N"],
            do_bench=False,
            hints={"auto_gen_config": False},
        )
        @triton.jit
        def _target_identity(x_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
            offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < N
            x = tl.load(x_ptr + offsets, mask=mask)
            tl.store(output_ptr + offsets, x, mask=mask)

        x = torch.randn(self.N, device="npu")
        out = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(self.N, meta["BLOCK_SIZE"]), )
        _target_identity[grid](x, out, self.N)

        # If override works → ×10.  If silently ignored → identity.
        torch.testing.assert_close(out, x * 10.0, msg=f"ir_override with .{ext} did not take effect")

    # ── test cases ─────────────────────────────────────────────────────

    @pytest.mark.autotune
    def test_override_ttir(self, donor_ir):
        """Override the ``.ttir`` compilation stage."""
        self._run_override_test(donor_ir, "ttir")

    @pytest.mark.autotune
    def test_override_ttadapter(self, donor_ir):
        """Override the ``.ttadapter`` compilation stage."""
        self._run_override_test(donor_ir, "ttadapter")

    @pytest.mark.autotune
    def test_override_bcmlir(self, donor_ir):
        """Override the ``.bcmlir`` compilation stage (bytecode → MLIR text)."""
        self._run_override_test(donor_ir, "bcmlir")

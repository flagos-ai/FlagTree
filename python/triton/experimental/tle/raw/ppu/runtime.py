# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Final

from triton._C.libtriton import llvm  # pyright: ignore[reportMissingImports]
from triton._C.libtriton.tle.llvm import parse_llvm_ir  # pyright: ignore[reportMissingImports]
from triton.experimental.tle.raw.runtime import RawJITFunction
from triton.experimental.tle.raw.source_store import register_source

# TODO: Temporarily shell out to clang; replace with LLVM Python bindings later.
CLANG = os.getenv("CLANG", "clang")

PPU_INTRINSIC_PASSTHROUGH_SENTINEL: Final[str] = "__flagtree_ppu_intrinsic__"


def _disguise_ppu_intrinsics(ir_text: str) -> str:
    # PPU target intrinsics (``llvm.ppu.*``) emitted by the PPU SDK clang fork are
    # unknown to the LLVM that Triton links at build time. Rewrite the ``@llvm.ppu.``
    # prefix to an ordinary (non-``llvm.``) external-symbol prefix so MLIR imports and
    # re-exports them as plain calls instead of failing intrinsic-id lookup.
    return ir_text.replace("@llvm.ppu.", "@" + PPU_INTRINSIC_PASSTHROUGH_SENTINEL)


class PPUJITFunction(RawJITFunction):

    def __init__(self, fn: Any, file: Path, *args, **kwargs) -> None:
        super().__init__(fn, **kwargs)
        self.code: Final[str] = file.read_text()
        self.region_dialect: Final[str] = "ppu"
        self.lowered_region_dialect: Final[str] = "llvm"
        self.arg_dialect: Final[str] = "llvm"
        self.source_file: Final[str] = str(file)

    def register_pending_source(self, *, hint: str = "") -> str:
        if not self.extern_func_name:
            raise RuntimeError("deferred tle_raw PPU source requires extern_func_name= "
                               "(the device function symbol in the .cu file)")
        return register_source(
            region_dialect=self.region_dialect,
            extern_func_name=self.extern_func_name,
            source=self.code,
            hint=hint,
            extra={"source_file": self.source_file},
        )

    def create_region_deferred(self, builder, source_id: str, handles, alias_indices, hint: str = ""):
        return builder.create_tle_raw_region_deferred(
            source_id,
            self.region_dialect,
            self.arg_dialect,
            handles,
            alias_indices,
            hint,
        )

    def make_llvm(self, mlir_context) -> str:
        build = subprocess.run(
            [
                CLANG,
                "-x",
                "hggc",
                "--hggc-device-only",
                "-emit-llvm",
                "-O2",
                "-S",
                "-",
                "-o",
                "-",
            ],
            input=self.code.encode(),
            capture_output=True,
        )
        assert build.returncode == 0, (f"clang failed\nstderr:\n{build.stderr.decode()}")
        ir_text = _disguise_ppu_intrinsics(build.stdout.decode())
        llvm_context = llvm.context()
        module = parse_llvm_ir(ir_text, llvm_context, mlir_context)
        return f"{module}"

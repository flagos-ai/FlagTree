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
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Final, List, Optional
import hashlib
from triton._C.libtriton import llvm
from triton._C.libtriton.tle.llvm import parse_llvm_ir
from triton.experimental.tle.raw.runtime import RawJITFunction


def _find_tops_include_dir() -> str:
    env_dir = os.getenv("TOPS_INCLUDE_DIR")
    if env_dir and os.path.isdir(env_dir):
        return env_dir

    workspace_tops = Path(__file__).resolve().parents[7] / "tops"
    if workspace_tops.is_dir():
        return str(workspace_tops)

    caps_include = Path(os.getenv("CAPS_PATH", "/opt/tops")) / "include"
    if caps_include.is_dir():
        return str(caps_include)

    return "/opt/tops/include"


def _get_topscc_path() -> str:
    topscc = os.getenv("TOPSCC")
    if topscc and os.path.isfile(topscc):
        return topscc
    caps_path = os.getenv("CAPS_PATH", "/opt/tops")
    candidate = os.path.join(caps_path, "bin", "topscc")
    if os.path.isfile(candidate):
        return candidate
    return "topscc"


def _get_gcu_arch() -> str:
    return os.getenv("GCU_ARCH", "gcu400")


class TOPSJITFunction(RawJITFunction):
    """TLE-Raw dialect for TOPS C++ (.tops) files compiled via topscc.

    Usage:
        @dialect(name="tops", file=Path("kernel.tops"), arch="gcu400")
        def edsl(*args, **kwargs):
            ...
    """

    def __init__(self, fn: Any, file: Optional[Path] = None, arch: Optional[str] = None,
                 extra_flags: Optional[List[str]] = None, *args, **kwargs) -> None:
        super().__init__(fn, **kwargs)
        self.arch: Final[str] = arch or _get_gcu_arch()
        self.extra_flags: Final[List[str]] = extra_flags or []
        self.region_dialect: Final[str] = "tops"
        self.arg_dialect: Final[str] = "llvm"
        self.extern_func_name: Final[Optional[str]] = kwargs.get("extern_func_name", None)
        self.deferred: Final[bool] = kwargs.get("deferred", True)

        if file is not None:
            if hasattr(file, "read_text") and not isinstance(file, Path):
                file_name = getattr(file, "name", "<deferred>")
            else:
                file_name = str(file)
        else:
            file_name = "<inline>"
        self.file_name: Final[str] = file_name

        # Read the .tops source text so it can participate in cache key computation.
        # This ensures that editing the .tops file invalidates the Triton compile cache.
        if file is not None:
            code = file.read_text() if hasattr(file, "read_text") else Path(file).read_text()
        else:
            code = ""
        self.code: Final[str] = code

    @property
    def cache_key(self) -> str:
        """Hash of the .tops source content + arch + extern_func_name.

        Included in JITFunction.cache_key via DependenciesFinder so that
        changes to the .tops file invalidate the compile cache.
        """
        payload = f"{self.region_dialect}\0{self.extern_func_name or ''}\0{self.arch}\0{self.code}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _detect_topscc_style(topscc: str) -> str:
        """Detect topscc flag style: 'new' (--device-only/--gcu-arch) or 'legacy' (--cuda-device-only/--cuda-gpu-arch)."""
        try:
            result = subprocess.run(
                [topscc, "--help"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if "--device-only" in result.stdout and "--gcu-arch" in result.stdout:
                return "new"
        except (subprocess.TimeoutExpired, OSError):
            pass
        return "legacy"

    def _build_compile_cmd(self, topscc: str, tops_include: str, src_path: str) -> List[str]:
        style = self._detect_topscc_style(topscc)
        if style == "new":
            target_triple = f"dtu-enflame-tops--{self.arch}"
            return [
                topscc,
                "-x",
                "c++",
                "--device-only",
                "-emit-llvm",
                "-S",
                f"--target={target_triple}",
                f"--gcu-arch={self.arch}",
                "-std=c++17",
                "-O2",
                f"-I{tops_include}",
                "-fno-exceptions",
                "-fno-rtti",
                *self.extra_flags,
                src_path,
                "-o",
                "-",
            ]
        else:
            return [
                topscc,
                "-x",
                "tops",
                "--cuda-device-only",
                "-emit-llvm",
                "-S",
                f"--cuda-gpu-arch={self.arch}",
                "-std=c++17",
                f"-I{tops_include}",
                "-fno-exceptions",
                "-fno-rtti",
                *self.extra_flags,
                src_path,
                "-o",
                "-",
            ]

    def _compile_tops_to_llvm_ir(self) -> str:
        topscc = _get_topscc_path()
        tops_include = _find_tops_include_dir()

        with tempfile.NamedTemporaryFile(suffix=".tops", mode="w", delete=False) as src_file:
            src_file.write(self.code)
            src_path = src_file.name

        try:
            cmd = self._build_compile_cmd(topscc, tops_include, src_path)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                return result.stdout

            raise RuntimeError(f"topscc compilation failed:\n"
                               f"Command: {' '.join(cmd)}\n"
                               f"stderr:\n{result.stderr}")

        finally:
            if os.path.exists(src_path):
                os.unlink(src_path)

    def create_region_by_llvm(self, builder, llvm: str, handles, alias_indices, hint: str = "",
                              extern_func_name: str = ""):
        return super().create_region_by_llvm(builder, llvm, handles, alias_indices, hint, extern_func_name)

    def register_pending_source(self, *, hint: str = "") -> str:
        if not self.extern_func_name:
            raise RuntimeError("enflame only support deferred tops tle_raw requires extern_func_name "
                               "(the device function symbol in the .tops file)")
        payload = f"{self.region_dialect}\0{self.extern_func_name or ''}\0{self.file_name}".encode()
        return hashlib.sha256(payload).hexdigest()

    def create_region_deferred(self, builder, source_id: str, handles, alias_indices, hint: str = ""):
        return builder.create_tle_raw_region_deferred(
            source_id,
            self.region_dialect,
            self.arg_dialect,
            handles,
            alias_indices,
            hint,
            self.file_name,
            self.extern_func_name,
        )

    def make_llvm(self, mlir_context) -> str:
        llvm_ir_text = self._compile_tops_to_llvm_ir()
        llvm_ctx = llvm.context()
        module = parse_llvm_ir(llvm_ir_text, llvm_ctx, mlir_context)
        return f"{module}"

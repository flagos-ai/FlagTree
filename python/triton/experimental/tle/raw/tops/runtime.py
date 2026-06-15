from __future__ import annotations
import os
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Final, List, Optional

from triton._C.libtriton import llvm
from triton._C.libtriton.tle.llvm import parse_llvm_ir
from triton.backends.enflame.gcu_intrinsics import rewrite_intrinsics_to_placeholders


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


class TOPSJITFunction(object):
    """TLE-Raw dialect for TOPS C++ (.tops) files compiled via topscc.

    Usage:
        @dialect(name="tops", file=Path("kernel.tops"), arch="gcu400")
        def edsl(*args, **kwargs):
            ...
    """

    def __init__(self, fn: Any, file: Optional[Path] = None, arch: Optional[str] = None,
                 extra_flags: Optional[List[str]] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fn: Final[Any] = fn
        self.arch: Final[str] = arch or _get_gcu_arch()
        self.extra_flags: Final[List[str]] = extra_flags or []
        self.region_dialect: Final[str] = "tops"
        self.arg_dialect: Final[str] = "llvm"
        self.__triton_builtin__: Final[bool] = True

        if file is not None:
            self.code: Final[str] = Path(file).read_text()
            self.filename: Final[str] = str(file)
        else:
            self.code: Final[str] = ""
            self.filename: Final[str] = "<inline>"

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

    @staticmethod
    def _rewrite_gcu_intrinsics(llvm_ir: str) -> str:
        """Replace GCU-specific LLVM intrinsics with plain external function calls.

        mlir::translateLLVMIRToModule requires every LLVM intrinsic to have a
        registered LLVMImportDialectInterface. GCU intrinsics (@llvm.tcle.*)
        don't have one, so we rewrite them to ordinary external function
        declarations that MLIR can import as llvm.call ops.
        """
        if os.environ.get("MLIR_ENABLE_DUMP"):
            print("// ---- TLE-Raw: LLVM IR before intrinsic rewrite ----")
            print(llvm_ir)
            print("// ---- end ----")
        result = rewrite_intrinsics_to_placeholders(llvm_ir)
        if os.environ.get("MLIR_ENABLE_DUMP"):
            print("// ---- TLE-Raw: LLVM IR after intrinsic rewrite ----")
            print(result)
            print("// ---- end ----")
        return result

    def create_region_by_llvm(self, builder, llvm: str, handles, alias_indices, hint: str = ""):
        return builder.create_tle_raw_region_by_llvm_func(
            llvm,
            self.region_dialect,
            self.arg_dialect,
            handles,
            alias_indices,
            hint,
        )

    def make_llvm(self, mlir_context) -> str:
        llvm_ir_text = self._compile_tops_to_llvm_ir()
        llvm_ir_text = self._rewrite_gcu_intrinsics(llvm_ir_text)
        llvm_ctx = llvm.context()
        module = parse_llvm_ir(llvm_ir_text, llvm_ctx, mlir_context)
        return f"{module}"

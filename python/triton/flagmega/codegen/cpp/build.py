# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Compile a generated C++ package into a loadable shared library.

A Triton package needs no build step: the runtime JIT-compiles the generated
Python at prepare time. A C++ package does, and the compiler is the backend
vendor's, so the toolchain facts live per architecture here rather than in the
renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import os
import shutil
import subprocess

from triton.flagmega.errors import CodegenError


FTT_INCLUDE_DIR = Path(__file__).parent / "ftt" / "include"


def validate_cpp_max_registers(
    value: int | None, *, target: str, codegen_backend: str = "cpp",
    emit_executable: bool = True,
) -> None:
    """Reject unsupported resource requests before compilation or output writes."""
    if value is None:
        return
    if type(value) is not int or not 1 <= value <= 255:
        raise CodegenError("cpp_max_registers must be an integer between 1 and 255.")
    if (codegen_backend != "cpp" or not emit_executable
            or str(target) not in {"nvidia-sm89", "nvidia-sm90", "nvidia-sm120"}):
        raise CodegenError(
            "cpp_max_registers requires an executable NVIDIA C++ artifact."
        )


@dataclass(frozen=True)
class CppToolchain:
    """One backend's device-C++ compiler invocation."""

    name: str
    compiler: Path
    # Flags that select the device target. Separate from the generic flags so
    # a second architecture only supplies these.
    target_flags: tuple[str, ...]
    link_flags: tuple[str, ...]
    # Preprocessor definition that selects the FTT arch headers.
    arch_define: str
    host_flags: tuple[str, ...] = ("-fPIC",)

    def command(
        self, source: Path, output: Path, *, max_registers_per_thread: int | None = None,
    ) -> tuple[str, ...]:
        validate_cpp_max_registers(max_registers_per_thread, target=self.name)
        resource_flags = (() if max_registers_per_thread is None else
                          (f"--maxrregcount={max_registers_per_thread}",))
        return (
            str(self.compiler),
            *self.target_flags,
            "-std=c++20",
            "-O3",
            *resource_flags,
            *self.host_flags,
            "-shared",
            f"-D{self.arch_define}",
            f"-I{FTT_INCLUDE_DIR}",
            str(source),
            "-o",
            str(output),
            *self.link_flags,
        )


def _corex_root() -> Path:
    """Resolve the COREX installation the device compiler lives in."""

    override = os.environ.get("COREX_HOME")
    if override:
        return Path(override)
    return Path("/usr/local/corex")


def iluvatar_bi_v150_toolchain() -> CppToolchain:
    root = _corex_root()
    return CppToolchain(
        name="iluvatar-bi-v150",
        compiler=root / "bin" / "clang++",
        # `-x ivcore` selects COREX's device-C++ driver mode; ivcore11 is the
        # BI-V150's device architecture. Extra load latency lets the vendor
        # scheduler move packed BF16 loads ahead of dependent dot instructions,
        # including its pass after register allocation. Measured together on
        # the full decode package; neither flag changes arithmetic ordering.
        target_flags=("-x", "ivcore", "--cuda-gpu-arch=ivcore11",
                      "-mllvm", "-iluvatar-extra-load-latency=80",
                      "-mllvm", "-iluvatar-extra-latency-post-ra=40"),
        link_flags=(f"-L{root / 'lib64'}", "-lcudart"),
        arch_define="FTT_ARCH_ILUVATAR",
    )


_TOOLCHAINS = {
    "iluvatar-bi-v150": iluvatar_bi_v150_toolchain,
}


def nvidia_toolchain(target: str) -> CppToolchain:
    """NVCC's driver syntax differs from the COREX clang driver."""
    root = Path(os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda")
    compiler = root / "bin" / "nvcc"
    if not compiler.is_file() and not (os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")):
        found = shutil.which("nvcc")
        if found:
            compiler = Path(found)
    return CppToolchain(
        name=target,
        compiler=compiler,
        target_flags=("-x", "cu", f"-arch=sm_{target.removeprefix('nvidia-sm')}"),
        link_flags=(),  # NVCC supplies cudart and its installation's library path.
        arch_define="FTT_ARCH_NVIDIA",
        host_flags=("-Xcompiler", "-fPIC"),
    )


for _target in ("nvidia-sm89", "nvidia-sm90", "nvidia-sm120"):
    _TOOLCHAINS[_target] = lambda target=_target: nvidia_toolchain(target)


def resolve_toolchain(target: str) -> CppToolchain:
    try:
        factory = _TOOLCHAINS[str(target)]
    except KeyError as error:
        raise CodegenError(
            f"The C++ backend has no device toolchain for target {target!r}; "
            f"known targets: {sorted(_TOOLCHAINS)}."
        ) from error
    toolchain = factory()
    if not toolchain.compiler.is_file():
        raise CodegenError(
            f"C++ backend device compiler {toolchain.compiler} is missing. Set "
            "CUDA_HOME/CUDA_PATH for NVIDIA or COREX_HOME for Iluvatar if "
            "the toolchain is installed elsewhere."
        )
    return toolchain


def build_cpp_package(
    source: str | Path,
    output: str | Path,
    *,
    target: str,
    max_registers_per_thread: int | None = None,
    build_provenance: dict[str, object] | None = None,
) -> Path:
    """Compile one generated source into a shared library and return its path."""

    validate_cpp_max_registers(max_registers_per_thread, target=target)
    toolchain = resolve_toolchain(target)
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    if not source_path.is_file():
        raise CodegenError(f"Generated C++ source {source_path} is missing.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = toolchain.command(
        source_path, output_path, max_registers_per_thread=max_registers_per_thread)
    if build_provenance is not None:
        version = subprocess.run(
            [str(toolchain.compiler), "--version"], capture_output=True, text=True, check=False)
        if version.returncode != 0:
            raise CodegenError(f"Cannot identify C++ compiler {toolchain.compiler}: {version.stderr.strip()}")
        build_provenance.update(
            argv=list(command), compiler={
                "path": str(toolchain.compiler), "realpath": str(toolchain.compiler.resolve()),
                "sha256": hashlib.sha256(toolchain.compiler.read_bytes()).hexdigest(),
                "version": version.stdout.strip(),
            }, started_UTC=datetime.now(timezone.utc).isoformat())
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if build_provenance is not None:
        build_provenance.update(
            actual_exit_code=result.returncode,
            stdout=result.stdout, stderr=result.stderr,
            completed_UTC=datetime.now(timezone.utc).isoformat())
    if result.returncode != 0:
        raise CodegenError(
            f"Building the generated C++ package failed ({toolchain.name}, exit "
            f"{result.returncode}).\nCommand: {' '.join(command)}\n"
            f"{result.stderr.strip()}"
        )
    if not output_path.is_file():
        raise CodegenError(
            f"The C++ package build reported success but produced no "
            f"{output_path}."
        )
    return output_path


def toolchain_available(target: str) -> bool:
    factory = _TOOLCHAINS.get(str(target))
    if factory is None:
        return False
    compiler = factory().compiler
    return compiler.is_file()


__all__ = [
    "CppToolchain",
    "FTT_INCLUDE_DIR",
    "build_cpp_package",
    "resolve_toolchain",
    "toolchain_available",
    "validate_cpp_max_registers",
]

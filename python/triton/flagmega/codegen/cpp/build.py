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
from pathlib import Path
import os
import shutil
import subprocess

from triton.flagmega.errors import CodegenError


FTT_INCLUDE_DIR = Path(__file__).parent / "ftt" / "include"


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

    def command(self, source: Path, output: Path) -> tuple[str, ...]:
        return (
            str(self.compiler),
            *self.target_flags,
            "-std=c++20",
            "-O3",
            "-fPIC",
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
            "COREX_HOME if the vendor toolchain is installed elsewhere."
        )
    return toolchain


def build_cpp_package(
    source: str | Path,
    output: str | Path,
    *,
    target: str,
) -> Path:
    """Compile one generated source into a shared library and return its path."""

    toolchain = resolve_toolchain(target)
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    if not source_path.is_file():
        raise CodegenError(f"Generated C++ source {source_path} is missing.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = toolchain.command(source_path, output_path)
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
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
    return compiler.is_file() or shutil.which(compiler.name) is not None


__all__ = [
    "CppToolchain",
    "FTT_INCLUDE_DIR",
    "build_cpp_package",
    "resolve_toolchain",
    "toolchain_available",
]

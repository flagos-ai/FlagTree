# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""C++ (FTT) codegen backend for FlagMega artifacts.

Selecting this backend produces a generated ``generated_kernels.cpp`` built
against the FTT header library under ``ftt/include``, compiled to one shared
library and launched through the plain-C ABI in ``<ftt/runtime.h>``. The
Triton backend under ``codegen/triton`` remains the default; this is the
alternative emission of the same selected, bufferized TIR.
"""

from triton.flagmega.codegen.cpp.build import (
    FTT_INCLUDE_DIR,
    CppToolchain,
    build_cpp_package,
    resolve_toolchain,
    toolchain_available,
)
from triton.flagmega.codegen.cpp.expressions import cpp_scalar_type, translate
from triton.flagmega.codegen.cpp.package import render_cpp_package_artifact
from triton.flagmega.codegen.cpp.renderer import (
    CPP_BLOCK_THREADS,
    CPP_PACKAGE_DESCRIPTOR_SCHEMA,
    CPP_RENDERER_VERSION,
    describe_cpp_package,
    render_cpp_package,
    render_cpp_source_package,
)
from triton.flagmega.codegen.cpp.templates import (
    CppKernelTemplateSpec,
    CppTemplateRegistry,
)

__all__ = [
    "CPP_BLOCK_THREADS",
    "CPP_PACKAGE_DESCRIPTOR_SCHEMA",
    "CPP_RENDERER_VERSION",
    "CppKernelTemplateSpec",
    "CppTemplateRegistry",
    "CppToolchain",
    "FTT_INCLUDE_DIR",
    "build_cpp_package",
    "cpp_scalar_type",
    "describe_cpp_package",
    "render_cpp_package",
    "render_cpp_package_artifact",
    "render_cpp_source_package",
    "resolve_toolchain",
    "toolchain_available",
    "translate",
]

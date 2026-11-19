"""Public TsingMicro backend interfaces."""

from .core_ir import (
    CORE_IR_VERSION,
    KERNEL_ABI_VERSION,
    ArgumentSpec,
    CompileArtifact,
    CompileOptions,
    Grid,
    KernelSignature,
    compile_core_ir,
)
from .runtime import (
    CoreIRRuntime,
    CoreIRRuntimeError,
    LaunchDiagnostics,
    load_core_ir_artifact,
)

__all__ = [
    "CORE_IR_VERSION",
    "KERNEL_ABI_VERSION",
    "ArgumentSpec",
    "CompileArtifact",
    "CompileOptions",
    "CoreIRRuntime",
    "CoreIRRuntimeError",
    "Grid",
    "KernelSignature",
    "LaunchDiagnostics",
    "compile_core_ir",
    "load_core_ir_artifact",
]

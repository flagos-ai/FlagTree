# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Triton package renderer."""

from triton.flagmega.codegen.triton.distributed_abi import (
    DistributedBufferABI,
    KernelExecutionKind,
    distributed_buffer_abi,
    kernel_execution_kind,
)
from triton.flagmega.codegen.triton.call_abi import (
    CALL_ABI_SCHEMA,
    FUNCTION_CALL_ABI_SCHEMA,
    LOCAL_BUFFER_ABI_SCHEMA,
    describe_function_call_abi,
    describe_kernel_call_abis,
    describe_local_buffer_abi,
)
from triton.flagmega.codegen.triton.function_schedule import (
    FUNCTION_SCHEDULE_SCHEMA,
    describe_function_schedule,
)
from triton.flagmega.codegen.triton.runtime_binding import (
    FUNCTION_RUNTIME_BINDING_SCHEMA,
    describe_function_runtime_binding,
)
from triton.flagmega.codegen.triton.function_call import (
    emit_function_call_arguments,
)
from triton.flagmega.codegen.triton.physical_access import (
    emit_active_extent,
    emit_buffer_pointer,
    emit_global_scalar_offset,
    emit_local_scalar_offset,
    emit_logical_coordinate,
    emit_owner_base,
    emit_storage_pointer,
)
from triton.flagmega.codegen.triton.renderer import renderer_registry, render_triton_package
from triton.flagmega.codegen.triton.registry import PackageRendererRegistry, PackageRendererSpec
from triton.flagmega.codegen.triton.tir_package import (
    TIR_PACKAGE_DESCRIPTOR_SCHEMA,
    describe_tir_package,
    render_tir_package,
)
from triton.flagmega.codegen.triton.templates import (
    KernelTemplateSpec,
    RenderedKernelTemplate,
    TritonTemplateRegistry,
)

__all__ = [
    "KernelTemplateSpec",
    "CALL_ABI_SCHEMA",
    "DistributedBufferABI",
    "FUNCTION_CALL_ABI_SCHEMA",
    "FUNCTION_RUNTIME_BINDING_SCHEMA",
    "FUNCTION_SCHEDULE_SCHEMA",
    "KernelExecutionKind",
    "LOCAL_BUFFER_ABI_SCHEMA",
    "PackageRendererRegistry",
    "PackageRendererSpec",
    "RenderedKernelTemplate",
    "TritonTemplateRegistry",
    "TIR_PACKAGE_DESCRIPTOR_SCHEMA",
    "describe_function_call_abi",
    "describe_function_runtime_binding",
    "describe_function_schedule",
    "describe_kernel_call_abis",
    "describe_local_buffer_abi",
    "describe_tir_package",
    "emit_active_extent",
    "emit_buffer_pointer",
    "emit_global_scalar_offset",
    "emit_function_call_arguments",
    "emit_local_scalar_offset",
    "emit_logical_coordinate",
    "emit_owner_base",
    "emit_storage_pointer",
    "distributed_buffer_abi",
    "kernel_execution_kind",
    "render_tir_package",
    "render_triton_package",
    "renderer_registry",
]

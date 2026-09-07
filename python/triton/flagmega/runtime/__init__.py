# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Generated artifact runtime namespace."""

from triton.flagmega.runtime.prepared import PreparedKernel, ResourceContract, prepare_jit_kernel
from triton.flagmega.runtime.loader import load
from triton.flagmega.runtime.lifecycle import RuntimeModule, RuntimeState
from triton.flagmega.runtime.module import (
    GeneratedAddModule,
    GeneratedElementwiseModule,
    GeneratedTirCallGraphModule,
    GeneratedTirGatedDeltaNetModule,
    GeneratedTirPagedAttentionLayerModule,
    GeneratedTirPagedAttentionModelModule,
    GeneratedTirSingleTensorModule,
    GatedDeltaNetState,
    create_tir_runtime,
)
from triton.flagmega.runtime.registry import RuntimePackageRegistry, RuntimePackageSpec, package_registry
from triton.flagmega.runtime.workspace_diagnostics import (
    WorkspaceViewSpec,
    materialize_memory_pool_views,
    materialize_workspace_views,
    memory_pool_view_specs,
    workspace_view_specs,
)
from triton.flagmega.runtime.tensor_descriptor import TensorDescriptorCache

__all__ = [
    "GeneratedAddModule",
    "GeneratedElementwiseModule",
    "GeneratedTirCallGraphModule",
    "GeneratedTirGatedDeltaNetModule",
    "GeneratedTirPagedAttentionLayerModule",
    "GeneratedTirPagedAttentionModelModule",
    "GeneratedTirSingleTensorModule",
    "PreparedKernel",
    "GatedDeltaNetState",
    "create_tir_runtime",
    "ResourceContract",
    "RuntimePackageRegistry",
    "RuntimePackageSpec",
    "RuntimeModule",
    "RuntimeState",
    "TensorDescriptorCache",
    "WorkspaceViewSpec",
    "load",
    "materialize_memory_pool_views",
    "materialize_workspace_views",
    "memory_pool_view_specs",
    "package_registry",
    "prepare_jit_kernel",
    "workspace_view_specs",
]

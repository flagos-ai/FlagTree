# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Compatibility imports for existing Python checkpoints and pass clients.

Normal lowering now creates kernel definitions, not per-op PrimFunctions.
"""

from .materialize_kernel_definitions import (
    MaterializeKernelDefinitionsPass as MaterializeKernelPrimFunctionsPass,
    materialize_kernel_definitions as materialize_kernel_prim_functions,
)

__all__ = ["MaterializeKernelPrimFunctionsPass", "materialize_kernel_prim_functions"]

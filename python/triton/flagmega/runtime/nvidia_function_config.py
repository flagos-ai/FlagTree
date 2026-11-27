# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Consistent shared/L1 preferences for jointly resident SM90 programs."""

import ctypes

from triton.flagmega.errors import RuntimeContractError

_PREFERRED_SHARED_MEMORY_CARVEOUT = 9


def configure_joint_shared_carveout(prepared, required_bytes, physical_capacity):
    """Request enough shared capacity for the admitted simultaneous programs.

    Native module loading assigns PREFER_SHARED only above 48 KiB per block.
    Mixing those functions with smaller default-cache functions may insert a
    device synchronization when the cache preference changes. A common
    percentage expresses the admitted joint footprint without depending on
    enqueue order or reserving all shared capacity for every DAG.
    """
    if physical_capacity <= 0 or required_bytes > physical_capacity:
        raise RuntimeContractError(
            "Joint program residency exceeds or lacks a physical shared-memory capacity."
        )
    percentage = -(-required_bytes * 100 // physical_capacity)
    try:
        library = ctypes.CDLL("libcuda.so.1")
        setter = library.cuFuncSetAttribute
    except (OSError, AttributeError) as error:
        raise RuntimeContractError(
            "The CUDA driver must expose cuFuncSetAttribute for concurrent program cache configuration."
        ) from error
    setter.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
    setter.restype = ctypes.c_int
    for index, kernel in enumerate(prepared):
        status = setter(
            ctypes.c_void_p(kernel.compiled_kernel.function),
            _PREFERRED_SHARED_MEMORY_CARVEOUT,
            percentage,
        )
        if status != 0:
            raise RuntimeContractError(
                f"cuFuncSetAttribute failed for concurrent program {index}: CUDA error {status}."
            )
    return percentage

# Copyright 2025-     FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Triton runtime driver entry point for EVAS."""

import sys
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget
from .launcher import EvasLauncher


class EvasUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(EvasUtils, cls).__new__(cls)
        return cls.instance

    # Supply the fields expected by Triton's runtime initialization.
    @staticmethod
    def get_device_properties(device):
        return {
            "max_shared_mem": 2**20,
            "multiprocessor_count": None,
            "sm_clock_rate": None,
            "mem_clock_rate": None,
            "mem_bus_width": None,
        }

    # Pass the ELF to the launcher, which owns the cached SDK module handle.
    @staticmethod
    def load_binary(name, kernel_obj, shared, device):
        return (None,  # module
                kernel_obj,  # function
                None,  # n_regs
                4,  # n_spills
                sys.maxsize,  # n_max_threads
                )


class EvasDriver(DriverBase):

    def __init__(self):
        super().__init__()
        self.utils = EvasUtils()
        self.launcher_cls = EvasLauncher
        self.binary_ext = "elf"

    @staticmethod
    def is_active():
        return True

    def map_python_to_cpp_type(self, ty: str) -> str:
        """
        Converts a Triton type string to its corresponding C++ type string for EV backend.
        """
        type_mapping = {
            "i1": "int8_t", "i8": "int8_t", "i16": "int16_t", "i32": "int32_t", "i64": "int64_t", "u1": "uint8_t", "u8":
            "uint8_t", "u16": "uint16_t", "u32": "uint32_t", "u64": "uint64_t", "fp16": "float", "bf16": "float",
            "fp32": "float", "f32": "float", "fp64": "double", "*fp32": "float*", "*fp16": "float*", "*fp8e4nv":
            "uint8_t*", "*i1": "int8_t*", "*i8": "int8_t*", "*i16": "int16_t*", "*i32": "int32_t*", "*i64": "int64_t*",
            "*u8": "uint8_t*"
        }
        return type_mapping.get(ty, "void*")

    def get_device_capability(self):
        return ("epu", 0)

    def get_active_torch_device(self):
        """
        The runtime adapter currently exposes no active torch device here.
        """
        return None

    def get_benchmarker(self):
        """
        Return the benchmarking function that this backend should use by default.
        For EV backend, we provide a simple benchmarker.
        """

        def simple_benchmarker(kernel_call, *, quantiles=None, **kwargs):
            # Simple benchmarker that just calls the kernel once
            # In a real implementation, you might want to call it multiple times
            # and measure performance
            kernel_call(**kwargs)
            if quantiles is None:
                quantiles = [0.5]  # Default to median
            return [0.0] * len(quantiles)  # Placeholder timing

        return simple_benchmarker

    def get_current_stream(self, device):
        return None

    def get_current_device(self):
        # Preserve the target identifier expected by Triton's runtime.
        return "epu"

    def set_current_device(self, device):
        assert device == "epu"
        return

    def get_current_target(self):
        return GPUTarget("epu", 0, 0)

    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        return args

    def get_device_interface(self):
        import torch

        return torch.lpu

    def get_empty_cache_for_benchmark(self):
        import torch

        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device="lpu")

    def clear_cache(self, cache):
        cache.zero_()

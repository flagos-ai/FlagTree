"""MThreads flagcx backend configuration.

Provides FlagcxRuntimeConfig and FlagCXBackendAdapter for MThreads GPUs.
"""

import os
from contextlib import contextmanager
from triton.experimental._flagcx_config import FlagcxRuntimeConfig, FlagCXBackendAdapter, Distributed


class MthreadsFlagcxRuntimeConfig(FlagcxRuntimeConfig):
    """MThreads-specific flagcx runtime configuration."""

    def _is_available_impl(self):
        from .flagcx_wrapper import FLAGCXLibrary  # noqa: F401
        return True

    def _get_bitcode_paths(self):
        return []


class MthreadsFlagCXBackendAdapter(FlagCXBackendAdapter):
    """MThreads backend adapter for flagcx."""

    @property
    def device_type(self) -> str:
        return 'musa'

    @property
    def distributed_backend_name(self) -> str:
        return 'mccl'

    @property
    def allocator_class(self):
        import torch_musa
        return torch_musa.memory.MUSAPluggableAllocator

    @contextmanager
    def compile_allocator_guard(self):
        import torch.utils.cpp_extension
        torch.utils.cpp_extension.CUDA_HOME = os.environ.get('MUSA_HOME', '/usr/local/musa')
        yield


flagcx_rt_conf = MthreadsFlagcxRuntimeConfig()
backend_adapter = MthreadsFlagCXBackendAdapter()

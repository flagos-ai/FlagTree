"""NVIDIA flagcx backend configuration.

Provides FlagcxRuntimeConfig and FlagCXBackendAdapter for NVIDIA GPUs.
"""

from pathlib import Path
from triton.experimental._flagcx_config import FlagcxRuntimeConfig, FlagCXBackendAdapter, Distributed


class NvidiaFlagcxRuntimeConfig(FlagcxRuntimeConfig):
    """NVIDIA-specific flagcx runtime configuration."""

    def _is_available_impl(self):
        from .flagcx_wrapper import FLAGCXLibrary  # noqa: F401
        return True


class NvidiaFlagCXBackendAdapter(FlagCXBackendAdapter):
    """NVIDIA backend adapter for flagcx."""

    @property
    def device_type(self) -> str:
        return 'cuda'

    @property
    def distributed_backend_name(self) -> str:
        return 'nccl'

    @property
    def allocator_class(self):
        import torch.cuda.memory
        return torch.cuda.memory.CUDAPluggableAllocator


flagcx_rt_conf = NvidiaFlagcxRuntimeConfig()
backend_adapter = NvidiaFlagCXBackendAdapter()

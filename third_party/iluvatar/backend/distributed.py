"""Iluvatar FlagCX runtime discovery."""

from pathlib import Path
from triton.experimental._flagcx_config import (
    CudaFlagCXBackendAdapter,
    Distributed as _Distributed,
    FlagcxRuntimeConfig as _FlagcxRuntimeConfig,
)


class FlagcxRuntimeConfig(_FlagcxRuntimeConfig):
    require_bitcode = False

    def __init__(self, path_order=0):
        super().__init__(Path(__file__).parent, path_order)


flagcx_rt_conf = FlagcxRuntimeConfig()
backend_adapter = CudaFlagCXBackendAdapter()


class Distributed(_Distributed):

    def __init__(self):
        super().__init__(flagcx_rt_conf)

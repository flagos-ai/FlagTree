"""Nvidia FlagCX runtime discovery."""

import os
from pathlib import Path
from triton.experimental._flagcx_config import (
    CudaFlagCXBackendAdapter,
    Distributed as _Distributed,
    FlagcxRuntimeConfig as _FlagcxRuntimeConfig,
)


class FlagcxRuntimeConfig(_FlagcxRuntimeConfig):

    def __init__(self, path_order=0):
        super().__init__(Path(__file__).parent, path_order, require_wheel=not self.is_dev_mode())

    def is_dev_mode(self):
        keys = ("DIST_DEV", "DEV_MODE", "FLAGCX_DEV")
        return any(os.environ.get(key) in ("1", "true", "True", "ON") for key in keys)


flagcx_rt_conf = FlagcxRuntimeConfig()
backend_adapter = CudaFlagCXBackendAdapter()


class Distributed(_Distributed):

    def __init__(self):
        super().__init__(flagcx_rt_conf)

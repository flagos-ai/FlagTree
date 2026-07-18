import os

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").upper() in {"1", "ON", "TRUE", "YES", "Y"}


class RPUUtils:

    def get_device_properties(self, device):
        return {
            "max_shared_mem": int(os.getenv("RPU_LOCAL_SPM_BYTES", "0")),
            "rpu_core_count": int(os.getenv("RPU_CORE_COUNT", "1")),
            "rpu_vlm_bytes": int(os.getenv("RPU_VLM_BYTES", "0")),
            "rpu_local_spm_bytes": int(os.getenv("RPU_LOCAL_SPM_BYTES", "0")),
        }

    def load_binary(self, name, kernel, shared, device):
        return kernel, name, 0, 0, 1


class RPULauncher:

    def __init__(self, src, metadata):
        self.src = src
        self.metadata = metadata

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Use triton.backends.rpu.aot.RPUAOTKernel for AOT launch")


class RPUDriver(DriverBase):
    launcher_cls = RPULauncher

    @staticmethod
    def is_active():
        return _env_flag("TRITON_RPU_ACTIVE")

    def __init__(self):
        self.utils = RPUUtils()

    def get_current_target(self):
        arch = os.getenv("RPU_ARCH", "rpu-v1")
        lanes = int(os.getenv("RPU_LOGICAL_LANES", "1"))
        return GPUTarget("rpu", arch, lanes)

    def get_current_device(self):
        return int(os.getenv("RPU_DEVICE", "0"))

    def get_current_stream(self, idx=None):
        return int(os.getenv("RPU_STREAM", "0"))

    def set_current_device(self, device):
        os.environ["RPU_DEVICE"] = str(device)

    # ---- v3.6 DriverBase abstract methods ----

    def map_python_to_cpp_type(self, ty: str) -> str:
        mapping = {
            "i1": "bool",
            "i8": "int8_t",
            "i16": "int16_t",
            "i32": "int32_t",
            "i64": "int64_t",
            "u8": "uint8_t",
            "u16": "uint16_t",
            "u32": "uint32_t",
            "u64": "uint64_t",
            "fp16": "_Float16",
            "bf16": "uint16_t",
            "fp32": "float",
            "fp64": "double",
        }
        if ty.startswith("*"):
            base = mapping.get(ty[1:])
            if base is None:
                raise NotImplementedError(f"RPU driver: unknown pointer base type {ty[1:]!r}")
            return f"{base}*"
        if ty in mapping:
            return mapping[ty]
        raise NotImplementedError(f"RPU driver: unknown type {ty!r}")

    def get_active_torch_device(self):
        import torch
        return torch.device("cpu")

    def get_benchmarker(self):

        def _rpu_benchmark_unsupported(kernel_call, *, quantiles, **_kwargs):
            raise NotImplementedError("RPU benchmarking is not supported in this port; the launcher "
                                      "raises by design (see RPULauncher.__call__)")

        return _rpu_benchmark_unsupported

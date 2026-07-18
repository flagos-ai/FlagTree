import os

from triton.backends.compiler import GPUTarget


def spec_get_stub_target() -> GPUTarget:
    arch = os.environ.get("TRITON_OVERRIDE_ARCH") or os.environ.get("TRITON_MUSA_ARCH") or "ph1"
    return GPUTarget("musa", arch, 32)

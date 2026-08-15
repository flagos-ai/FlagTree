from triton.backends.compiler import GPUTarget


def spec_get_stub_target() -> GPUTarget:
    return GPUTarget("cuda", 80, 32)

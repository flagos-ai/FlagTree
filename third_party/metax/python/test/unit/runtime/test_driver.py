import sys
from concurrent.futures import ThreadPoolExecutor
import torch
import pytest
from types import SimpleNamespace

import triton
import triton.language as tl


@pytest.mark.parametrize("nested", [False, True])
def test_metax_tensor_descriptor_launcher(nested):
    from triton.backends.metax import driver
    from triton.runtime.build import compile_module_from_src

    a = SimpleNamespace(base=0, shape=[37, 64], strides=[80, 1], padding="nan")
    b = SimpleNamespace(base=0, shape=[259], strides=[1], padding="zero")
    a_type = "tensordesc<fp8e5[16,64]>"
    signature = {0: (a_type, "i32") if nested else a_type, 1: "constexpr", 2: "fp32", 3: "tensordesc<fp8e4nv[64]>"}
    metadata = (0, 1, 1, 0, 0, None, None, (4, 1, 0, 1, 1, 1), None, None, None)
    args = ((a, 7) if nested else a, "constant", 1.25, b)
    packed_a = [0, 37, 64, 80, 1, True, 37, 64, 80, 1]
    packed_b = [0, 259, 1, False, 259, 1]
    expected = [tuple([*packed_a, 7])] if nested else packed_a
    capture = driver._wrap_tensor_descriptors(lambda *values: values, signature)
    assert capture(*metadata, *args) == (*metadata, *expected, "constant", 1.25, *packed_b)
    with pytest.raises(TypeError, match="MetaX launcher expected"):
        capture(*metadata, *args[:-1])

    # Exercise the generated C parser without a GPU: null bases and a zero grid.
    source = driver.make_launcher({}, signature, {})
    module = compile_module_from_src(source, "__triton_launcher", include_dirs=driver.hdrmaca_dirs(),
                                     library_dirs=driver.library_dirs(), libraries=["mcruntime"])
    launch = driver._wrap_tensor_descriptors(module.launch, signature)
    assert launch(*metadata, *args) is None


def test_is_lazy():
    from importlib import reload
    reload(sys.modules["triton.runtime.driver"])
    reload(sys.modules["triton.runtime"])
    assert triton.runtime.driver._active is None
    assert triton.runtime.driver._default is None
    assert isinstance(triton.runtime.driver.active, getattr(triton.backends.driver, "DriverBase"))
    assert isinstance(triton.runtime.driver.default, getattr(triton.backends.driver, "DriverBase"))
    utils = triton.runtime.driver.active.utils  # noqa: F841


def test_kernel_in_thread(device):
    # Test calling in a new thread sets a valid device context
    buf = torch.zeros((38016 * 1024, ), dtype=torch.float32, device=device)

    @triton.jit
    def _kernel(P, BLOCK: tl.constexpr):
        pid = tl.program_id(0).to(tl.int64)
        offset = pid * BLOCK + tl.arange(0, BLOCK)

        p = tl.load(P + offset)
        tl.store(P + offset, p)

    def call_triton():
        N = buf.numel()
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]), )
        _kernel[grid](buf, BLOCK=1024)
        getattr(torch, device).synchronize()

    call_triton()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(call_triton)
        future.result()

import pytest

from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend


def test_min_ctas_per_sm_is_a_typed_cuda_compile_option():
    backend = CUDABackend(GPUTarget("cuda", 120, 32))
    default = backend.parse_options({})
    bounded = backend.parse_options({"min_ctas_per_sm": 1})

    assert default.min_ctas_per_sm is None
    assert bounded.min_ctas_per_sm == 1
    assert bounded.hash() != default.hash()


def test_min_ctas_per_sm_must_be_positive():
    backend = CUDABackend(GPUTarget("cuda", 120, 32))

    with pytest.raises(ValueError, match="must be positive"):
        backend.parse_options({"min_ctas_per_sm": 0})

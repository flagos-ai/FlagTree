from importlib import import_module
import pytest
from triton.backends.compiler import GPUTarget
from triton._C import libtriton
if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU backend is not built", allow_module_level=True)
_backend = import_module("triton.backends.hcu.compiler")
HIPBackend, HIPOptions = _backend.HIPBackend, _backend.HIPOptions


@pytest.mark.parametrize('value', [0, 1, 33024, 65536, 1 << 40])
def test_budget_is_offline_resource_intent(value):
    backend = HIPBackend(GPUTarget('hip', 'gfx936', 64))
    options = backend.parse_options(dict(min_shared_memory=value))
    assert options.min_shared_memory == value


@pytest.mark.parametrize('value', [-1, True, False, 1.5, '65536', None])
def test_invalid_budget_rejected_before_launch(value):
    with pytest.raises(ValueError, match='min_shared_memory'):
        HIPOptions(arch='gfx936', min_shared_memory=value)


def test_default_does_not_reserve_memory():
    assert HIPOptions(arch='gfx936').min_shared_memory == 0

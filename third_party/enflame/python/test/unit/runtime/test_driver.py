import sys

import triton
import importlib.util
if importlib.util.find_spec("triton.backends.enflame") is None:
    import triton_gcu.triton
import torch_gcu


def test_is_lazy():
    from importlib import reload
    reload(sys.modules["triton.runtime.driver"])
    reload(sys.modules["triton.runtime"])
    mod = sys.modules[triton.runtime.driver.__module__]
    assert isinstance(triton.runtime.driver.active, getattr(mod, "LazyProxy"))
    assert triton.runtime.driver.active._obj is None
    utils = triton.runtime.driver.active.utils  # noqa: F841
    assert issubclass(triton.runtime.driver.active._obj.__class__, getattr(triton.backends.driver, "DriverBase"))

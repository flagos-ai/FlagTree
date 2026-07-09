import sys
import os

import pytest
import triton


def test_is_lazy():
    from importlib import reload
    reload(sys.modules["triton.runtime.driver"])
    reload(sys.modules["triton.runtime"])
    assert triton.runtime.driver._active is None
    assert triton.runtime.driver._default is None
    utils = triton.runtime.driver.active.utils  # noqa: F841
    assert issubclass(triton.runtime.driver.active.__class__, getattr(triton.backends.driver, "DriverBase"))


@pytest.mark.skipif(os.environ.get("FLAGTREE_BACKEND") != "xpu", reason="XPU backend is not active")
def test_xpu_driver_smoke():
    from triton.runtime import driver

    assert set(triton.backends.backends) == {"xpu"}
    assert type(driver.active).__name__ == "XPUDriver"
    assert driver.active.get_current_target().backend == "xpu"

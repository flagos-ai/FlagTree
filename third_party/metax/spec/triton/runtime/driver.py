from __future__ import annotations

from ..backends import backends, DriverBase
from .. import knobs
from ..backends.metax.driver import MacaDriver


def _create_driver() -> DriverBase:
    active_drivers = [x.driver for x in backends.values() if x.driver.is_active()]
    # prefer maca driver since knobs use_maca is set when torch.cuda.is_available()
    if knobs.metax.use_maca and len(active_drivers) > 1:
        active_drivers = [active_driver for active_driver in active_drivers if active_driver == MacaDriver]
    if len(active_drivers) != 1:
        raise RuntimeError(f"{len(active_drivers)} active drivers ({active_drivers}). There should only be one.")
    return active_drivers[0]()


class DriverConfig:

    def __init__(self) -> None:
        self._default: DriverBase | None = None
        self._active: DriverBase | None = None

    @property
    def default(self) -> DriverBase:
        if self._default is None:
            self._default = _create_driver()
        return self._default

    @property
    def active(self) -> DriverBase:
        if self._active is None:
            self._active = self.default
        return self._active

    def set_active(self, driver: DriverBase) -> None:
        self._active = driver

    def reset_active(self) -> None:
        self._active = self.default


driver = DriverConfig()


# flagtree backend specialization
def spec(function_name: str, *args, **kwargs):
    if hasattr(driver.active, "spec"):
        spec = driver.active.spec
        if hasattr(spec, function_name):
            func = getattr(spec, function_name)
            return func(*args, **kwargs)
    return None


# flagtree backend func specialization
def spec_func(function_name: str):
    if hasattr(driver.active, "spec"):
        spec = driver.active.spec
        if hasattr(spec, function_name):
            func = getattr(spec, function_name)
            return func
    return None


# flagtree backend path specialization
def spec_path(path_list: list):
    import os
    from triton._flagtree_backend import FLAGTREE_BACKEND
    if not path_list or not FLAGTREE_BACKEND:
        return
    current_path = path_list[0].replace(os.sep, "/")
    marker = "/triton/"
    idx = current_path.find(marker)
    if idx == -1:
        return
    triton_root = current_path[:idx + len("/triton")]
    rel_path = current_path[idx + len(marker):]
    backend_path = os.path.join(triton_root, "backends", FLAGTREE_BACKEND, "spec", "triton", rel_path)
    if os.path.isdir(backend_path) and backend_path not in path_list:
        path_list.insert(0, backend_path)

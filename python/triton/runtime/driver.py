from __future__ import annotations

from ..backends import backends, DriverBase


def _create_driver() -> DriverBase:
    active_drivers = [x.driver for x in backends.values() if x.driver.is_active()]
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

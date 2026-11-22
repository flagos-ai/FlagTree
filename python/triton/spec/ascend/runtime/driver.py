from __future__ import annotations

from ..backends import backends, DriverBase


def _create_driver() -> DriverBase:
    active = {name: backend.driver for name, backend in backends.items() if backend.driver.is_active()}
    if "ascend" in active and "nvidia" in active:
        import torch

        # transfer_to_npu can expose the same availability probe under both names.
        if torch.cuda.is_available is getattr(getattr(torch, "npu", None), "is_available", None):
            del active["nvidia"]
    active_drivers = list(active.values())
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

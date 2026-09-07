# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit generated-runtime resource lifecycle."""

from __future__ import annotations

from enum import Enum

from triton.flagmega.errors import RuntimeContractError


class RuntimeState(str, Enum):
    CREATED = "created"
    LOADED = "loaded"
    PREPARED = "prepared"
    CLOSED = "closed"


class RuntimeModule:
    """Base lifecycle matching runtime initialize/prepare/uninitialize phases."""

    def __init__(self) -> None:
        self._runtime_state = RuntimeState.CREATED
        self._device = None

    @property
    def runtime_state(self) -> RuntimeState:
        return self._runtime_state

    @property
    def device(self) -> str | None:
        return self._device

    def _mark_loaded(self, device: str) -> None:
        if self._runtime_state is not RuntimeState.CREATED:
            raise RuntimeContractError(
                f"Runtime module can only load from created state, got {self._runtime_state.value!r}.")
        self._device = str(device)
        self._runtime_state = RuntimeState.LOADED

    def _require_loaded(self) -> None:
        if self._runtime_state not in {RuntimeState.LOADED, RuntimeState.PREPARED}:
            raise RuntimeContractError(
                f"Runtime module must be loaded, current state is {self._runtime_state.value!r}.")

    def _mark_prepared(self) -> None:
        self._require_loaded()
        self._runtime_state = RuntimeState.PREPARED

    def _require_prepared(self) -> None:
        if self._runtime_state is not RuntimeState.PREPARED:
            raise RuntimeContractError(
                f"Runtime module must be prepared, current state is {self._runtime_state.value!r}.")

    def close(self, *, synchronize: bool = True) -> None:
        if self._runtime_state is RuntimeState.CLOSED:
            return
        if synchronize and self._device is not None and self._device.startswith("cuda"):
            try:
                import torch
            except ImportError as error:
                raise RuntimeContractError("Closing a CUDA runtime module requires PyTorch.") from error
            if torch.cuda.is_available():
                torch.cuda.synchronize(torch.device(self._device))
        self._release_resources()
        self._device = None
        self._runtime_state = RuntimeState.CLOSED

    def _release_resources(self) -> None:
        pass

    def __enter__(self):
        if self._runtime_state is RuntimeState.CLOSED:
            raise RuntimeContractError("A closed runtime module cannot be re-entered.")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = ["RuntimeModule", "RuntimeState"]

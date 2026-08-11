# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Strictly detect the active device behind one FlagTune runtime.

This module is the only FlagTune layer that interprets Triton's active target.
Callers receive a backend-neutral :class:`DeviceDescriptor`; they must not
guess vendors from ``torch.cuda`` or translate every architecture into NVIDIA
compute-capability terminology.

Only explicitly registered backends are accepted.  Detection never substitutes
placeholder hardware metadata because doing so could select or train a model
under the wrong device identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict


class FlagTuneDeviceError(RuntimeError):
    """Base class for actionable FlagTune device-boundary failures."""


class UnsupportedFlagTuneDeviceError(FlagTuneDeviceError):
    """Report an active Triton backend without a registered FlagTune adapter."""


class DeviceProbeError(FlagTuneDeviceError):
    """Report incomplete or inconsistent metadata for a supported backend."""


@dataclass(frozen=True)
class DeviceDescriptor:
    """Describe one active accelerator without exposing a vendor API object.

    Args:
        backend: Triton's canonical backend name, such as ``cuda`` or ``hip``.
        vendor: Stable hardware-vendor name used in artifact identities.
        torch_device_type: Device type accepted by the installed PyTorch build.
            ROCm intentionally uses ``cuda`` here for PyTorch compatibility.
        device_name: Runtime-reported product name.
        architecture: Backend-native target, such as ``sm90`` or ``gfx942``.
        device_index: Logical device ordinal visible to the current process.
    """

    backend: str
    vendor: str
    torch_device_type: str
    device_name: str
    architecture: str
    device_index: int


@dataclass(frozen=True)
class _BackendDescriptor:
    vendor: str
    torch_device_type: str
    normalize_architecture: Callable[[Any], str]


def _nvidia_architecture(value: Any) -> str:
    text = str(value).strip().lower()
    if text.startswith("sm"):
        suffix = text[2:].replace("_", "")
    else:
        suffix = text.replace(".", "").replace("_", "")
    if not suffix.isdigit():
        raise DeviceProbeError(f"CUDA target has invalid architecture {value!r}; expected smNN or NN")
    return f"sm{suffix}"


def _amd_architecture(value: Any) -> str:
    text = str(value).strip().lower()
    if not text.startswith("gfx") or len(text) <= 3:
        raise DeviceProbeError(f"HIP target has invalid architecture {value!r}; expected gfxNNN")
    return text


# ---------------------------------------------------------------------------
# Supported backend descriptors
# TODO add more backends here (ascend, mthreads, hygon, etc.)
# ---------------------------------------------------------------------------
_BACKENDS: Dict[str, _BackendDescriptor] = {
    "cuda": _BackendDescriptor("nvidia", "cuda", _nvidia_architecture),
    # PyTorch ROCm deliberately exposes its runtime through torch.cuda.
    "hip": _BackendDescriptor("amd", "cuda", _amd_architecture),
}


def registered_device_backends() -> tuple[str, ...]:
    """Return supported Triton backend names in deterministic order."""
    return tuple(sorted(_BACKENDS))


def _active_driver() -> Any:
    """Return Triton's active driver through a replaceable test seam."""
    from triton.runtime import driver

    return driver.active


def probe_flagtune_device(device_index: int | None = None) -> DeviceDescriptor:
    """Return strict metadata for one device on the active Triton target.

    Args:
        device_index: Optional process-visible logical ordinal.  When omitted,
            the active Triton device interface supplies its current device.

    Raises:
        UnsupportedFlagTuneDeviceError: If no adapter is registered for the
            active Triton backend.
        DeviceProbeError: If a supported backend cannot provide a product name,
            native architecture, or logical device index.
    """
    try:
        active = _active_driver()
        target = active.get_current_target()
    except Exception as exc:
        raise DeviceProbeError(f"cannot query the active Triton target: {exc}") from exc

    backend = str(getattr(target, "backend", "")).strip().lower()
    descriptor = _BACKENDS.get(backend)
    if descriptor is None:
        supported = ", ".join(registered_device_backends())
        shown = backend or "<unknown>"
        raise UnsupportedFlagTuneDeviceError(f"FlagTune does not support Triton backend {shown!r}; "
                                             f"registered backends: {supported}")

    raw_architecture = getattr(target, "arch", None)
    if raw_architecture is None:
        raise DeviceProbeError(f"supported backend {backend!r} did not report a target architecture")
    architecture = descriptor.normalize_architecture(raw_architecture)

    try:
        interface = active.get_device_interface()
        index = (int(interface.current_device()) if device_index is None else int(device_index))
        device_name = str(interface.get_device_name(index)).strip()
    except Exception as exc:
        raise DeviceProbeError(f"cannot query {backend!r} device metadata for index "
                               f"{device_index if device_index is not None else '<current>'}: {exc}") from exc
    if not device_name:
        raise DeviceProbeError(f"supported backend {backend!r} returned an empty device name "
                               f"for index {index}")

    return DeviceDescriptor(
        backend=backend,
        vendor=descriptor.vendor,
        torch_device_type=descriptor.torch_device_type,
        device_name=device_name,
        architecture=architecture,
        device_index=index,
    )

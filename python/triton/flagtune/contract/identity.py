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
"""Build safe, canonical identities for FlagTune model artifacts.

Each archive is addressed by ``platform_key/op_id/variant/dtype_key``.  The helpers
here normalize user- and runtime-derived values before they become path
segments, URL-manifest keys, or fields embedded in ``flagtune_config.yaml``.
Training/export code records the identity, while registry validation and the
model manager require an exact match at load time.

Platform keys contain only vendor and normalized product identity. Backend-native
architecture values such as ``sm90`` and ``gfx942`` remain independent device
metadata and never participate in model names, paths, or cache keys. Device
detection and backend validation live in ``triton.flagtune.runtime.device``;
this module canonicalizes the identity metadata they provide.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from triton.flagtune.contract.expressions import SafeExpressionError


class ModelIdentityError(SafeExpressionError):
    """Report an unsafe or unsupported model identity component."""


_SAFE_SEGMENT = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_DTYPE_ALIASES = {
    "bool": "bool",
    "uint8": "u8",
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
    "float16": "f16",
    "half": "f16",
    "bfloat16": "bf16",
    "float32": "f32",
    "float": "f32",
    "float64": "f64",
    "double": "f64",
}
# TODO: Add explicit, backend-neutral aliases for FP8, FP6, FP4, and related
# low-precision formats once their canonical names and artifact compatibility
# policy are defined.  Do not silently collapse distinct FP8/FP4 encodings.
_DTYPE_CANONICAL = {
    "bool": "bool",
    "u8": "uint8",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    "f16": "float16",
    "bf16": "bfloat16",
    "f32": "float32",
    "f64": "float64",
}


def validate_identity_segment(value: Any, location: str) -> str:
    """Normalize and validate one lowercase path/manifest-key segment."""
    if not isinstance(value, str):
        raise ModelIdentityError(f"{location} must be a string")
    segment = value.strip().lower()
    if not _SAFE_SEGMENT.fullmatch(segment) or segment in (".", ".."):
        raise ModelIdentityError(f"{location} is not a safe identity segment: {value!r}")
    return segment


def validate_platform_key(value: Any, location: str = "platform_key") -> str:
    """Normalize and validate a platform identity."""
    return validate_identity_segment(value, location)


def validate_op_id(value: Any, location: str = "op_id") -> str:
    """Validate a slash-separated, globally namespaced operator identifier."""
    if not isinstance(value, str) or not value.strip():
        raise ModelIdentityError(f"{location} must be a non-empty string")
    parts = value.strip().split("/")
    if len(parts) < 2:
        raise ModelIdentityError(f"{location} must be globally namespaced, for example 'vendor_namespace/operator'")
    return "/".join(validate_identity_segment(part, f"{location} segment[{index}]") for index, part in enumerate(parts))


def validate_variant_name(value: Any, location: str = "variant") -> str:
    return validate_identity_segment(value, location)


def normalize_dtype_name(value: Any) -> str:
    """Return a canonical long dtype name from torch dtype objects or strings."""
    text = str(value).strip().lower()
    if text.startswith("torch."):
        text = text[6:]
    short = _DTYPE_ALIASES.get(text, text if text in _DTYPE_CANONICAL else None)
    if short is None:
        raise ModelIdentityError(f"unsupported tensor dtype: {value!r}")
    return _DTYPE_CANONICAL[short]


def dtype_abbreviation(value: Any) -> str:
    canonical = normalize_dtype_name(value)
    return _DTYPE_ALIASES[canonical]


def make_dtype_key(dtypes: Iterable[Any]) -> str:
    abbreviations = tuple(dtype_abbreviation(value) for value in dtypes)
    if not abbreviations:
        raise ModelIdentityError("dtype identity must contain at least one tensor dtype")
    return validate_identity_segment("-".join(abbreviations), "dtype_key")


def normalize_device_name(value: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", str(value).lower())
    while tokens and tokens[0] in ("nvidia", "amd", "intel"):
        tokens.pop(0)
    if not tokens:
        raise ModelIdentityError(f"GPU device name has no usable tokens: {value!r}")
    return "-".join(tokens)


def _normalize_vendor(value: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", str(value).lower())
    if not tokens:
        raise ModelIdentityError(f"GPU vendor has no usable tokens: {value!r}")
    aliases = {
        "advanced-micro-devices": "amd",
        "nvidia-corporation": "nvidia",
        "intel-corporation": "intel",
    }
    slug = "-".join(tokens)
    return aliases.get(slug, slug)


_PLATFORM_DEVICE_ALIASES = {
    ("nvidia", "h20-3e"): "h20",
}


def make_platform_key(vendor: str, device_name: str) -> str:
    """Return the stable vendor/product component used for model artifacts."""
    vendor_key = _normalize_vendor(vendor)
    device_key = normalize_device_name(device_name)
    device_key = _PLATFORM_DEVICE_ALIASES.get((vendor_key, device_key), device_key)
    return validate_platform_key(f"{vendor_key}-{device_key}", "platform_key")


@dataclass(frozen=True)
class ModelIdentity:
    platform_key: str
    op_id: str
    variant: str
    dtype_key: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "platform_key", validate_platform_key(self.platform_key))
        object.__setattr__(self, "op_id", validate_op_id(self.op_id))
        object.__setattr__(self, "variant", validate_variant_name(self.variant))
        object.__setattr__(self, "dtype_key", validate_identity_segment(self.dtype_key, "dtype_key"))

    @property
    def artifact_key(self) -> str:
        return f"{self.platform_key}/{self.op_id}/{self.variant}/{self.dtype_key}"


def artifact_key(platform_key: str, op_id: str, variant: str, dtype_key: str) -> str:
    return ModelIdentity(platform_key, op_id, variant, dtype_key).artifact_key


def gpu_metadata(
    *,
    backend: str,
    vendor: str,
    device_name: str,
    architecture: str,
) -> Mapping[str, Any]:
    """Return serializable GPU metadata plus the matching canonical ``platform_key``."""
    return {
        "backend": validate_identity_segment(backend, "GPU backend"),
        "vendor": str(vendor),
        "device_name": str(device_name),
        "architecture": validate_identity_segment(architecture, "GPU architecture"),
        "platform_key": make_platform_key(vendor, device_name),
    }


def discover_gpu_metadata() -> Mapping[str, Any]:
    """Return canonical metadata for the active Triton GPU device.

    Device ordinal and UUID are intentionally excluded so identical cards on
    one or more hosts resolve to the same model identity.
    """
    from triton.flagtune.runtime.device import probe_flagtune_device

    descriptor = probe_flagtune_device()
    return gpu_metadata(
        backend=descriptor.backend,
        vendor=descriptor.vendor,
        device_name=descriptor.device_name,
        architecture=descriptor.architecture,
    )

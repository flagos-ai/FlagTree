import os
import sys
from typing import List, Optional

TRITON_MAX_TENSOR_NUMEL = sys.maxsize


def parse_xpu_arch(raw: Optional[str], default: Optional[int] = None) -> Optional[int]:
    """Parse an XPU architecture value, returning the default when invalid."""
    if raw is None or not raw.isascii() or not raw.isdigit():
        return default
    try:
        return int(raw, 10)
    except ValueError:
        return default


def get_xpu_arch(default: int = 3) -> int:
    """Return ``TRITON_XPU_ARCH`` as an integer, or ``default`` when invalid."""
    return parse_xpu_arch(os.environ.get("TRITON_XPU_ARCH"), default)


def validate_block_shape(shape: List[int]):
    numel = 1
    for i, d in enumerate(shape):
        if not isinstance(d, int):
            raise TypeError(f"Shape element {i} must have type `constexpr[int]`, got `constexpr[{type(d)}]")
        # XPU accepts non-power-of-two block-shape dimensions.
        numel *= d

    if numel > TRITON_MAX_TENSOR_NUMEL:
        raise ValueError(f"numel ({numel}) exceeds triton maximum tensor numel ({TRITON_MAX_TENSOR_NUMEL})")
    return numel


def _utils_extend_globals(globals_dict):
    # XPU's prebuilt Python modules import these helpers from ``triton._utils``.
    # Publish them only for an active XPU build.
    globals_dict["TRITON_MAX_TENSOR_NUMEL"] = TRITON_MAX_TENSOR_NUMEL
    globals_dict["get_xpu_arch"] = get_xpu_arch
    globals_dict["parse_xpu_arch"] = parse_xpu_arch

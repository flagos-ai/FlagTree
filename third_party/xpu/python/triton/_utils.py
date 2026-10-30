from __future__ import annotations

import os
import sys
from functools import reduce
from typing import Any, Callable, Optional, TYPE_CHECKING, Union, List, Dict

if TYPE_CHECKING:
    from .language import core
    IterableType = Union[list[Any], tuple[Any, ...], core.tuple, core.tuple_type]
    ObjPath = tuple[int, ...]

# For Triton XPU: the legalize pass does not need the max tensor numel limit,
# so allow arbitrarily large tensors (mirrors the XPU fork).
TRITON_MAX_TENSOR_NUMEL = sys.maxsize


def get_iterable_path(iterable: IterableType, path: ObjPath) -> Any:
    return reduce(lambda a, idx: a[idx], path, iterable)  # type: ignore[index]


def set_iterable_path(iterable: IterableType, path: tuple[int, ...], val: Any):
    from .language import core
    assert len(path) != 0
    prev = iterable if len(path) == 1 else get_iterable_path(iterable, path[:-1])
    assert isinstance(prev, core.tuple)
    prev._setitem(path[-1], val)


def find_paths_if(iterable: Union[IterableType, Any], pred: Callable[[ObjPath, Any], bool]) -> list[ObjPath]:
    from .language import core
    is_iterable: Callable[[Any], bool] = lambda x: isinstance(x, (list, tuple, core.tuple, core.tuple_type))
    # We need to use dict so that ordering is maintained, while set doesn't guarantee order
    ret: dict[ObjPath, None] = {}

    def _impl(path: tuple[int, ...], current: Any):
        if is_iterable(current):
            for idx, item in enumerate(current):
                _impl((*path, idx), item)
        elif pred(path, current):
            ret[path] = None

    _impl((), iterable)

    return list(ret.keys())


def is_power_of_two(x):
    return (x & (x - 1)) == 0


def parse_xpu_arch(raw: Optional[str], default: Optional[int] = None) -> Optional[int]:
    """Return ``raw`` as an architecture number, or ``default`` when it is not one.

    Why: architecture values arrive as strings from more than one place -- an
    environment variable, or an explicitly passed ``arch=`` -- and every site
    needs the same answer for "this is not a number": treat it as absent rather
    than raising.  get_xpu_arch() is the TRITON_XPU_ARCH-specific wrapper over
    this; the tle raw runtime uses it directly because it iterates a
    configurable list of variable names (TRITON_XCN_ARCH before TRITON_XPU_ARCH).

    How:
      - accept only a non-empty ASCII all-digit string, matching the full-string
        integer parse is_xpu3_backend() documents (whitespace, underscores and
        non-ASCII digits are rejected);
      - return ``default`` for anything else, including None -- so a caller that
        wants "absent" to be distinguishable passes default=None;
      - guard the conversion too: isdigit() alone is not enough, because Python
        3.11+ (and 3.10 with the limit enabled, which is the default) refuses a
        digit string longer than sys.get_int_max_str_digits().
    """
    if raw is None or not raw.isascii() or not raw.isdigit():
        return default
    try:
        return int(raw, 10)
    except ValueError:
        return default


def get_xpu_arch(default: int = 3) -> int:
    """Return TRITON_XPU_ARCH as an int, or ``default`` when it is unset or malformed.

    Why: the variable is parsed in several places that each have their own
    default (3 for the xpu backend, 4 for xcn, 5 for jupiter), but only some of
    them guarded the conversion, so a malformed value produced a bare
    ValueError from whichever site ran first -- for a plain ``import triton``
    that is python/triton/backends/__init__.py.  A malformed value is not a
    meaningful architecture, so treat it like an out-of-range one and keep the
    caller's default: that is already what _select_xpu_backend does for an
    unknown number (_XPU_ARCH_BACKEND.get(arch, "xpu")) and what
    is_xpu3_backend() does for its predicate.  Never raises.

    How:
      - delegate the parse to parse_xpu_arch(), the one place that decides what
        a malformed value means, so no caller can disagree about it;
      - the default is the caller's, because it genuinely differs per backend.

    Callers that need to distinguish "unset" from "bad value" should read
    os.environ themselves; this helper deliberately collapses both to the
    default, like every existing call site that used os.environ.get(..., "N").
    """
    return parse_xpu_arch(os.environ.get("TRITON_XPU_ARCH"), default)


def validate_block_shape(shape: List[int]):
    numel = 1
    for i, d in enumerate(shape):
        if not isinstance(d, int):
            raise TypeError(f"Shape element {i} must have type `constexpr[int]`, got `constexpr[{type(d)}]")
        # ===-------------------- For Triton XPU -----------------------===
        # [sync source] Triton XPU does not require block-shape dims to be a power
        # of two (matches the 3.0 fork). Kept disabled so kernels using e.g.
        # BLOCK_SIZE=1536 / cluster=12 / block_size_candidates tiles compile.
        # if not is_power_of_two(d):
        #     raise ValueError(f"Shape element {i} must be a power of 2")
        # ===-----------------------------------------------------------===
        numel *= d

    if numel > TRITON_MAX_TENSOR_NUMEL:
        raise ValueError(f"numel ({numel}) exceeds triton maximum tensor numel ({TRITON_MAX_TENSOR_NUMEL})")
    return numel


type_canonicalisation_dict = {
    # we canonicalise all bools to be unsigned:
    "bool": "u1",
    "int1": "u1",
    "uint1": "u1",
    "i1": "u1",
    # floating-point dtypes:
    "float8e4nv": "fp8e4nv",
    "float8e5": "fp8e5",
    "float8e4b15": "fp8e4b15",
    "float8_e4m3fn": "fp8e4nv",
    "float8e4b8": "fp8e4b8",
    "float8_e4m3fnuz": "fp8e4b8",
    "float8_e5m2": "fp8e5",
    "float8e5b16": "fp8e5b16",
    "float8_e5m2fnuz": "fp8e5b16",
    "half": "fp16",
    "float16": "fp16",
    "bfloat16": "bf16",
    "float": "fp32",
    "float32": "fp32",
    "double": "fp64",
    "float64": "fp64",
    # signed integers:
    "int8": "i8",
    "int16": "i16",
    "int": "i32",
    "int32": "i32",
    "int64": "i64",
    # unsigned integers:
    "uint8": "u8",
    "uint16": "u16",
    "uint32": "u32",
    "uint64": "u64",
    "void": "void",
}

for v in list(type_canonicalisation_dict.values()):
    type_canonicalisation_dict[v] = v


def canonicalize_dtype(dtype):
    dtype_str = str(dtype).split(".")[-1]
    return type_canonicalisation_dict[dtype_str]


def canonicalize_ptr_dtype(dtype, is_const):
    return f"{'*k' if is_const else '*'}{canonicalize_dtype(dtype)}"


BITWIDTH_DICT: Dict[str, int] = {
    **{f"u{n}": n
       for n in (1, 8, 16, 32, 64)},
    **{f"i{n}": n
       for n in (1, 8, 16, 32, 64)},
    **{f"fp{n}": n
       for n in (16, 32, 64)},
    **{f"fp8{suffix}": 8
       for suffix in ("e4nv", "e4b15", "e4b8", "e5", "e5b16")},
    "bf16": 16,
    "void": 0,
}

for k, v in type_canonicalisation_dict.items():
    BITWIDTH_DICT[k] = BITWIDTH_DICT[v]


def get_primitive_bitwidth(dtype: str) -> int:
    return BITWIDTH_DICT[dtype]


def is_namedtuple(val):
    return isinstance(val, type) and issubclass(val, tuple) and hasattr(val, "_fields")

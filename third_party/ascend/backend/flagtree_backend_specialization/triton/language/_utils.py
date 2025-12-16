from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union, Dict
if TYPE_CHECKING:
    from triton.language import core
    IterableType = Union[list[Any], tuple[Any, ...], core.tuple, core.tuple_type]
    ObjPath = tuple[int, ...]


def block_shape_disable_check_power_of_two():
    return True


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


def get_primitive_bitwidth(dtype: str) -> int:
    return BITWIDTH_DICT[dtype]



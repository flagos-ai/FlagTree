from typing import TYPE_CHECKING, Any, Union, Dict

def get_language_utils_IterableType_ObjPath():
    from triton.language import core
    IterableType = Union[list[Any], tuple[Any, ...], core.tuple, core.tuple_type]
    ObjPath = tuple[int, ...]
    return IterableType, ObjPath


TRITON_MAX_TENSOR_NUMEL = 1048576

def get_triton_max_tensor_numel():
    return TRITON_MAX_TENSOR_NUMEL


def is_block_shape_check_power_of_two():
    return False


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


def get_language_utils_BITWIDTH_DICT():
    return BITWIDTH_DICT


def get_primitive_bitwidth(dtype: str) -> int:
    return BITWIDTH_DICT[dtype]



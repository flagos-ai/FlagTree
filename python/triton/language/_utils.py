from __future__ import annotations

from typing import List

# flagtree backend specialization
from triton.runtime.driver import flagtree_backend_specialization
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    IterableType, ObjPath = flagtree_backend_specialization('get_language_utils_IterableType_ObjPath')


TRITON_MAX_TENSOR_NUMEL = flagtree_backend_specialization('get_triton_max_tensor_numel')


def is_power_of_two(x):
    return (x & (x - 1)) == 0


def validate_block_shape(shape: List[int]):
    numel = 1
    for i, d in enumerate(shape):
        if not isinstance(d, int):
            raise TypeError(f"Shape element {i} must have type `constexpr[int]`, got `constexpr[{type(d)}]")
        if flagtree_backend_specialization('is_block_shape_check_power_of_two') and not is_power_of_two(d):
            raise ValueError(f"Shape element {i} must be a power of 2")
        numel *= d

    if numel > TRITON_MAX_TENSOR_NUMEL:
        raise ValueError(f"numel ({numel}) exceeds triton maximum tensor numel ({TRITON_MAX_TENSOR_NUMEL})")
    return numel


# flagtree backend specialization
from triton.runtime.driver import flagtree_backend_specialization
BITWIDTH_DICT = flagtree_backend_specialization('get_language_utils_BITWIDTH_DICT')


# flagtree backend specialization
from triton.runtime.driver import flagtree_backend_func_specialization
get_primitive_bitwidth = flagtree_backend_func_specialization("get_primitive_bitwidth")

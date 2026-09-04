# Copyright 2026- Xcoresigma Technology Co., Ltd
from enum import IntEnum

import triton.language as tl


class SortImpl(IntEnum):
    SORT_IMPL_BASE = 0
    SORT_IMPL_S4096_K129_512 = 1
    SORT_IMPL_S4096_K1_128_K2048 = 2


SORT_IMPL_BASE = tl.constexpr(SortImpl.SORT_IMPL_BASE)
SORT_IMPL_S4096_K129_512 = tl.constexpr(SortImpl.SORT_IMPL_S4096_K129_512)
SORT_IMPL_S4096_K1_128_K2048 = tl.constexpr(SortImpl.SORT_IMPL_S4096_K1_128_K2048)

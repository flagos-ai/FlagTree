import ast
from types import SimpleNamespace

from triton.runtime.adjust_kernel_param import (
    analyze_kernel_dependencies,
    auto_adjust_block_sizes,
    clear_analysis_cache,
)


class _ParsedKernel:

    def __init__(self, source):
        self.source = source
        self.__globals__ = {}

    def parse(self):
        return ast.parse(self.source)


def _analyze(source):
    clear_analysis_cache()
    return analyze_kernel_dependencies(_ParsedKernel(source))[0]


def test_semantic_mask_does_not_pair_block_size_with_init_blocks():
    source = """
def topk_kernel(
    s_ptr,
    prefix_len,
    init_blocks: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    off_k = tl.arange(0, BLOCK_SIZE_K)
    valid_blocks = prefix_len + 1
    s_ptrs = s_ptr + off_k
    for i in tl.range(0, valid_blocks, BLOCK_SIZE_K):
        causal_mask = i + off_k < valid_blocks
        init_mask = i + off_k < init_blocks
        score = tl.load(s_ptrs, mask=causal_mask)
        score = tl.where(init_mask, 1.0, score)
        s_ptrs += BLOCK_SIZE_K
"""
    load_map = _analyze(source)

    assert load_map == {}

    clear_analysis_cache()
    config = SimpleNamespace(pre_hook=None, kwargs={"BLOCK_SIZE_K": 2048})
    current = {"BLOCK_SIZE_K": 2048}
    auto_adjust_block_sizes(
        {"init_blocks": 4},
        _ParsedKernel(source),
        [config],
        current,
        config,
    )
    assert current["BLOCK_SIZE_K"] == 2048
    assert config.kwargs["BLOCK_SIZE_K"] == 2048


def test_load_mask_pairing_ignores_unrelated_semantic_mask():
    load_map = _analyze("""
def masked_load_kernel(
    x_ptr,
    N,
    init_blocks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    init_mask = offsets < init_blocks
    boundary_mask = offsets < N
    value = tl.load(x_ptr + offsets, mask=boundary_mask)
    value = tl.where(init_mask, 1.0, value)
""")

    assert load_map == {"BLOCK_SIZE": "N"}


def test_standard_boundary_mask_adjusts_block_size():
    source = """
def masked_load_kernel(
    x_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    value = tl.load(x_ptr + offsets, mask=offsets < N)
"""
    assert _analyze(source) == {"BLOCK_SIZE": "N"}

    clear_analysis_cache()
    config = SimpleNamespace(pre_hook=None, kwargs={"BLOCK_SIZE": 128})
    current = {"BLOCK_SIZE": 128}
    auto_adjust_block_sizes(
        {"N": 33},
        _ParsedKernel(source),
        [config],
        current,
        config,
    )
    assert current["BLOCK_SIZE"] == 64
    assert config.kwargs["BLOCK_SIZE"] == 64


def test_explicit_cdiv_pairing_does_not_require_load_mask():
    load_map = _analyze("""
def cdiv_load_kernel(
    x_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    num_blocks = tl.cdiv(N, BLOCK_SIZE)
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    value = tl.load(x_ptr + offsets)
""")

    assert load_map == {"BLOCK_SIZE": "N"}

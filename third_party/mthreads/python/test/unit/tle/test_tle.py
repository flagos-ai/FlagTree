import inspect

import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from test_tle_utils import compile_to_ttir


def test_tle_language_import_exports_load_signature():
    assert tle.load is not tl.load

    assert list(inspect.signature(tle.load).parameters) == [
        "pointer",
        "mask",
        "other",
        "boundary_check",
        "padding_option",
        "cache_modifier",
        "eviction_policy",
        "volatile",
        "is_async",
        "_semantic",
    ]
    assert list(inspect.signature(tle.gpu.memory_space).parameters) == [
        "input",
        "space",
        "_builder",
        "_semantic",
    ]


def test_tle_load_sets_async_bool_attr():

    @triton.jit
    def tl_kernel(src, dst):
        offsets = tl.arange(0, 16)
        values = tl.load(src + offsets)
        tl.store(dst + offsets, values)

    @triton.jit
    def tle_kernel(src, dst, ASYNC: tl.constexpr):
        offsets = tl.arange(0, 16)
        values = tle.load(src + offsets, is_async=ASYNC)
        tl.store(dst + offsets, values)

    signature = {"src": "*fp32", "dst": "*fp32", "ASYNC": "constexpr"}
    tl_ttir = compile_to_ttir(tl_kernel, {"src": "*fp32", "dst": "*fp32"})
    non_async_ttir = compile_to_ttir(tle_kernel, signature, {"ASYNC": False})
    async_ttir = compile_to_ttir(tle_kernel, signature, {"ASYNC": True})

    assert "tt.load.async" not in tl_ttir
    assert tl_ttir.count(" = tt.load ") == 1
    assert non_async_ttir.count(" = tt.load ") == 1
    assert "tt.load.async = false" in non_async_ttir
    assert "tt.load.async = true" in async_ttir


def test_tle_gpu_memory_space_sets_shared_memory_string_attr():

    @triton.jit
    def kernel(src, dst):
        offsets = tl.arange(0, 16)
        values = tle.load(src + offsets)
        values = tle.gpu.memory_space(values, "shared_memory")
        tl.store(dst + offsets, values)

    ttir = compile_to_ttir(kernel, {"src": "*fp32", "dst": "*fp32"})

    assert ttir.count(" = tt.load ") == 1
    assert 'tt.memory_space = "shared_memory"' in ttir


def test_tle_load_forwards_tl_load_options():

    @triton.jit
    def kernel(src, dst):
        block = tl.make_block_ptr(src, shape=(16, ), strides=(1, ), offsets=(0, ), block_shape=(16, ), order=(0, ))
        values = tle.load(block, boundary_check=(0, ), padding_option="zero", cache_modifier=".cg",
                          eviction_policy="evict_last", volatile=True, is_async=True)
        offsets = tl.arange(0, 16)
        tl.store(dst + offsets, values)

    ttir = compile_to_ttir(kernel, {"src": "*fp32", "dst": "*fp32"})

    assert "tt.load.async = true" in ttir
    assert "boundaryCheck = array<i32: 0>" in ttir
    assert "padding = 1 : i32" in ttir

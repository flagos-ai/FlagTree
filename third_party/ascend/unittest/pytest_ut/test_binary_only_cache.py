import triton
import triton.language as tl
from triton import knobs
from triton.compiler import ASTSource
from triton.runtime.cache import FileCacheManager


def test_binary_only_cache_retains_npubin(tmp_path):

    @triton.jit
    def store_offsets(out):
        offsets = tl.arange(0, 32)
        tl.store(out + offsets, offsets)

    source = ASTSource(store_offsets, {"out": "*i32"})
    with knobs.cache.scope(), knobs.compilation.scope():
        knobs.cache.dir = str(tmp_path)
        knobs.cache.manager_class = FileCacheManager
        knobs.compilation.store_binary_only = True
        knobs.compilation.always_compile = False

        cold = triton.compile(source)
        assert cold.kernel
        assert f"{source.name}.npubin" in cold.metadata_group
        assert f"{source.name}.ttir" not in cold.metadata_group

        warm = triton.compile(source)
        assert warm.kernel == cold.kernel

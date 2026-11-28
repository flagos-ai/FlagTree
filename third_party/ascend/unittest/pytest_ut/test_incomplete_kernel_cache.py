from pathlib import Path

import pytest

import triton
import triton.language as tl
from triton import knobs
from triton.compiler import ASTSource
from triton.runtime.cache import FileCacheManager, get_cache_manager


@pytest.mark.parametrize("trigger", ["missing", "empty", "evicted", "forced"])
def test_recompile_incomplete_kernel_cache(tmp_path, monkeypatch, trigger):

    @triton.jit
    def store_offsets(out):
        offsets = tl.arange(0, 32)
        tl.store(out + offsets, offsets)

    source = ASTSource(store_offsets, {"out": "*i32"})
    with knobs.cache.scope(), knobs.compilation.scope():
        knobs.cache.dir = str(tmp_path)
        knobs.cache.manager_class = FileCacheManager
        knobs.compilation.store_binary_only = False
        knobs.compilation.always_compile = False
        cache_hits = []
        knobs.compilation.listener = lambda *, cache_hit, **_: cache_hits.append(cache_hit)

        compiled = triton.compile(source)
        metadata_name = f"{source.name}.json"
        binary_name = f"{source.name}.npubin"
        binary = Path(compiled.metadata_group[binary_name])
        stale_name = f"{source.name}.stale"
        cache = get_cache_manager(compiled.hash)
        old_group = dict(compiled.metadata_group)
        old_group[stale_name] = cache.put("stale artifact", stale_name)
        cache.put_group(metadata_name, old_group)

        if trigger == "missing":
            binary.unlink()
        elif trigger == "empty":
            binary.write_bytes(b"")
        elif trigger == "forced":
            knobs.compilation.always_compile = True

        with monkeypatch.context() as patch:
            if trigger == "evicted":
                get_group = FileCacheManager.get_group

                def evict_after_lookup(manager, filename):
                    group = get_group(manager, filename)
                    if filename == metadata_name:
                        assert group is not None and group.get(binary_name) == str(binary)
                        binary.unlink()
                    return group

                patch.setattr(FileCacheManager, "get_group", evict_after_lookup)
            repaired = triton.compile(source)

        knobs.compilation.always_compile = False
        assert repaired.kernel
        assert binary.read_bytes() == repaired.kernel
        assert stale_name not in cache.get_group(metadata_name)
        warm = triton.compile(source)
        assert warm.kernel == repaired.kernel
        assert cache_hits == [False, False, True]

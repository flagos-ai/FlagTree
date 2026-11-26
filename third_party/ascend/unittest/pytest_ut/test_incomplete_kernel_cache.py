from pathlib import Path

import pytest

import triton
import triton.language as tl
from triton import knobs
from triton.compiler import ASTSource
from triton.runtime.cache import FileCacheManager


@pytest.mark.parametrize("damage", ["missing", "empty"])
def test_recompile_incomplete_kernel_cache(tmp_path, damage):

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

        compiled = triton.compile(source)
        binary = Path(compiled.metadata_group[f"{source.name}.npubin"])
        if damage == "missing":
            binary.unlink()
        else:
            binary.write_bytes(b"")

        repaired = triton.compile(source)
        assert repaired.kernel
        assert binary.read_bytes() == repaired.kernel

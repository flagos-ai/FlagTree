"""Explicit SM120 compiler regressions; these tests never initialize CUDA."""
import re

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language.gpu as gpu
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

E = tl.constexpr(gpu.BlockEncoding([1, 1], [1, 32], [4, 1], [1, 0]))


@triton.jit
def _planes(P, Y, EXTENT: tl.constexpr, GLU: tl.constexpr):
    w = gpu.set_layout(tl.broadcast_to(tl.arange(0, 4)[:, None], (4, 32)), E)
    l = gpu.set_layout(tl.broadcast_to(tl.arange(0, 32)[None, :], (4, 32)), E)
    s = gpu.alloc((EXTENT,), tl.float32, nv_mma_shared_layout=False, alignment_bytes=16)
    i = 32 * tl.maximum(w - 1, 0) + l
    v = gpu.set_layout(tl.load(P + 32 * w + l), E)
    tl.store(gpu.local_ptr(s, (i,), shape=(4, 32)), v, w > 0)
    if GLU:
        tl.store(gpu.local_ptr(s, (96 + i,), shape=(4, 32)), v + 1, w > 0)
    tl.debug_barrier()
    for j in tl.static_range(3):
        v += tl.load(gpu.local_ptr(s, (32 * j + l,), shape=(4, 32)), w == 0, 0)
        if GLU:
            v += tl.load(gpu.local_ptr(s, (96 + 32 * j + l,), shape=(4, 32)), w == 0, 0)
    tl.store(Y + 32 * w + l, v)


@triton.jit
def _alias(P, Y, OFFSET: tl.constexpr):
    s = gpu.alloc((192,), tl.float32, nv_mma_shared_layout=False, alignment_bytes=16)
    a = gpu.alloc((96,), tl.float32, alias=s, alias_offset_bytes=OFFSET, nv_mma_shared_layout=False)
    i = tl.arange(0, 32)
    p = gpu.local_ptr(a, (i,))
    tl.store(p, tl.load(P + i))
    tl.debug_barrier()
    tl.store(Y + i, tl.load(p))


@triton.jit(noinline=True)
def _stage_helper(s, P, Y):
    i = tl.arange(0, 32)
    ptr = gpu.local_ptr(s, (i,))
    tl.store(ptr, tl.load(P + i))
    tl.debug_barrier()
    tl.store(Y + i, tl.load(ptr))


@triton.jit
def _helper_calls(P, Y):
    s = gpu.alloc((192,), tl.float32, nv_mma_shared_layout=False, alignment_bytes=16)
    a = gpu.alloc((96,), tl.float32, alias=s, alias_offset_bytes=0, nv_mma_shared_layout=False)
    b = gpu.alloc((96,), tl.float32, alias=s, alias_offset_bytes=384, nv_mma_shared_layout=False)
    _stage_helper(a, P, Y)
    _stage_helper(b, P + 32, Y + 32)


def _compile(fn, signature, constants, static=True, **options):
    return triton.compile(ASTSource(fn, signature, constexprs=constants),
                          target=GPUTarget('cuda', 120, 32),
                          options=dict(num_warps=4, enable_static_shared=static, **options))


@pytest.mark.parametrize('extent,glu', [(96, False), (192, True), (128, False), (256, True)])
def test_flat_planes_exact_bytes_and_single_barrier(tmp_path, monkeypatch, extent, glu):
    monkeypatch.setenv('TRITON_CACHE_DIR', str(tmp_path / 'fresh'))
    c = _compile(_planes, dict(P='*fp32', Y='*fp32', EXTENT='constexpr', GLU='constexpr'),
                 dict(EXTENT=extent, GLU=glu))
    assert c.metadata.shared == 0
    assert c.metadata.static_shared == extent * 4
    assert len(re.findall(r'bar\.sync\s+0,\s*128;', c.asm['ptx'])) == 1
    assert c.asm['ptx'].count('bar.sync') == 1
    assert f'global [{extent * 4} x i8] undef, align 16' in c.asm['llir']
    assert c.asm['cubin']


@pytest.mark.parametrize('offset', [4, 384])
def test_alias_origin_and_alignment(tmp_path, monkeypatch, offset):
    monkeypatch.setenv('TRITON_CACHE_DIR', str(tmp_path / 'fresh'))
    c = _compile(_alias, dict(P='*fp32', Y='*fp32', OFFSET='constexpr'), dict(OFFSET=offset))
    assert c.metadata.static_shared == 768
    assert c.metadata.shared == 0
    if offset == 4:
        assert not re.search(r'(?:ld|st)\.shared\.v[24]', c.asm['ptx'])


def test_dynamic_default_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv('TRITON_CACHE_DIR', str(tmp_path / 'fresh'))
    c = _compile(_planes, dict(P='*fp32', Y='*fp32', EXTENT='constexpr', GLU='constexpr'),
                 dict(EXTENT=128, GLU=False), static=False)
    assert c.metadata.shared == 512
    assert c.metadata.static_shared == 0
    assert 'external addrspace(3) global [0 x i8]' in c.asm['llir']


def test_static_rejects_cluster_collective(tmp_path, monkeypatch):
    monkeypatch.setenv('TRITON_CACHE_DIR', str(tmp_path / 'fresh'))
    with pytest.raises(RuntimeError, match='PassManager::run failed'):
        _compile(_planes, dict(P='*fp32', Y='*fp32', EXTENT='constexpr', GLU='constexpr'),
                 dict(EXTENT=128, GLU=False), num_ctas=2)


def test_static_helper_shared_offset_abi(tmp_path, monkeypatch):
    monkeypatch.setenv('TRITON_CACHE_DIR', str(tmp_path / 'fresh'))
    c = _compile(_helper_calls, dict(P='*fp32', Y='*fp32'), {})
    assert c.metadata.shared == 0
    assert c.metadata.static_shared == 768
    assert '.func ' in c.asm['ptx']
    definitions = re.findall(r'^define internal .*_stage_helper.*', c.asm['llir'], re.M)
    assert definitions and all('ptr addrspace(3)' not in line and 'i32' in line for line in definitions)
    assert c.asm['cubin']

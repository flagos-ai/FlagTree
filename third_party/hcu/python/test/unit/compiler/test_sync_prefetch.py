"""Public HCU prefetch uses global and local carries only on eligible loops."""
import re
import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton import knobs
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def _mm(A, B, C, STAGES: tl.constexpr, AA: tl.constexpr, AB: tl.constexpr,
        VOLATILE: tl.constexpr):
    m = tl.program_id(0) * 64 + tl.arange(0, 64)
    n = tl.program_id(1) * 64 + tl.arange(0, 64)
    k = tl.arange(0, 64)
    acc = tl.zeros((64, 64), tl.float32)
    for i in tl.range(0, 5, num_stages=STAGES):
        ap = A + m[:, None] * 320 + (i * 64 + k)[None, :]
        bp = B + (i * 64 + k)[:, None] * 128 + n[None, :]
        if VOLATILE:
            a = tl.load(ap, volatile=True)
        elif AA is None:
            a = tl.load(ap)
        else:
            a = tle.load(ap, is_async=AA)
        if AB is None:
            b = tl.load(bp)
        else:
            b = tle.load(bp, is_async=AB)
        acc = tl.dot(a, b, acc)
    tl.store(C + m[:, None] * 128 + n[None, :], acc)


class _Captured(Exception):
    pass


@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
@pytest.mark.parametrize('stages,hint,aa,ab,volatile,local,global_,kpack', [
    (2, 'local-prefetch', False, False, False, 2, 0, 1),
    (3, 'local-prefetch', False, False, False, 2, 2, 1),
    (3, 'local-prefetch', None, None, False, 2, 2, 1),
    (3, 'local-prefetch', None, None, False, 2, 2, 2),
    (3, 'none', False, False, False, 0, 0, 1),
    (3, 'local-prefetch', True, True, False, 0, 0, 1),
    (3, 'local-prefetch', True, False, False, 0, 0, 1),
    (3, 'local-prefetch', False, True, False, 0, 0, 1),
    (3, 'local-prefetch', False, False, True, 0, 0, 1),
])
def test_public_sync_prefetch(dtype, stages, hint, aa, ab, volatile, local,
                              global_, kpack, monkeypatch, tmp_path):
    pytest.importorskip('triton.backends.hcu.compiler_hcu')
    from triton.compiler.compiler import make_backend
    instance = make_backend(GPUTarget('hip', 'gfx936', 64))
    captured = []

    def capture(module, metadata, options):
        captured.append(module.str())
        raise _Captured

    monkeypatch.setattr(knobs.cache, 'dir', str(tmp_path / 'cache'))
    monkeypatch.setattr(knobs.hcu, 'use_buffer_ops', True)
    monkeypatch.setattr(knobs.hcu, 'use_async_copy', False)
    monkeypatch.setattr(knobs.hcu, 'use_block_pingpong', False)
    monkeypatch.setattr(knobs.hcu, 'use_in_thread_transpose', False)
    monkeypatch.setattr(type(instance), 'make_llir', staticmethod(capture))
    source = ASTSource(
        _mm, signature={p: '*' + dtype for p in ['A', 'B', 'C']},
        constexprs={'STAGES': stages, 'AA': aa, 'AB': ab, 'VOLATILE': volatile},
        attrs={(i,): [('tt.divisibility', 16), ('tt.pointer_range', 32)]
               for i in range(3)},
    )
    with pytest.raises(_Captured):
        triton.compile(source, target=GPUTarget('hip', 'gfx936', 64),
                       options={'num_warps': 4, 'num_stages': stages,
                                'mmac_transposed': True, 'schedule_hint': hint, 'kpack': kpack})
    assert len(captured) == 1
    text = captured[0]
    aliases = dict(re.findall(r'^(#\w+)\s*=\s*([^\n]+)', text, re.M))
    loops = [line for line in text.splitlines() if 'scf.for ' in line]
    assert len(loops) == 1
    header = loops[0]
    actual_local = header.count('#ttg.dot_op<')
    for alias, encoding in aliases.items():
        if encoding.startswith('#ttg.dot_op<'):
            actual_local += len(re.findall(re.escape(alias) + r'(?=[,> ])', header))
    assert actual_local == local
    definitions = dict(re.findall(r'^\s*(%[\w.$]+)\s*=\s*([\w.]+)', text, re.M))
    initial = re.search(r'iter_args\((.*?)\)\s*->', header)[1]
    values = re.findall(r'%[\w.$]+\s*=\s*(%[\w.$]+)', initial)
    actual_global = sum(definitions.get(v) == 'hcug.buffer_load' for v in values)
    assert actual_global == global_

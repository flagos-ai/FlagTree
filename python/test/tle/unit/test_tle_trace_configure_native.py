# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Real frontend/pybind/native ring configuration through CPU compilation."""
import re
import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit(noinline=True)
def _trace_leaf(layer, original_tail):
    tle.gpu.trace_event(272, 11, layer)
    tl.store(original_tail, layer)
    tle.gpu.trace_event(273, 11, layer)


@triton.jit(noinline=True)
def _trace_consumer(data, original_tail):
    _trace_leaf(tl.full((), 17, tl.int32), original_tail)


@triton.jit(noinline=True)
def _trace_producer(data, original_tail):
    _trace_leaf(tl.full((), 29, tl.int32), original_tail + 1)


@triton.jit
def _trace_entry(data, output):
    tle.gpu.trace_configure(64, (0, 15, 64, 127))
    tle.gpu.trace_configure(64, (0, 15, 64, 127))
    tle.gpu.warp_specialize([
        (_trace_consumer, (data, output)), (_trace_producer, (data, output)),
    ], [1], [32])


def compile_cpu(kernel, constexprs=None):
    return triton.compile(ASTSource(kernel, {'data': '*i32', 'output': '*i32'},
                          constexprs=constexprs), target=GPUTarget('cuda', 90, 32),
                          options={'num_warps': 8, 'ptx_version': 87})


def test_trace_configure_normal_frontend_native():
    compiled = compile_cpu(_trace_entry)
    assert compiled.metadata.num_warps == 12
    assert compiled.metadata.profile_scratch_size == 12 * 65 * 64
    assert compiled.metadata.profile_scratch_align == 64
    for stage in ('ttir', 'ttgir'):
        text = compiled.asm[stage]
        assert 'tle.ws_trace.capacity = 64 : i32' in text
        assert 'tle.ws_trace.typed_ctas = array<i32: 0, 15, 64, 127>' in text
    llir, ptx = compiled.asm['llir'], compiled.asm['ptx']
    assert 'tle.trace_event' not in llir
    leaf_definitions = [line for line in llir.splitlines()
                        if line.startswith('define internal ') and '._trace_leaf' in line]
    assert len(leaf_definitions) == 2  # Separate real consumer/producer contexts.
    assert all(line.count('ptr addrspace(1)') == 2 for line in leaf_definitions)
    leaf_calls = [line for line in llir.splitlines() if 'call ' in line and '._trace_leaf' in line]
    assert len(leaf_calls) == 2
    assert all(line.count('ptr addrspace(1)') == 2 for line in leaf_calls)
    assert 'mov.u64 timestamp, %globaltimer' in ptx
    assert 'st.global.u32 [stream+4], 2' in ptx
    assert all(f'setp.eq.u64 match, cta, {cta}' in ptx for cta in (0, 15, 64, 127))
    assert ptx.index('mov.u64 timestamp, %globaltimer') < ptx.index('ld.global.u32 en, [stream+36]')


@triton.jit
def _trace_conflict(data, output, CASE: tl.constexpr):
    if CASE == 0:
        tle.gpu.trace_configure(64, (0,))
        tle.gpu.trace_configure(65, (0,))
    elif CASE == 1:
        tle.gpu.trace_configure(64, (0,))
        tle.gpu.trace_configure(64, (1,))
    elif CASE == 2:
        tle.gpu.trace_configure(64)
        tle.gpu.trace_configure(64, (0,))
    elif CASE == 3:
        tle.gpu.trace_configure(64, (0,))
        tle.gpu.trace_configure(64)
    elif CASE == 4:
        tle.gpu.census_configure(4, 128)
        tle.gpu.trace_configure(64, (0,))
    elif CASE == 5:
        tle.gpu.trace_configure(64, (0,))
        tle.gpu.census_configure(4, 128)
    elif CASE == 6:
        tle.gpu.trace_configure(64, (0,))
        tle.gpu.trace_event(272, 11, tl.full((), 17, tl.int32))
    elif CASE == 7:
        tle.gpu.trace_event(272, 11, tl.full((), 17, tl.int32))
    elif CASE == 8:
        tle.gpu.trace_configure(tl.load(data), (0,))
    tl.store(output, 1)


@pytest.mark.parametrize('case,message', [
    (0, 'conflicting.*trace'), (1, 'conflicting.*trace'),
    (2, 'conflicting.*trace'), (3, 'conflicting.*trace'),
    (4, 'census.*ring|ring.*census'), (5, 'census.*ring|ring.*census'),
    (6, 'enabled, configured WS profile-scratch ABI'),
    (7, 'enabled, configured WS profile-scratch ABI'), (8, 'capacity must be'),
])
def test_native_rejects_conflicts_or_missing_abi(case, message, capfd):
    with pytest.raises(Exception) as failure:
        compile_cpu(_trace_conflict, {'CASE': case})
    assert re.search(message, str(failure.value) + '\n' + capfd.readouterr().err)

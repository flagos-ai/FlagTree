# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Execute native packed C++ kernels on CPU, preserving their descriptor ABI."""
import ctypes
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import triton.flagmega.codegen.cpp as cpp_codegen
from triton.flagmega.codegen.cpp.templates import CppTemplateRegistry

FTT_INCLUDE = Path(cpp_codegen.__file__).parent / 'ftt/include'
LUT = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float32)
pytestmark = pytest.mark.skipif(shutil.which('c++') is None, reason='A host C++ compiler is required')


def bf16(value):
    value = np.asarray(value, np.float32)
    bits = value.view(np.uint32)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def f32(value):
    return (np.asarray(value, np.uint16).astype(np.uint32) << 16).view(np.float32)


def packed_rows(encoding, rows=4, blocks=2):
    """Independent packing oracle with distinct rows, scales and both nibbles."""
    rng = np.random.default_rng(613)
    size = 136 if encoding == 'IQ4_XS' else 144
    packed = np.zeros((rows, blocks, size), np.uint8)
    decoded = np.empty((rows, blocks, 256), np.float32)
    for row in range(rows):
        for block in range(blocks):
            data = packed[row, block]
            d = np.float16((row + block + 1) / 1024)
            data[:2] = np.frombuffer(d.tobytes(), np.uint8)
            codes = rng.integers(0, 16, (8, 32), dtype=np.uint8)
            if encoding == 'IQ4_XS':
                scales = np.array([-32, -17, -1, 0, 1, 15, 30, 31], np.int32)
                scales = np.roll(scales, row + block)
                unsigned = scales + 32
                high = sum(int(unsigned[g] >> 4) << (2*g) for g in range(8))
                data[2:4] = np.frombuffer(np.uint16(high).tobytes(), np.uint8)
                for g in range(8):
                    data[4 + g//2] |= np.uint8((unsigned[g] & 15) << (4*(g%2)))
                    data[8 + g*16:8 + (g+1)*16] = codes[g, :16] | (codes[g, 16:] << 4)
                decoded[row, block] = (np.float32(d) * scales[:, None] * LUT[codes]).reshape(-1)
            else:
                dmin = np.float16((row + 1) / 2048)
                data[2:4] = np.frombuffer(dmin.tobytes(), np.uint8)
                scales = rng.integers(0, 64, 8, dtype=np.uint8)
                mins = rng.integers(0, 64, 8, dtype=np.uint8)
                for g in range(4):
                    data[4+g] = scales[g] | ((scales[g+4] >> 4) << 6)
                    data[8+g] = mins[g] | ((mins[g+4] >> 4) << 6)
                    data[12+g] = (scales[g+4] & 15) | ((mins[g+4] & 15) << 4)
                for g in range(4):
                    data[16 + g*32:16 + (g+1)*32] = codes[2*g] | (codes[2*g+1] << 4)
                decoded[row, block] = (np.float32(d) * scales[:, None] * codes - np.float32(dmin) * mins[:, None]).reshape(-1)
    return packed.reshape(rows, -1), decoded.reshape(rows, -1)


def render(call, types, mesh=(2,)):
    # These hand-built host fixtures place disjoint feature shards explicitly;
    # normal descriptors instead derive the predicate from their physical ABI.
    if call['family'] == 'embedding' and 'outputs' not in call:
        call = dict(call, writer_active='True')
    return CppTemplateRegistry().render(f"kernels/{call['family']}/{call['variant']}.cpp.jinja", {
        'render_calls': [dict(call, symbol='kernel', arguments=list(types))],
        'argument_types': {name: {'type': typ, 'pointer': True} for name, typ in types.items()},
        'mesh_hierarchy': list(mesh), 'distributed_entry': True, 'block_threads': 4,
        'grid_barrier_axis_groups': [], 'grid_total_blocks': int(np.prod(mesh)),
    })


def library(tmp_path, body, types, mesh=(2,), phases=('kernel',), extra=''):
    body = re.sub(r'ftt::arch::mesh_shard_id<(\d+)>\([^)]*\)', r'test_shard[\1]', body)
    body = body.replace('ftt::arch::thread_id()', 'test_thread')
    parameters = ', '.join(f'{typ} *{name}' for name, typ in types.items())
    phase_calls = '\n'.join(f'''for (int shard = 0; shard < {int(np.prod(mesh))}; ++shard) {{
        {' '.join(f'test_shard[{i}] = (shard / {int(np.prod(mesh[i+1:]))}) % {m};' for i,m in enumerate(mesh))}
        for (test_thread = 0; test_thread < 4; ++test_thread) {phase}({', '.join(types)});
    }}''' for phase in phases)
    source = f'''#include <cmath>
#include <cstdint>
#include <utility>
struct float4 {{ float x,y,z,w; }};
#include <ftt/ftt.h>
static ftt::dim_t test_shard[{len(mesh)}];
static ftt::dim_t test_thread;
static unsigned int _ftt_grid_barrier_counters[2];
{body}
extern "C" void run({parameters}) {{ {phase_calls} }}
{extra}
'''
    cpp = tmp_path / 'native.cpp'
    cpp.write_text(source)
    so = tmp_path / 'native.so'
    subprocess.run(['c++', '-std=c++17', '-O1', '-shared', '-fPIC', '-DFTT_ARCH_ILUVATAR', f'-I{FTT_INCLUDE}', str(cpp), '-o', str(so)], check=True, capture_output=True, text=True)
    return ctypes.CDLL(str(so))


def run(lib, *arrays):
    lib.run(*(a.ctypes.data_as(ctypes.c_void_p) for a in arrays))


@pytest.mark.parametrize('encoding', ['IQ4_XS', 'Q4_K'])
def test_native_embedding_decodes_all_groups_and_batched_ids(tmp_path, encoding):
    packed, decoded = packed_rows(encoding)
    types = {'indices': 'int', 'weight': 'unsigned char', 'result': 'ftt::bfloat16_t'}
    call = dict(family='embedding', variant='decode', indices='indices', weight='weight', result='result',
                rows=3, row_active='embedding_row < 3', indices_offset='embedding_row',
                vocab_size=4, padding_idx=0, local_capacity=256, active='256',
                weight_encoding=encoding, weight_hidden=512, weight_block_bytes=packed.shape[1]//2, global_feature='shard_index * 256 + local_offsets',
                weight_offset='token_id * 512 + shard_index * 256 + local_offsets',
                result_offset='embedding_row * 512 + shard_index * 256 + local_offsets')
    lib = library(tmp_path, render(call, types), types)
    ids = np.array([3, 0, 1], np.int32)
    actual = np.full((3, 512), 0x7fc0, np.uint16)
    run(lib, ids, packed, actual)
    expected = bf16(decoded[ids]); expected[1] = 0
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('encoding', ['IQ4_XS', 'Q4_K'])
@pytest.mark.parametrize('glu', [False, True])
def test_native_projections_use_global_rows_and_output_dtype(tmp_path, encoding, glu):
    packed, decoded = packed_rows(encoding, rows=6)
    up = packed[::-1].copy(); up_decoded = decoded[::-1].copy()
    source = bf16(np.random.default_rng(72).normal(0, 0.04, 512))
    types = {'value': 'ftt::bfloat16_t', 'weight': 'unsigned char'}
    call = dict(family='block_fp8', variant='simt', value='value', weight='weight', result='result',
                local_n_capacity=4, active_n='3', n_mask='local_n_offsets < 3', k=512,
                weight_encoding=encoding, weight_row_offset=f'(shard_index * 3 + local_n_offsets) * {packed.shape[1]}',
                destination_offset='shard_index * 3 + local_n_offsets')
    if glu:
        call.update(family='matmul_glu', gate_weight='weight', up_weight='up',
                    up_weight_row_offset=call['weight_row_offset'], gate_clamp_max='None', up_clamp='None')
        types['up'] = 'unsigned char'
    types['result'] = 'ftt::bfloat16_t' if glu else 'float'
    lib = library(tmp_path, render(call, types), types)
    actual = np.full(6, 0x7fc0, np.uint16) if glu else np.full(6, np.nan, np.float32)
    run(lib, source, packed, *((up,) if glu else ()), actual)
    expected = decoded @ f32(source)
    if glu:
        expected = expected / (1 + np.exp(-expected)) * (up_decoded @ f32(source))
        np.testing.assert_array_equal(actual, bf16(expected))
    else:
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def test_quantized_half_decode_preserves_subnormals_and_special_values(tmp_path):
    body = '#include <ftt/quantized_ops.h>'
    types = {}
    extra = '''extern "C" void decode_half(const unsigned char *source, float *output) {
        for (int i=0;i<65536;++i) output[i] = ftt::quantized::load_f16(source + 2*i);
    }'''
    lib = library(tmp_path, body, types, mesh=(1,), phases=(), extra=extra)
    source = np.arange(65536, dtype=np.uint16)
    actual = np.zeros(65536, np.float32)
    lib.decode_half(source.ctypes.data_as(ctypes.c_void_p), actual.ctypes.data_as(ctypes.c_void_p))
    expected = source.view(np.float16).astype(np.float32)
    np.testing.assert_array_equal(actual, expected)


def test_ordinary_embedding_preserves_batched_row_and_local_table_addresses(tmp_path):
    types = {'indices':'int', 'weight':'ftt::bfloat16_t', 'result':'ftt::bfloat16_t'}
    call = dict(family='embedding', variant='decode', indices='indices', weight='weight', result='result',
                rows=3, row_active='embedding_row < 3', indices_offset='embedding_row',
                vocab_size=4, padding_idx=0, local_capacity=8, active='7', weight_encoding=None,
                weight_offset='shard_index * 28 + token_id * 7 + local_offsets',
                result_offset='embedding_row * 14 + shard_index * 7 + local_offsets')
    lib = library(tmp_path, render(call,types), types)
    ids = np.array([3,0,1],np.int32)
    canonical = bf16(np.arange(56,dtype=np.float32).reshape(4,14)/16)
    weight = np.stack([canonical[:,:7],canonical[:,7:]])
    result = np.full((3,14),0x7fc0,np.uint16)
    run(lib,ids,weight,result)
    expected = canonical[ids].copy(); expected[1] = 0
    np.testing.assert_array_equal(result,expected)


@pytest.mark.parametrize('encoding',['IQ4_XS','Q4_K'])
def test_native_linear_rounds_at_bf16_destination_only(tmp_path,encoding):
    packed, decoded = packed_rows(encoding,rows=6)
    source = bf16(np.random.default_rng(915).normal(0,0.04,512))
    types = {'value':'ftt::bfloat16_t','weight':'unsigned char','result':'ftt::bfloat16_t'}
    call = dict(family='block_fp8', variant='simt', value='value', weight='weight', result='result',
                local_n_capacity=3, active_n='3', n_mask='local_n_offsets < 3', k=512,
                weight_encoding=encoding, weight_row_offset=f'(shard_index*3+local_n_offsets)*{packed.shape[1]}',
                destination_offset='shard_index*3+local_n_offsets')
    lib = library(tmp_path,render(call,types),types)
    result = np.zeros(6,np.uint16)
    run(lib,source,packed,result)
    np.testing.assert_array_equal(result,bf16(decoded @ f32(source)))

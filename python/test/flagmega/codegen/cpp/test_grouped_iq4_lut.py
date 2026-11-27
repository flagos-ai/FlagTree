# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Exact software semantics of the grouped NVIDIA helper, without a GPU."""
import ctypes
import importlib.util
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location('ordered_dot_helpers', Path(__file__).with_name('test_ordered_quantized_dot.py'))
_ordered = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ordered)
FTT_INCLUDE, HOST_PREFIX, K_VALUES, TYPES = _ordered.FTT_INCLUDE, _ordered.HOST_PREFIX, _ordered.K_VALUES, _ordered.TYPES
equal_bits_or_nan, pack, ptr, source = _ordered.equal_bits_or_nan, _ordered.pack, _ordered.ptr, _ordered.source


GROUPED = r'''
template<class T>
float software_grouped(const T *x,const unsigned char *row,long long k) {
    float a=0;long long i=0,complete=k>0?k-k%256:0;
    for(;i<complete;i+=256) {
        const unsigned char *block=row+i/256*136;
        float d=ftt::quantized::load_f16(block);
        unsigned int high=ftt::quantized::load_u16(block+2),low=ftt::quantized::detail::load_u32(block+4);
        for(int g=0;g<8;++g) {
            int scale=static_cast<int>(((low>>(4*g))&15)|(((high>>(2*g))&3)<<4))-32;
            float factor=d*static_cast<float>(scale);const unsigned char *p=block+8+g*16;
            unsigned int w0=ftt::quantized::detail::load_u32(p),w1=ftt::quantized::detail::load_u32(p+4),
                         w2=ftt::quantized::detail::load_u32(p+8),w3=ftt::quantized::detail::load_u32(p+12);
            a=ftt::quantized::detail::iq4_nibble_dot_grouped<0>(x+i+g*32,w0,w1,w2,w3,factor,a);
            a=ftt::quantized::detail::iq4_nibble_dot_grouped<4>(x+i+g*32+16,w0,w1,w2,w3,factor,a);
        }
    }
    for(;i<k;++i)a+=ftt::quantized::decode<encoding::iq4_xs>(row,i)*ftt::to_float(x[i]);
    return a;
}
extern "C" unsigned int permutation(unsigned int a,unsigned int b,unsigned int s) {
    return ftt::arch::byte_permute_default(a,b,s);
}
extern "C" long long exhaustive_lut() {
    const int table[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    for(unsigned c=0;c<65536;++c)for(int nibble=0;nibble<2;++nibble) {
        unsigned p=0,expected=0;
        for(int i=0;i<4;++i) {
            unsigned q=(c>>(4*i))&15,other=15-q;
            p|=(nibble?(q<<4)|other:q|(other<<4))<<(8*i);
            expected|=static_cast<unsigned>(table[q]&255)<<(8*i);
        }
        unsigned levels=nibble?ftt::quantized::detail::iq4_lut4<4>(p):ftt::quantized::detail::iq4_lut4<0>(p);
        if(levels!=expected)return -1;
    }
    for(unsigned b=0;b<256;++b) {
        unsigned w=b*0x01010101u;int e=static_cast<int>(b)-static_cast<int>((b&128)<<1);
        if(ftt::arch::signed_byte_extract<0>(w)!=e||ftt::arch::signed_byte_extract<1>(w)!=e
            ||ftt::arch::signed_byte_extract<2>(w)!=e||ftt::arch::signed_byte_extract<3>(w)!=e)return -2;
    }
    return 131072;
}
extern "C" float grouped_empty(long long k) {
    return software_grouped(static_cast<const float*>(nullptr),nullptr,k);
}
extern "C" long long grouped_observed(const unsigned char *w,long long k) {
    observed x[1025];for(long long i=0;i<k;++i)x[i]={float(i%7)-3.0f,i};
    observed::next=0;volatile float a=software_grouped(x,w,k);(void)a;return observed::next;
}
extern "C" float grouped_guarded(const unsigned char *row,long long nbytes,long long k) {
    const long page=sysconf(_SC_PAGESIZE);
    auto *wr=static_cast<unsigned char*>(mmap(nullptr,page*2,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));
    auto *xr=static_cast<unsigned char*>(mmap(nullptr,page*3,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));
    if(wr==MAP_FAILED||xr==MAP_FAILED)std::abort();
    mprotect(wr+page,page,PROT_NONE);mprotect(xr+page*2,page,PROT_NONE);
    auto *w=wr+page-nbytes;std::memcpy(w,row,nbytes);float*x=reinterpret_cast<float*>(xr+page*2)-k;
    for(long long i=0;i<k;++i)x[i]=1;
    float a=software_grouped(x,w,k);munmap(wr,page*2);munmap(xr,page*3);return a;
}
'''


@pytest.fixture(scope='module')
def grouped(tmp_path_factory):
    if shutil.which('c++') is None:
        pytest.skip('host C++ compiler unavailable')
    if not Path('/usr/local/cuda/include/cuda_bf16.h').exists():
        pytest.skip('NVIDIA host dtype headers unavailable')
    path = tmp_path_factory.mktemp('grouped-iq4')
    wrappers = '\n'.join(
        f'extern "C" float grouped_{name}(const {typ}*x,const unsigned char*w,long long k){{return software_grouped(x,w,k);}}\n'
        f'extern "C" float reference_{name}(const {typ}*x,const float*w,long long k){{return reference(x,w,k);}}'
        for name, typ in TYPES.items()
    )
    cpp = path / 'grouped.cpp'
    cpp.write_text('#include <utility>\n' + HOST_PREFIX + GROUPED + wrappers)
    binary = path / 'grouped.so'
    command = ['c++', '-std=c++20', '-O2', '-ffp-contract=off', '-fno-fast-math',
               '-shared', '-fPIC', '-DFTT_ARCH_NVIDIA', f'-I{FTT_INCLUDE}',
               '-I/usr/local/cuda/include', '-I/usr/local/cuda/include/cccl',
               str(cpp), '-o', str(binary)]
    result = subprocess.run(command, capture_output=True, text=True)
    (path / 'compile.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stderr
    lib = ctypes.CDLL(str(binary))
    lib.exhaustive_lut.restype = lib.grouped_observed.restype = ctypes.c_longlong
    lib.permutation.restype = ctypes.c_uint
    lib.permutation.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_uint]
    for name in ('grouped_empty', 'grouped_guarded', *(f'grouped_{n}' for n in TYPES), *(f'reference_{n}' for n in TYPES)):
        getattr(lib, name).restype = ctypes.c_float
    return lib


def test_all_four_nibble_tuples_signs_and_intrinsic_negative_witness(grouped):
    assert grouped.exhaustive_lut() == 131072
    assert grouped.permutation(0x71dd0181, 0, 0x9991) == 1
    assert grouped.permutation(0x71dd0181, 0, 0x9991 & 0x7777) == 0x01010101


def test_grouped_matches_independent_weights_for_all_source_types(grouped):
    row, decoded = pack('iq4_xs')
    for offset in (0, 1, 2, 3, 7):
        storage = np.zeros(row.size + offset, np.uint8)
        storage[offset:] = row
        for name in TYPES:
            x = source(name, decoded.size)
            for k in K_VALUES:
                actual = getattr(grouped, 'grouped_' + name)(ptr(x), ptr(storage[offset:]), ctypes.c_longlong(k))
                expected = getattr(grouped, 'reference_' + name)(ptr(x), ptr(decoded), ctypes.c_longlong(k))
                equal_bits_or_nan(actual, expected)


def test_grouped_source_order_positive_zero_and_minimal_tail_reads(grouped):
    for k in (-2**63, -1, 0):
        equal_bits_or_nan(grouped.grouped_empty(ctypes.c_longlong(k)), 0.)
    row, decoded = pack('iq4_xs')
    for k in K_VALUES[3:]:
        assert grouped.grouped_observed(ptr(row), ctypes.c_longlong(k)) == k
        blocks, tail = divmod(k, 256)
        if tail:
            g, n = (tail-1)//32, tail-((tail-1)//32)*32
            nbytes = blocks*136 + 8 + g*16 + min(16, n)
        else:
            nbytes = blocks*136
        actual = grouped.grouped_guarded(ptr(row), ctypes.c_longlong(nbytes), ctypes.c_longlong(k))
        expected = grouped.reference_f32(ptr(np.ones(k, np.float32)), ptr(decoded), ctypes.c_longlong(k))
        equal_bits_or_nan(actual, expected)


def test_grouped_keeps_all_low_quads_before_high_quads(grouped):
    row = np.full(136, 0x88, np.uint8)
    row[:2] = np.frombuffer(np.uint16(0x3c00).tobytes(), np.uint8)
    row[2:4] = 0xaa
    row[4:8] = 0x11
    x = np.zeros(256, np.uint16)
    x[0], x[4], x[16] = 0x60ad, 0xe0ad, 0x4040
    equal_bits_or_nan(grouped.grouped_bf16(ptr(x), ptr(row), ctypes.c_longlong(256)), 3.)
    # Wrong word0-low/high interleaving would insert 3 before cancellation.
    wrong = np.float32(np.float32(np.float32(9.972771014849226e19)+np.float32(3.))
                       + np.float32(-9.972771014849226e19))
    assert wrong == 0.


@pytest.mark.parametrize('half', [0, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e01])
def test_grouped_half_boundaries_and_old_scalar_conversions(grouped, half):
    row, decoded = pack('iq4_xs', half_bits=half)
    for name in ('f32', 'f64', 'f16', 'bf16'):
        x = source(name, decoded.size)
        for k in (255, 256, 257, 511, 512):
            equal_bits_or_nan(getattr(grouped, 'grouped_' + name)(ptr(x), ptr(row), ctypes.c_longlong(k)),
                              getattr(grouped, 'reference_' + name)(ptr(x), ptr(decoded), ctypes.c_longlong(k)))

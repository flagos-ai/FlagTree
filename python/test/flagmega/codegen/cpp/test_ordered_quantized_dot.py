# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Ordered native dot contract. Host tests disable contraction; NVCC is compile-only."""
import ctypes
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import triton.flagmega.codegen.cpp as cpp_codegen

FTT_INCLUDE = Path(cpp_codegen.__file__).parent / 'ftt/include'
LUT = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float32)
TYPES = {'f32': 'float', 'f64': 'double', 'f16': '_Float16', 'bf16': 'ftt::bfloat16_t',
         'i8': 'signed char', 'u8': 'unsigned char', 'i16': 'short', 'u16': 'unsigned short',
         'i32': 'int', 'u32': 'unsigned int', 'i64': 'long long', 'u64': 'unsigned long long', 'bool': 'bool'}
K_VALUES = [-2**63, -1, 0, 1, 15, 16, 17, 31, 32, 33, 127, 255, 256, 257, 511, 512, 513, 767, 768, 769, 1025]


def pack(format, blocks=5, half_bits=None):
    """Byte fixture and independent GGML equations; no production decode helper."""
    rng = np.random.default_rng(50005)
    width = 136 if format == 'iq4_xs' else 144
    row = rng.integers(0, 256, (blocks, width), dtype=np.uint8)
    values = []
    for b, data in enumerate(row):
        half = (0x3c01, 0x0001, 0x7bff, 0xbc01, 0x8000)[b % 5] if half_bits is None else half_bits
        data[:2] = np.frombuffer(np.uint16(half).tobytes(), np.uint8)
        d = np.array([half], np.uint16).view(np.float16).astype(np.float32)[0]
        if format == 'q4_k':
            data[2:4] = np.frombuffer(np.uint16(0x0801).tobytes(), np.uint8)
        with np.errstate(all='ignore'):
            for g in range(8):
                if format == 'iq4_xs':
                    lo = (int(data[4 + g//2]) >> (4*(g%2))) & 15
                    hi = ((int(data[2]) | int(data[3]) << 8) >> (2*g)) & 3
                    factor = np.float32(d * np.float32((lo | hi << 4) - 32))
                    code = np.concatenate([data[8+g*16:24+g*16] & 15, data[8+g*16:24+g*16] >> 4])
                    values.extend(factor * LUT[code])
                else:
                    a, bscale, c = (int(data[4+g%4]), int(data[8+g%4]), int(data[12+g%4]))
                    scale = a & 63 if g < 4 else (c & 15) | (a >> 2) & 48
                    minimum = bscale & 63 if g < 4 else (c >> 4) | (bscale >> 2) & 48
                    code = (data[16+(g//2)*32:48+(g//2)*32] >> (4*(g%2))) & 15
                    dmin = np.array([0x0801], np.uint16).view(np.float16).astype(np.float32)[0]
                    values.extend(np.float32(d * np.float32(scale)) * code.astype(np.float32) - np.float32(dmin * np.float32(minimum)))
    return row.reshape(-1), np.asarray(values, np.float32)


HOST_PREFIX = '''#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
#if defined(FTT_ARCH_ILUVATAR)
struct float4 { float x,y,z,w; };
#endif
#include <ftt/quantized_ops.h>
using ftt::quantized::encoding;
template<class T> float reference(const T *x, const float *w, long long k) {
    float a = 0.0f;
    for (long long i=0; i<k; ++i) a += w[i] * ftt::to_float(x[i]);
    return a;
}
struct observed {
    float value; long long index;
    static long long next;
    operator float() const { if (index != next++) std::abort(); return value; }
};
long long observed::next;
extern "C" long long source_order(const unsigned char *w, long long k) {
    observed x[1025];
    for (long long i=0; i<k; ++i) x[i] = {float(i%7)-3.0f,i};
    observed::next=0;
    volatile float a=ftt::quantized::dot<encoding::iq4_xs>(x,w,k);
    (void)a;
    return observed::next;
}
extern "C" float empty(long long k) {
    return ftt::quantized::dot<encoding::iq4_xs>(static_cast<const float*>(nullptr),nullptr,k);
}
extern "C" void decode_iq4(const unsigned char *w, float *out, long long k) {
    for (long long i=0;i<k;++i) out[i]=ftt::quantized::decode<encoding::iq4_xs>(w,i);
}
extern "C" float fused_reference(const float *x, const float *w, long long k) {
    float a=0;
    for(long long i=0;i<k;++i) a=std::fma(w[i],x[i],a);
    return a;
}
extern "C" float guarded(const unsigned char *row, long long nbytes, long long k) {
    const long page=sysconf(_SC_PAGESIZE);
    auto *wr=static_cast<unsigned char*>(mmap(nullptr,page*2,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));
    auto *xr=static_cast<unsigned char*>(mmap(nullptr,page*3,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));
    if(wr==MAP_FAILED || xr==MAP_FAILED) std::abort();
    mprotect(wr+page,page,PROT_NONE); mprotect(xr+page*2,page,PROT_NONE);
    unsigned char *w=wr+page-nbytes; std::memcpy(w,row,nbytes);
    float *x=reinterpret_cast<float*>(xr+page*2)-k;
    for(long long i=0;i<k;++i) x[i]=1.0f;
    float a=ftt::quantized::dot<encoding::iq4_xs>(x,w,k);
    munmap(wr,page*2);munmap(xr,page*3);return a;
}
'''


@pytest.fixture(scope='module', params=['ILUVATAR', 'NVIDIA'])
def host(request, tmp_path_factory):
    if shutil.which('c++') is None:
        pytest.skip('host C++ compiler unavailable')
    arch = request.param
    if arch == 'NVIDIA' and not Path('/usr/local/cuda/include/cuda_bf16.h').exists():
        pytest.skip('NVIDIA host dtype headers unavailable')
    path = tmp_path_factory.mktemp('ordered-dot-' + arch)
    wrappers = '\n'.join(f'extern "C" float reference_{n}(const {t} *x,const float*w,long long k) {{return reference(x,w,k);}}' for n,t in TYPES.items())
    wrappers += '\n' + '\n'.join(f'extern "C" float {fmt}_{n}(const {t} *x,const unsigned char*w,long long k) {{return ftt::quantized::dot<encoding::{fmt}>(x,w,k);}}' for fmt in ['iq4_xs','q4_k'] for n,t in TYPES.items())
    source=path/'test.cpp'; source.write_text(HOST_PREFIX+wrappers)
    target=path/'test.so'
    command=['c++','-std=c++17','-O2','-ffp-contract=off','-fno-fast-math','-shared','-fPIC',f'-DFTT_ARCH_{arch}',f'-I{FTT_INCLUDE}']
    if arch=='NVIDIA': command += ['-I/usr/local/cuda/include','-I/usr/local/cuda/include/cccl']
    command += [str(source),'-o',str(target)]
    p=subprocess.run(command,capture_output=True,text=True)
    (path/'compile.log').write_text(p.stdout+p.stderr)
    assert p.returncode==0,p.stderr
    lib=ctypes.CDLL(str(target))
    for name in ['empty','guarded','fused_reference',*(f'{fmt}_{n}' for fmt in ['iq4_xs','q4_k'] for n in TYPES),*(f'reference_{n}' for n in TYPES)]:
        getattr(lib,name).restype=ctypes.c_float
    lib.source_order.restype=ctypes.c_longlong
    return lib


def ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def equal_bits_or_nan(a,b):
    if math.isnan(a) and math.isnan(b): return
    assert np.float32(a).view(np.uint32)==np.float32(b).view(np.uint32),(a,b)


def source(name,k):
    values=np.resize(np.array([0.,-0.,1.,-1.,1.0078125,2**-20,32.25,-64.5],np.float32),k)
    if name=='bf16':
        bits=values.view(np.uint32); b=((bits+0x7fff+((bits>>16)&1))>>16).astype(np.uint16)
        return b
    if name=='bool':return values>0
    dtype={'f32':np.float32,'f64':np.float64,'f16':np.float16,'i8':np.int8,'u8':np.uint8,'i16':np.int16,'u16':np.uint16,'i32':np.int32,'u32':np.uint32,'i64':np.int64,'u64':np.uint64}[name]
    if name.startswith('u'): values=np.abs(values)
    return values.astype(dtype)


@pytest.mark.parametrize('format',['iq4_xs','q4_k'])
def test_all_sources_tails_unaligned_and_decode_neighbor(host,format):
    row,decoded=pack(format)
    for offset in [0,1,2,3,7]:
        storage=np.zeros(row.size+offset,np.uint8);storage[offset:]=row;w=storage[offset:]
        for name in TYPES:
            x=source(name,decoded.size)
            for k in K_VALUES:
                a=getattr(host,format+'_'+name)(ptr(x),ptr(w),ctypes.c_longlong(k))
                b=getattr(host,'reference_'+name)(ptr(x),ptr(decoded),ctypes.c_longlong(k))
                equal_bits_or_nan(a,b)
    if format=='iq4_xs':
        out=np.empty_like(decoded);host.decode_iq4(ptr(row),ptr(out),ctypes.c_longlong(decoded.size))
        np.testing.assert_array_equal(out.view(np.uint32),decoded.view(np.uint32))


def test_zero_negative_source_order_and_minimal_tail_extent(host):
    for k in [-2**63,-1,0]: equal_bits_or_nan(host.empty(ctypes.c_longlong(k)),0.)
    row,decoded=pack('iq4_xs')
    for k in K_VALUES[3:]:
        assert host.source_order(ptr(row),ctypes.c_longlong(k))==k
        blocks,tail=divmod(k,256)
        if tail:
            g=(tail-1)//32;n=tail-g*32
            nbytes=blocks*136+8+g*16+min(16,n)
        else:nbytes=blocks*136
        a=host.guarded(ptr(row),ctypes.c_longlong(nbytes),ctypes.c_longlong(k))
        x=np.ones(k,np.float32)
        b=host.reference_f32(ptr(x),ptr(decoded),ctypes.c_longlong(k))
        equal_bits_or_nan(a,b)


@pytest.mark.parametrize('half',[0x0000,0x8000,0x0001,0x8001,0x03ff,0x0400,0x7bff,0xfbff,0x7c00,0xfc00,0x7e01,0xfe01,0x7c01])
def test_half_specials_and_source_specials(host,half):
    row,decoded=pack('iq4_xs',half_bits=half)
    for kind in ['f32','f64','f16','bf16']:
        x=source(kind,decoded.size)
        for k in [1,16,17,31,32,255,256,257,511,512]:
            a=getattr(host,'iq4_xs_'+kind)(ptr(x),ptr(row),ctypes.c_longlong(k))
            b=getattr(host,'reference_'+kind)(ptr(x),ptr(decoded),ctypes.c_longlong(k))
            equal_bits_or_nan(a,b)
    row,decoded=pack('iq4_xs')
    for special in [0.,-0.,np.nextafter(np.float32(0),np.float32(1)),np.inf,-np.inf,np.nan]:
        for index in [0,15,16,31,255,256,511]:
            x=np.ones(decoded.size,np.float32);x[index]=special
            equal_bits_or_nan(host.iq4_xs_f32(ptr(x),ptr(row),ctypes.c_longlong(512)),host.reference_f32(ptr(x),ptr(decoded),ctypes.c_longlong(512)))


def test_oracle_detects_order_fma_and_premature_weight_rounding(host):
    row,decoded=pack('iq4_xs',blocks=2,half_bits=0x3c01)
    rng=np.random.default_rng(8931)
    x=(rng.normal(0,1,512)*np.geomspace(2**-20,2**20,512)).astype(np.float32)
    ascending=host.reference_f32(ptr(x),ptr(decoded),ctypes.c_longlong(512))
    reordered=decoded.reshape(-1,32)[:,np.r_[16:32,0:16]].reshape(-1).copy()
    assert ascending!=host.reference_f32(ptr(x),ptr(reordered),ctypes.c_longlong(512))
    # A deterministic cancellation witness exposes FMA contraction.
    wx=np.array([1.,1.0078125],np.float32)
    ww=np.array([-4099.75,4067.96875],np.float32)
    assert host.reference_f32(ptr(wx),ptr(ww),ctypes.c_longlong(2))==0.
    assert host.fused_reference(ptr(wx),ptr(ww),ctypes.c_longlong(2))==-0.000244140625
    bits=decoded.view(np.uint32);early=(((bits+0x7fff+((bits>>16)&1))>>16)<<16).view(np.float32)
    assert ascending!=host.reference_f32(ptr(x),ptr(early),ctypes.c_longlong(512))
    equal_bits_or_nan(host.iq4_xs_f32(ptr(x),ptr(row),ctypes.c_longlong(512)),ascending)


def test_every_signed_scale_and_lut_code_in_each_group(host):
    for scale in range(-32,32):
        u=scale+32
        row=np.zeros(136,np.uint8)
        row[:2]=np.frombuffer(np.uint16(0x3c01).tobytes(),np.uint8)
        hi=sum((u>>4)<<(2*g) for g in range(8))
        row[2:4]=np.frombuffer(np.uint16(hi).tobytes(),np.uint8)
        row[4:8]=(u&15)*17
        for g in range(8):row[8+g*16:24+g*16]=np.arange(16,dtype=np.uint8)*17
        factor=np.float32(np.float32(1.0009765625)*np.float32(scale))
        for group in range(8):
            for lane in range(32):
                x=np.zeros(256,np.float32);x[group*32+lane]=1.
                expected=np.float32(0.)+np.float32(factor*LUT[lane%16])
                equal_bits_or_nan(host.iq4_xs_f32(ptr(x),ptr(row),ctypes.c_longlong(256)),expected)


@pytest.mark.parametrize('k',[5120,6144,17408])
def test_long_native_rows_preserve_single_ascending_accumulator(host,k):
    row,decoded=pack('iq4_xs',blocks=k//256)
    rng=np.random.default_rng(9031)
    for kind in ['f32','bf16']:
        x=source(kind,k)
        if kind=='f32': x=(rng.normal(0,1,k)*np.geomspace(2**-10,2**10,k)).astype(np.float32)
        equal_bits_or_nan(getattr(host,'iq4_xs_'+kind)(ptr(x),ptr(row),ctypes.c_longlong(k)),getattr(host,'reference_'+kind)(ptr(x),ptr(decoded),ctypes.c_longlong(k)))


def test_source_conversion_extremes_and_nonfinite_values(host):
    row,decoded=pack('iq4_xs',blocks=2)
    for kind in TYPES:
        x=source(kind,512)
        if kind=='bf16':
            values=np.array([0,0x8000,1,0x8001,0x7f7f,0xff7f,0x7f80,0xff80,0x7fc1,0x7f81],np.uint16)
        elif kind=='bool':values=np.array([True,False])
        elif kind.startswith(('i','u')):
            bounds=np.iinfo(x.dtype)
            values=np.array([bounds.min,bounds.min+1,0,1,bounds.max-1,bounds.max],dtype=x.dtype)
        else:
            bounds=np.finfo(x.dtype)
            values=np.array([0.,-0.,bounds.smallest_subnormal,-bounds.smallest_subnormal,bounds.max,-bounds.max,np.inf,-np.inf,np.nan],dtype=x.dtype)
        x=np.resize(values,512)
        for k in [255,256,257,511,512]:
            equal_bits_or_nan(getattr(host,'iq4_xs_'+kind)(ptr(x),ptr(row),ctypes.c_longlong(k)),getattr(host,'reference_'+kind)(ptr(x),ptr(decoded),ctypes.c_longlong(k)))

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Host byte/boundary/numeric witnesses; these do not execute CUDA."""
import ast
import ctypes
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[5]
FTT = REPO / 'python/triton/flagmega/codegen/cpp/ftt/include'
LOCAL = REPO / 'python/tutorials/flagmega/03-qwen3.8-27b-iq4/nvidia-rtx5060ti-cuda/local_optimizations'
TYPES = {'f32': 'float', 'f64': 'double', 'f16': '_Float16', 'bf16': 'ftt::bfloat16_t', 'i64': 'long long'}
KS = (-2**63, -1, 0, 1, 15, 16, 17, 31, 32, 33, 255, 256, 257, 511, 512, 513, 767, 768)

SOURCE = r'''
#include <cstdint>
#include <utility>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
#if defined(FTT_ARCH_ILUVATAR)
struct float4 { float x,y,z,w; };
#endif
#include <ftt/quantized_ops.h>
using ftt::quantized::encoding;
extern "C" unsigned int reader_word(const unsigned char *row,long long offset) {
    const ftt::quantized::detail::iq4_row_reader reader(row);
    if(reader.uses_global_words())std::abort();
    return reader.word(offset);
}
extern "C" float empty(long long k) {
    return ftt::quantized::dot<encoding::iq4_xs>(static_cast<const float*>(nullptr),nullptr,k);
}
extern "C" float guarded(const unsigned char *src,long long nbytes,long long k) {
    const long page=sysconf(_SC_PAGESIZE);
    auto*w=static_cast<unsigned char*>(mmap(nullptr,page*2,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));
    auto*x=static_cast<unsigned char*>(mmap(nullptr,page*3,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));
    if(w==MAP_FAILED||x==MAP_FAILED)std::abort();
    mprotect(w+page,page,PROT_NONE);mprotect(x+page*2,page,PROT_NONE);
    auto*row=w+page-nbytes;std::memcpy(row,src,nbytes);auto*values=reinterpret_cast<float*>(x+page*2)-k;
    for(long long i=0;i<k;++i)values[i]=1.0f;
    float a=ftt::quantized::dot<encoding::iq4_xs>(values,row,k);
    munmap(w,page*2);munmap(x,page*3);return a;
}
struct observed {float value;long long index;static long long next;
 operator float()const{if(index!=next++)std::abort();return value;}};
long long observed::next;
extern "C" long long order(const unsigned char*w,long long k) {
 observed x[1025];for(long long i=0;i<k;++i)x[i]={float(i%7)-3,i};observed::next=0;
 volatile float a=ftt::quantized::dot<encoding::iq4_xs>(x,w,k);(void)a;return observed::next;
}
'''


def equal_bits(a, b):
    if math.isnan(a) and math.isnan(b):
        return
    assert np.float32(a).view(np.uint32) == np.float32(b).view(np.uint32), (a, b)


def ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def weights(fmt='iq4_xs', half=0x3c01):
    width = 136 if fmt == 'iq4_xs' else 144
    row = np.arange(3*width, dtype=np.uint16).astype(np.uint8)*np.uint8(37)+np.uint8(11)
    row = row.reshape(3, width)
    row[:, :2] = np.frombuffer(np.uint16(half).tobytes(), np.uint8)
    if fmt == 'q4_k':
        row[:, 2:4] = np.frombuffer(np.uint16(0x0801).tobytes(), np.uint8)
    return row.reshape(-1)


def source(name, k):
    x = np.resize(np.array([0., -0., 1., -1., 1.0078125, 2**-20, 32.25, -64.5], np.float32), k)
    if name == 'bf16':
        bits=x.view(np.uint32)
        return ((bits+0x7fff+((bits>>16)&1))>>16).astype(np.uint16)
    return x.astype({'f32': np.float32, 'f64': np.float64, 'f16': np.float16, 'i64': np.int64}[name])


@pytest.fixture(scope='module', params=['NVIDIA', 'ILUVATAR'])
def host(request, tmp_path_factory):
    p = tmp_path_factory.mktemp('checked-reader-'+request.param)
    wrappers='\n'.join(
        f'extern "C" float {fmt}_{name}(const {typ}*x,const unsigned char*w,long long k)'
        f'{{return ftt::quantized::dot<encoding::{fmt}>(x,w,k);}}\n'
        f'extern "C" float ref_{fmt}_{name}(const {typ}*x,const unsigned char*w,long long k)'
        f'{{float a=0;for(long long i=0;i<k;++i)a+=ftt::quantized::decode<encoding::{fmt}>(w,i)*ftt::to_float(x[i]);return a;}}'
        for fmt in ('iq4_xs','q4_k') for name,typ in TYPES.items())
    h2 = r'''
extern "C" void h2_pair(const ftt::bfloat16_t*x,const unsigned char*w,const unsigned char*v,long long k,int lane,float*out) {
 const auto a=_flagmega_h2_lane_dot<true>(x,w,v,k,lane);out[0]=a.first;out[1]=a.second;
}
extern "C" float h2_ref(const ftt::bfloat16_t*x,const unsigned char*w,long long k,int lane) {
 float a=0;for(long long block=0;block<k/256;++block) {
 const long long start=block*256+(lane/4)*32+(lane%4)*4;
 for(int i=0;i<4;++i){a+=ftt::quantized::decode<encoding::iq4_xs>(w,start+i)*ftt::to_float(x[start+i]);
 a+=ftt::quantized::decode<encoding::iq4_xs>(w,start+16+i)*ftt::to_float(x[start+16+i]);}}
 return a;
}
'''
    cpp=p/'host.cpp';cpp.write_text(SOURCE+'\n'+(LOCAL/'templates/kernels/_iq4_warp_f32.cpp.jinja').read_text()+wrappers+h2)
    library=p/'host.so';cmd=['c++','-std=c++20','-O3','-ffp-contract=off','-fno-fast-math','-shared','-fPIC',
        '-DFTT_ARCH_'+request.param,f'-I{FTT}',str(cpp),'-o',str(library)]
    if request.param=='NVIDIA':
        cmd += ['-I/usr/local/cuda/include','-I/usr/local/cuda/include/cccl']
    result=subprocess.run(cmd,capture_output=True,text=True)
    (p/'compiler-receipt.json').write_text(json.dumps({'argv':cmd,'actual_exit_code':result.returncode,'scope':'host only; no CUDA initialization'},indent=2)+'\n')
    (p/'compile.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stderr
    lib=ctypes.CDLL(str(library));lib.reader_word.restype=ctypes.c_uint;lib.order.restype=ctypes.c_longlong
    for name in ['empty','guarded','h2_ref',*(f'{fmt}_{n}' for fmt in ('iq4_xs','q4_k') for n in TYPES),*(f'ref_{fmt}_{n}' for fmt in ('iq4_xs','q4_k') for n in TYPES)]:
        getattr(lib,name).restype=ctypes.c_float
    return lib


def test_checked_reader_host_little_endian_all_components_offsets_and_final_word(host):
    row=np.arange(136,dtype=np.uint8)*np.uint8(19)+np.uint8(13)
    for shift in range(4):
        storage=np.zeros(136+shift,np.uint8);storage[shift:]=row
        for off in range(0,133,4):
            expected=sum(int(row[off+i])<<(8*i) for i in range(4))
            assert host.reader_word(ptr(storage[shift:]),ctypes.c_longlong(off))==expected


@pytest.mark.parametrize('name', list(TYPES))
def test_checked_reader_five_source_types_and_unchanged_q4_neighbor(host,name):
    for fmt in ('iq4_xs','q4_k'):
        row=weights(fmt);x=source(name,768)
        for shift in range(4):
            storage=np.zeros(row.size+shift,np.uint8);storage[shift:]=row
            for k in KS:
                equal_bits(getattr(host,fmt+'_'+name)(ptr(x),ptr(storage[shift:]),ctypes.c_longlong(k)),
                           getattr(host,'ref_'+fmt+'_'+name)(ptr(x),ptr(storage[shift:]),ctypes.c_longlong(k)))


def test_checked_reader_positive_zero_order_and_minimal_prefix_tail_spans(host):
    for k in KS[:3]:
        equal_bits(host.empty(ctypes.c_longlong(k)),0.)
    row=weights();x=np.ones(768,np.float32)
    for k in KS[3:]:
        assert host.order(ptr(row),ctypes.c_longlong(k))==k
        blocks,tail=divmod(k,256)
        nbytes=blocks*136
        if tail:
            group=(tail-1)//32;features=tail-group*32
            nbytes+=8+group*16+min(16,features)
        equal_bits(host.guarded(ptr(row),ctypes.c_longlong(nbytes),ctypes.c_longlong(k)),
                   host.ref_iq4_xs_f32(ptr(x),ptr(row),ctypes.c_longlong(k)))


def test_checked_reader_low_before_high_cancellation(host):
    row=np.full(136,0x88,np.uint8);row[:2]=np.frombuffer(np.uint16(0x3c00).tobytes(),np.uint8)
    row[2:4]=0xaa;row[4:8]=0x11;x=np.zeros(256,np.uint16);x[0],x[4],x[16]=0x60ad,0xe0ad,0x4040
    equal_bits(host.iq4_xs_bf16(ptr(x),ptr(row),ctypes.c_longlong(256)),3.)


def test_checked_reader_registry_identity_binds_both_transitive_headers(tmp_path):
    # Execute the actual pure fingerprint function in isolation. Registration and
    # selection are exercised by the normal full build, without loading them here.
    tree=ast.parse((LOCAL/'registry.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='source_identity')
    root=tmp_path/'local';inc=tmp_path/'include';root.mkdir()
    (root/'helper.jinja').write_text('unchanged local helper')
    for name in ['ftt/quantized_ops.h','ftt/arch/nvidia/primitive_ops.h']:
        p=inc/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('initial '+name)
    ns={'ROOT':root,'FTT_INCLUDE_DIR':inc,'RECIPE':'fixed','hashlib':hashlib,'json':json,
        '_configuration':lambda recipe,families,mode:(recipe,('block_fp8','matmul_glu'),mode)}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'source_identity','exec'),ns)
    first=ns['source_identity']();(inc/'ftt/arch/nvidia/primitive_ops.h').write_text('changed primitive only')
    second=ns['source_identity']()
    assert first['code_sha256']==second['code_sha256']
    assert first['core_reader_sources']['ftt/quantized_ops.h']==second['core_reader_sources']['ftt/quantized_ops.h']
    assert first['core_reader_sources']['ftt/arch/nvidia/primitive_ops.h']!=second['core_reader_sources']['ftt/arch/nvidia/primitive_ops.h']
    assert first['sha256']!=second['sha256']


def test_checked_reader_h2_host_single_dual_perlane_order_without_gpu_collectives(host):
    row=weights();other=np.roll(row,13).copy();other.reshape(3,136)[:,:2]=np.frombuffer(np.uint16(0xbc01).tobytes(),np.uint8)
    x=source('bf16',768);out=np.zeros(2,np.float32)
    for shift in range(4):
        w=np.zeros(row.size+shift,np.uint8);v=np.zeros(other.size+shift,np.uint8);w[shift:]=row;v[shift:]=other
        for k in (0,256,512,768):
            for lane in range(32):
                host.h2_pair(ptr(x),ptr(w[shift:]),ptr(v[shift:]),ctypes.c_longlong(k),ctypes.c_int(lane),ptr(out))
                equal_bits(out[0],host.h2_ref(ptr(x),ptr(w[shift:]),ctypes.c_longlong(k),ctypes.c_int(lane)))
                equal_bits(out[1],host.h2_ref(ptr(x),ptr(v[shift:]),ctypes.c_longlong(k),ctypes.c_int(lane)))


def test_checked_reader_exact_full_prefix_span_and_load_count_model():
    # Source access model, not measured memory transactions/device load counts.
    for k in KS:
        blocks=max(0,k)//256
        for block in range(blocks):
            base=136*block
            words=[base+4]+[base+8+16*g+4*j for g in range(8) for j in range(4)]
            bytes_read=[base+i for i in range(4)]+[p+i for p in words for i in range(4)]
            assert len(words)==33 and len(bytes_read)==136
            assert sorted(bytes_read)==list(range(base,base+136))
            h2_words=[base,base+4]+[base+8+4*lane for lane in range(32)]
            assert sorted(p+i for p in h2_words for i in range(4))==list(range(base,base+136))
        if k<=255:
            assert blocks==0

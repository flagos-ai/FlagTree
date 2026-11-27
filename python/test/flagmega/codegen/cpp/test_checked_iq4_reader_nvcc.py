# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Actual device-code compilation only: no context or GPU execution."""
import json
import re
import subprocess
from pathlib import Path

REPO=Path(__file__).resolve().parents[5]
FTT=REPO/'python/triton/flagmega/codegen/cpp/ftt/include'
LOCAL=REPO/'python/tutorials/flagmega/03-qwen3.8-27b-iq4/nvidia-rtx5060ti-cuda/local_optimizations'
NVCC='/usr/local/cuda/bin/nvcc'

PROBES=r'''
#include <ftt/quantized_ops.h>
extern "C" __global__ void reader_global(const unsigned char*w,unsigned int*out,int offset) {
 const ftt::quantized::detail::iq4_row_reader reader(w+offset);
 out[0]=reader.uses_global_words();out[1]=reader.word(0);out[2]=reader.word(132);
}
extern "C" __global__ void reader_shared(unsigned int*out,int offset,unsigned int seed) {
 __shared__ __align__(4) unsigned char row[140];
 for(unsigned i=threadIdx.x;i<140;i+=blockDim.x)row[i]=static_cast<unsigned char>(i+seed);
 __syncthreads();
 if(threadIdx.x==0){const ftt::quantized::detail::iq4_row_reader reader(row+offset);
 out[0]=reader.uses_global_words();out[1]=reader.word(0);out[2]=reader.word(132);}
}
extern "C" __global__ void reader_local(unsigned int*out,int offset,unsigned int seed) {
 __align__(4) unsigned char row[140];
 for(unsigned i=0;i<140;++i)row[i]=static_cast<unsigned char>(i+seed);
 asm volatile(""::"l"(row):"memory");
 const ftt::quantized::detail::iq4_row_reader reader(row+offset);
 out[0]=reader.uses_global_words();out[1]=reader.word(0);out[2]=reader.word(132);
}
extern "C" __global__ void reader_empty(const float*x,const unsigned char*w,float*out) {
 out[0]=ftt::quantized::dot<ftt::quantized::encoding::iq4_xs>(x,w,0);
}
extern "C" __global__ void reader_tail(const float*x,const unsigned char*w,float*out) {
 out[0]=ftt::quantized::dot<ftt::quantized::encoding::iq4_xs>(x,w,255);
}
extern "C" __global__ void reader_q4_neighbor(const float*x,const unsigned char*w,float*out,long long k) {
 out[0]=ftt::quantized::dot<ftt::quantized::encoding::q4_k>(x,w,k);
}
extern "C" __global__ void reader_decode_neighbor(const unsigned char*w,float*out,long long k) {
 out[0]=ftt::quantized::decode<ftt::quantized::encoding::iq4_xs>(w,k);
}
'''


def test_checked_reader_real_nvcc_global_shared_local_no_prefix_and_h2_single_dual(tmp_path):
    source=tmp_path/'reader-probes.cu'
    dots='\n'.join(f'extern "C" __global__ void reader_dot_{name}(const {typ}*x,const unsigned char*w,float*out,long long k)'
                  '{out[0]=ftt::quantized::dot<ftt::quantized::encoding::iq4_xs>(x,w,k);}'
                  for name,typ in [('bf16','ftt::bfloat16_t'),('f32','float'),('f16','__half'),('f64','double'),('i64','long long')])
    h2='\n'+(LOCAL/'templates/kernels/_iq4_warp_f32.cpp.jinja').read_text()+r'''
extern "C" __global__ void reader_h2_single(const ftt::bfloat16_t*x,const unsigned char*w,float*out,long long k) {
 const auto a=_flagmega_h2_warp_dot<false>(x,w,nullptr,k);if(threadIdx.x==0)out[0]=a.first;
}
extern "C" __global__ void reader_h2_dual(const ftt::bfloat16_t*x,const unsigned char*w,const unsigned char*v,float*out,long long k) {
 const auto a=_flagmega_h2_warp_dot<true>(x,w,v,k);if(threadIdx.x==0){out[0]=a.first;out[1]=a.second;}
}
'''
    source.write_text(PROBES+dots+h2);receipts=[]
    for kind in ['ptx','cubin']:
        output=tmp_path/('reader-probes.'+kind)
        cmd=[NVCC,'-std=c++20','-O3','-arch=sm_120','--maxrregcount=168','-DFTT_ARCH_NVIDIA',f'-I{FTT}',
             '--'+kind,str(source),'-o',str(output)]
        result=subprocess.run(cmd,capture_output=True,text=True)
        (tmp_path/(kind+'-compile.log')).write_text(result.stdout+result.stderr)
        receipts.append({'argv':cmd,'actual_exit_code':result.returncode,'scope':'compile only; default FMA, no GPU'})
        assert result.returncode==0,result.stderr
    text=(tmp_path/'reader-probes.ptx').read_text()
    bodies=dict(re.findall(r'\.visible \.entry (\w+)(.*?)(?=\.visible \.entry |$)',text,re.S))
    for name in ['reader_global','reader_shared','reader_local','reader_dot_bf16','reader_dot_f32','reader_dot_f16','reader_dot_f64','reader_dot_i64','reader_h2_single']:
        body=bodies[name]
        assert body.count('isspacep.global')==1,name
        assert re.search(r'cvta\.to\.global\.u64',body),name
        assert re.search(r'ld\.global\.u32',body),name
    assert bodies['reader_h2_dual'].count('isspacep.global')==2
    assert '.shared' in bodies['reader_shared'] or 'ld.shared' in bodies['reader_shared']
    assert '.local' in bodies['reader_local'] or 'ld.local' in bodies['reader_local']
    for name in ['reader_empty','reader_tail','reader_q4_neighbor','reader_decode_neighbor']:
        assert 'isspacep.global' not in bodies[name]
    assert not re.search(r'\bld\.(?:global|shared|local|const|generic|[busf]\d+)\b',bodies['reader_empty'])
    assert all('shfl.sync' not in bodies['reader_dot_'+name] for name in ['bf16','f32','f16','f64','i64'])
    for flag,kind in [('--dump-resource-usage','resources'),('--dump-sass','sass')]:
        cmd=['/usr/local/cuda/bin/cuobjdump',flag,str(tmp_path/'reader-probes.cubin')]
        result=subprocess.run(cmd,capture_output=True,text=True)
        receipts.append({'argv':cmd,'actual_exit_code':result.returncode,'scope':'offline ELF; no GPU'})
        (tmp_path/(kind+'.log')).write_text(result.stdout+result.stderr)
        assert result.returncode==0,result.stderr
    (tmp_path/'compiler-receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')

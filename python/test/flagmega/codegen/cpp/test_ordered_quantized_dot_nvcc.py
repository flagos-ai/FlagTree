# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Compile-only proof that complete IQ4 blocks use register caches and ordered math."""
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import triton.flagmega.codegen.cpp as cpp_codegen

FTT_INCLUDE = Path(cpp_codegen.__file__).parent / 'ftt/include'
CUDA_ROOT = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda'))
NVCC = shutil.which('nvcc') or str(CUDA_ROOT / 'bin/nvcc')


def anchors():
    source = '#include <ftt/quantized_ops.h>\n#include <cuda_fp16.h>\n'
    for name, typ in [('bf16', 'ftt::bfloat16_t'), ('f32', 'float'), ('f16', '__half'),
                      ('f64', 'double'), ('i64', 'long long')]:
        source += f'''extern "C" __global__ void iq4_fixed_{name}(const {typ}*x,const unsigned char*w,float*y) {{
    y[threadIdx.x]=ftt::quantized::dot<ftt::quantized::encoding::iq4_xs>(x,w,512);
}}
'''
    source += '''extern "C" __global__ void iq4_dynamic(const float*x,const unsigned char*w,float*y,long long k) {
    y[threadIdx.x]=ftt::quantized::dot<ftt::quantized::encoding::iq4_xs>(x,w,k);
}
extern "C" __global__ void q4_neighbor(const float*x,const unsigned char*w,float*y) {
    y[threadIdx.x]=ftt::quantized::dot<ftt::quantized::encoding::q4_k>(x,w,512);
}
extern "C" __global__ void decode_neighbor(const unsigned char*w,float*y,long long k) {
    y[threadIdx.x]=ftt::quantized::decode<ftt::quantized::encoding::iq4_xs>(w,k);
}
'''
    return source


@pytest.mark.skipif(not Path(NVCC).is_file(), reason='NVCC unavailable; no device execution in this test')
@pytest.mark.parametrize('contraction', [True, False])
def test_iq4_full_block_register_cache_and_target_contraction(tmp_path, contraction):
    source=tmp_path/'ordered.cu';source.write_text(anchors())
    output=tmp_path/'ordered.ptx'
    command=[NVCC,'-std=c++20','-O3','-arch=sm_120','-DFTT_ARCH_NVIDIA',f'-I{FTT_INCLUDE}',
             f'--fmad={str(contraction).lower()}','--ptx',str(source),'-o',str(output)]
    p=subprocess.run(command,capture_output=True,text=True)
    (tmp_path/'compile.log').write_text(p.stdout+p.stderr)
    assert p.returncode==0,p.stderr
    bodies=dict(re.findall(r'\.visible \.entry (\w+)(.*?)(?=\.visible \.entry |$)',output.read_text(),re.S))
    for name in ['bf16','f32','f16','f64','i64']:
        body=bodies['iq4_fixed_'+name]
        # The pre-cache implementation materializes its dynamically indexed
        # LUT in .local memory. Complete blocks must use register selection.
        assert '.local' not in body
        assert '.shared' not in body
        assert '.ftz' not in body
        if contraction:
            assert 'fma.rn.f32' in body
        else:
            assert 'fma.' not in body
            assert re.search(r'add(?:\.rn)?\.f32',body)
        assert re.search(r'mul(?:\.rn)?\.f32',body)
        assert re.search(r'ld\.global(?:\.nc)?\.[bu]8',body)  # Byte-aligned rows.
    # Both adjacent paths still compile. Dynamic IQ4's exact scalar tail and
    # standalone decode may retain the original LUT; they are not cache misses
    # for complete blocks and must not be replaced by speculative wider reads.
    assert {'iq4_dynamic','q4_neighbor','decode_neighbor'} <= bodies.keys()

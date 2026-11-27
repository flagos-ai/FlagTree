# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Compile-only default-PRMT and ordered grouped arithmetic witnesses."""
import re
import importlib.util
import subprocess
from pathlib import Path

import pytest
_spec = importlib.util.spec_from_file_location('ordered_dot_NVCC_helpers', Path(__file__).with_name('test_ordered_quantized_dot_nvcc.py'))
_ordered = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ordered)
FTT_INCLUDE, NVCC, anchors = _ordered.FTT_INCLUDE, _ordered.NVCC, _ordered.anchors


@pytest.mark.skipif(not Path(NVCC).is_file(), reason='NVCC unavailable; no GPU execution')
def test_grouped_default_prmt_signed_selectors_and_scalar_sources(tmp_path):
    source = tmp_path / 'grouped.cu'
    source.write_text(anchors() + '''
extern "C" __global__ void primitive_probe(unsigned int w,unsigned int*y,int*z) {
    y[0]=ftt::arch::byte_permute_default(w,0u,0x9991u);
    y[1]=__byte_perm(w,0u,0x9991u);
    z[0]=ftt::arch::signed_byte_extract<0>(w);
    z[1]=ftt::arch::signed_byte_extract<1>(w);
    z[2]=ftt::arch::signed_byte_extract<2>(w);
    z[3]=ftt::arch::signed_byte_extract<3>(w);
}
''')
    output = tmp_path / 'grouped.ptx'
    command = [NVCC, '-std=c++20', '-O3', '-arch=sm_120', '--fmad=true',
               '-DFTT_ARCH_NVIDIA', f'-I{FTT_INCLUDE}', '--ptx', str(source), '-o', str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    (tmp_path / 'compile.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stderr
    bodies = dict(re.findall(r'\.visible \.entry (\w+)(.*?)(?=\.visible \.entry |$)', output.read_text(), re.S))
    for name in ('bf16', 'f32', 'f16', 'f64', 'i64'):
        body = bodies['iq4_fixed_' + name]
        assert len(re.findall(r'\bprmt\.b32\b', body)) == 12
        for selector in (34944, 39313, 43682, 48051):
            assert len(re.findall(rf'prmt\.b32 [^;]*, {selector};', body)) == 2
        assert len(re.findall(r'\bfma\.rn\.f32\b', body)) == 8
        assert re.search(r'mul(?:\.rn)?\.f32', body)
        assert '.local' not in body and '.shared' not in body and 'shfl.' not in body
        assert '.ftz' not in body and 'ld.global.v' not in body
        assert re.search(r'ld\.global(?:\.nc)?\.[bu]8', body)
    assert re.search(r'prmt\.b32[^;]*0x1111U;', bodies['primitive_probe'])
    # This demonstrates the intrinsic masks its sign selectors at compile time;
    # it does not claim a device result. Default sign controls above are distinct.
    assert {'iq4_dynamic', 'q4_neighbor', 'decode_neighbor'} <= bodies.keys()

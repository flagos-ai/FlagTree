# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Small standalone GDN compile checks; no device execution."""
from pathlib import Path
import re
import subprocess
import pytest

from triton.flagmega.codegen.cpp.templates import CppTemplateRegistry
from python.test.flagmega.codegen.cpp.test_scalar_head_gates import TYPES, call
from python.test.flagmega.codegen.cpp.test_native_quantized_kernels import FTT_INCLUDE

NVCC=Path('/usr/local/cuda/bin/nvcc')


@pytest.mark.skipif(not NVCC.is_file(),reason='NVCC unavailable')
@pytest.mark.parametrize('contraction',[True,False])
@pytest.mark.parametrize('hierarchy',[(4,8),(4,24)])
def test_scalar_gate_publication_compiles_with_original_contraction(tmp_path,contraction,hierarchy):
    body=CppTemplateRegistry().render('kernels/gdn_recurrent/persistent.cpp.jinja',dict(
        render_calls=[call(offset=32,hierarchy=hierarchy)],argument_types={n:dict(type=t,pointer=True) for n,t in TYPES.items()},
        mesh_hierarchy=hierarchy,distributed_entry=True,platform='nvidia',block_threads=128,
        grid_barrier_axis_groups=[],grid_total_blocks=hierarchy[0]*hierarchy[1]))
    params=', '.join([*(t+' *'+n for n,t in TYPES.items()),'unsigned int *ftt_grid_barrier'])
    args=', '.join([*TYPES,'ftt_grid_barrier'])
    source=tmp_path/'c2.cu';source.write_text('#include <ftt/ftt.h>\n'+body+f'\nextern "C" __global__ void c2({params}){{kernel__core({args});}}\n')
    ptx=tmp_path/'c2.ptx'
    cmd=[str(NVCC),'-std=c++20','-O3','-arch=sm_120','-DFTT_ARCH_NVIDIA',f'-I{FTT_INCLUDE}',
         f'--fmad={str(contraction).lower()}','--ptx',str(source),'-o',str(ptx)]
    result=subprocess.run(cmd,capture_output=True,text=True)
    (tmp_path/'compile.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stderr
    body=ptx.read_text().split('.visible .entry c2(',1)[1]
    assert len(re.findall(r'bar\.sync',body))==1
    assert body.index('st.shared')<body.index('bar.sync')<body.index('ld.shared')
    if contraction:
        assert 'fma.rn.f32' in body
    else:
        # CUDA's exp/log implementations contain explicit intrinsic FMAs even
        # with contraction disabled; the scalar product chain uses mul/add.
        assert 'mul.rn.f32' in body and 'add.rn.f32' in body

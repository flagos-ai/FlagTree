# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A shared sliced weight must keep distinct physical values on the device."""

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.importer import DirectoryCheckpoint
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule
from python.test.flagmega.devices import is_nvidia_sm90


@pytest.mark.parametrize('k,n,parts', [(64,128,2), (8192,4096,4)])
def test_offline_sliced_parameter_values_reach_each_projection(tmp_path,k,n,parts):
    if not is_nvidia_sm90():
        pytest.skip('SM90 CUDA is required')
    from safetensors.torch import save_file

    class Graph(fm.Module):
        def forward(self):
            x_type,w_type=fm.tensor_type('bfloat16',(1,k)),fm.tensor_type('bfloat16',(n,k))
            x=self.input('worker_x',x_type,id='worker_x',metadata={'function_parameter':'worker'})
            w=self.input('worker_w',w_type,id='worker_w',metadata={'function_parameter':'worker'})
            output=[]
            for p in range(parts):
                a=fm.F.tensors.slice(x,starts=(p*k//parts,),ends=((p+1)*k//parts,),axes=(1,))
                b=fm.F.tensors.slice(w,starts=(p*k//parts,),ends=((p+1)*k//parts,),axes=(1,))
                output.append(fm.F.math.matmul(a,b,transpose_b=True,name=f'dot_{p}'))
            self.function('worker',(x,w),output,attrs={'reusable':True,'calling_convention':'device'})
            value=self.input('value',x_type,id='value')
            weight=self.weight('weight',w_type,source='model.safetensors',key='weight',id='weight')
            call=fm.F.builtin.call(value,weight,callee='worker',result_type=fm.TupleType(tuple(o.type for o in output)))
            self.function('main',(value,),fm.F.tensors.get_items(call,*range(parts)))

    weights=torch.zeros((n,k),dtype=torch.bfloat16)
    rows=torch.arange(n)
    for p in range(parts):
        weights[rows,p*k//parts+rows%(k//parts)]=p+1
    checkpoint=tmp_path/'checkpoint'
    checkpoint.mkdir()
    (checkpoint/'config.json').write_text('{}\n')
    save_file({'weight':weights},checkpoint/'model.safetensors')
    module=Graph(dialect='high_level',stage='imported',entry='main').build()
    artifact=write_artifact(Compiler().compile(module).module,tmp_path/'artifact',target='nvidia-sm90',
                            checkpoint=DirectoryCheckpoint(checkpoint),emit_executable=True)
    source=(artifact/'generated_kernels.py').read_text()
    assert '# flagmega-kernel: pack/' not in source
    runtime=load(artifact,device='cuda:0')
    try:
        value=torch.cat([p*16+torch.arange(k//parts)%16 for p in range(parts)]).reshape(1,k).bfloat16().cuda()
        buffers=tuple(torch.empty((1,n),device='cuda',dtype=torch.bfloat16) for _ in range(parts))
        result_specs=[s for s in runtime.external_arguments if s['role']=='result']
        outputs={s['value']:v for s,v in zip(result_specs,buffers,strict=True)}
        args=tuple(outputs[s['value']] if s['role']=='result' else value for s in runtime.external_arguments)
        GeneratedTirCallGraphModule.prepare(runtime,*args)
        GeneratedTirCallGraphModule.run_into(runtime,*args)
        for p,buffer in enumerate(buffers):
            expected=((p+1)*(p*16+rows%16)).reshape(1,n).bfloat16()
            torch.testing.assert_close(buffer.cpu(),expected,rtol=0,atol=0)
        assert runtime.resource_report['spill_bytes']==0
    finally:
        runtime.close()

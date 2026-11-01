"""Give the original four sharedGLU outputs one lane each along N."""
from pathlib import Path
import hashlib
from jinja2 import Environment,ChoiceLoader,DictLoader
from triton.flagmega.errors import CodegenError

HERE=Path(__file__).resolve().parent
KEY='kernels/matmul_glu/platforms/nvidia/sm90/cooperative_wgmma.py.jinja'

def transform(source):
    changes={
        '    [1, 1], [4, 8], [1, {{ num_warps }}], [1, 0]))':
        '{% if call.shared_merge_n4 | default(false) %}\n    [1, 1], [8, 4], [{{ num_warps }}, 1], [1, 0]))\n{% else %}\n    [1, 1], [4, 8], [1, {{ num_warps }}], [1, 0]))\n{% endif %}',
        'for local_n_start in tl.range(0, {{ call.local_n_capacity }}, 16):':
        'for local_n_start in tl.range(0, {{ call.local_n_capacity }}, {{ 4 if call.shared_merge_n4 | default(false) else 16 }}):',
        'local_n_offsets = local_n_start + tl.arange(0, 16)':
        'local_n_offsets = local_n_start + tl.arange(0, {{ 4 if call.shared_merge_n4 | default(false) else 16 }})',
        'tl.full((16,), 0., tl.float32)':
        'tl.full(({{ 4 if call.shared_merge_n4 | default(false) else 16 }},), 0., tl.float32)',
        'tl.full((1, 16), _fm_block, tl.int32)':
        'tl.full((1, {{ 4 if call.shared_merge_n4 | default(false) else 16 }}), _fm_block, tl.int32)',
        '(16,))':
        '({{ 4 if call.shared_merge_n4 | default(false) else 16 }},))',
    }
    counts=[1,1,1,2,1,4]
    for (old,new),count in zip(changes.items(),counts,strict=True):
        if source.count(old)!=count:raise CodegenError('Original ordered sharedGLU merge template differs: '+old)
        source=source.replace(old,new)
    return source

def activate():
    from triton.flagmega.codegen.triton import renderer,tir_package
    from triton.flagmega.codegen.triton.registry import PackageRendererSpec
    previous=tir_package.describe_tir_package
    def describe(module):
        d=previous(module)
        calls=[c for c in d['render_calls'] if c['family']=='matmul_glu' and 'shared_up' in c['symbol']]
        if len(calls)!=2:raise CodegenError('Exactly two original sharedGLU definitions required')
        expected=dict(variant='cooperative_wgmma',local_n_capacity=4,local_rows=4,chip_rows=512,k_blocks=32,k_block_lanes=32,row_owners=128,raw_tasks=256)
        for c in calls:
            bad={k:(c.get(k),v) for k,v in expected.items() if c.get(k)!=v}
            if bad or c['parameters']['compute_num_warps']!=4:raise CodegenError('Complete sharedGLU physical contract differs: '+repr(bad))
            c['shared_merge_n4']=True
        d['shared_merge_n4']=dict(symbols=[c['symbol'] for c in calls],old_shape=[32,16],new_shape=[32,4],all4_outputs_and32_ordered_fma=True,original_raw_fp8_and_scratch=True,barriers_unchanged=True)
        d['implementation_override']['inputs']['shared_merge_n4.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return d
    tir_package.describe_tir_package=describe
    spec=renderer.renderer_registry._specs['bufferized-tir']
    renderer.renderer_registry._specs[spec.name]=PackageRendererSpec(spec.name,spec.kind,spec.matches,describe,spec.render)
    original=Environment.get_template
    def selected(self,name,*args,**kwargs):
        if name in {'module.py.jinja',KEY}:
            source=self.loader.get_source(self,KEY)[0]
            if 'shared_merge_n4' not in source:self.loader=ChoiceLoader([DictLoader({KEY:transform(source)}),self.loader])
        return original(self,name,*args,**kwargs)
    Environment.get_template=selected

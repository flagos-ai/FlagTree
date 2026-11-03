"""Group two original K128 updates in a four-stage complete-K FP8 pipeline."""
from jinja2 import Environment,ChoiceLoader,DictLoader
from triton.flagmega.errors import CodegenError
KEY='kernels/sparse_experts_gate_up/platforms/nvidia/sm90/complete_k_wgmma.py.jinja'
SHARED_KEY='kernels/matmul_glu/platforms/nvidia/sm90/cooperative_wgmma.py.jinja'
PIPELINE='num_stages=4, loop_unroll_factor=2'
def transform(source):
    anchor='num_stages={{ call.num_stages }}'
    assert source.count(anchor)==1
    source=source.replace(anchor,PIPELINE)
    for old,new in [('_fm_total = tl.zeros(({{ call.matrix_rows }}, 16), tl.float32)','_fm_total = tl.zeros(({{ call.matrix_rows }}, 8), tl.float32)'),('_fm_column = tl.arange(0, 16)','_fm_column = tl.arange(0, 8)')]:
        assert source.count(old)==1
        source=source.replace(old,new)
    return source
def transform_shared_up(source):
    old='tl.broadcast_to(_fm_q[:, None], (128, 16))'
    if source.count(old)!=1:
        raise CodegenError('Original shared GLU activation broadcast differs')
    return source.replace(old,'tl.broadcast_to(_fm_q[:, None], (128, 8))')
def activate():
    original=Environment.get_template
    def selected_template(self,name,*args,**kwargs):
        if name in {'module.py.jinja',KEY}:
            source=self.loader.get_source(self,KEY)[0]
            if PIPELINE not in source:
                self.loader=ChoiceLoader([DictLoader({KEY:transform(source)}),self.loader])
        if name in {'module.py.jinja',SHARED_KEY}:
            source=self.loader.get_source(self,SHARED_KEY)[0]
            if 'tl.broadcast_to(_fm_q[:, None], (128, 8))' not in source:
                self.loader=ChoiceLoader([DictLoader({SHARED_KEY:transform_shared_up(source)}),self.loader])
        return original(self,name,*args,**kwargs)
    Environment.get_template=selected_template

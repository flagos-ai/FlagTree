"""Explicit private eight-stage cp.async pipeline for original complete-K gate/up."""
from jinja2 import Environment,ChoiceLoader,DictLoader
from triton.flagmega.errors import CodegenError
KEY='kernels/sparse_experts_gate_up/platforms/nvidia/sm90/complete_k_wgmma.py.jinja'
def transform(source):
    anchor='num_stages={{ call.num_stages }}'
    assert source.count(anchor)==1
    source=source.replace(anchor,'num_stages=8')
    for old,new in [('_fm_total = tl.zeros(({{ call.matrix_rows }}, 16), tl.float32)','_fm_total = tl.zeros(({{ call.matrix_rows }}, 8), tl.float32)'),('_fm_column = tl.arange(0, 16)','_fm_column = tl.arange(0, 8)')]:
        assert source.count(old)==1
        source=source.replace(old,new)
    return source
def activate():
    original=Environment.get_template
    def selected_template(self,name,*args,**kwargs):
        if name in {'module.py.jinja',KEY}:
            source=self.loader.get_source(self,KEY)[0]
            if 'num_stages=8' not in source:
                self.loader=ChoiceLoader([DictLoader({KEY:transform(source)}),self.loader])
        return original(self,name,*args,**kwargs)
    Environment.get_template=selected_template

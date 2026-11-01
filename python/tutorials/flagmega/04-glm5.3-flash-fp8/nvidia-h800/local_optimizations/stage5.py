"""Explicit private five-stage cp.async pipeline for original complete-K gate/up."""
from jinja2 import Environment,ChoiceLoader,DictLoader
from triton.flagmega.errors import CodegenError
KEY='kernels/sparse_experts_gate_up/platforms/nvidia/sm90/complete_k_wgmma.py.jinja'
def transform(source):
    anchor='num_stages={{ call.num_stages }}'
    assert source.count(anchor)==1
    return source.replace(anchor,'num_stages=5')
def activate():
    original=Environment.get_template
    def selected_template(self,name,*args,**kwargs):
        if name in {'module.py.jinja',KEY}:
            source=self.loader.get_source(self,KEY)[0]
            if 'num_stages=5' not in source:
                self.loader=ChoiceLoader([DictLoader({KEY:transform(source)}),self.loader])
        return original(self,name,*args,**kwargs)
    Environment.get_template=selected_template

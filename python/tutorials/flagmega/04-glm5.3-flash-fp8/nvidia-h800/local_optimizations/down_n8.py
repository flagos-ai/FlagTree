from jinja2 import Environment,ChoiceLoader,DictLoader
KEY='kernels/partitioned_sparse_experts_down/platforms/nvidia/sm90/wgmma.py.jinja'
def transform(source):
 for old,new in [('_fm_acc = tl.zeros(({{ call.output_tile_n }}, 16), tl.float32)','_fm_acc = tl.zeros(({{ call.output_tile_n }}, 8), tl.float32)'),('(128, 16))','(128, 8))')]:
  assert source.count(old)==1
  source=source.replace(old,new)
 return source
def activate():
 original=Environment.get_template
 def selected_template(self,name,*args,**kwargs):
  if name in {'module.py.jinja',KEY}:
   source=self.loader.get_source(self,KEY)[0]
   if '(128, 8))' not in source:self.loader=ChoiceLoader([DictLoader({KEY:transform(source)}),self.loader])
  return original(self,name,*args,**kwargs)
 Environment.get_template=selected_template

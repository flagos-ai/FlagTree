"""Give each token one warp for repeated mHC mix arithmetic; keep original loops."""
from jinja2 import Environment,ChoiceLoader,DictLoader
from triton.flagmega.errors import CodegenError
KEY='kernels/mhc_pre/cooperative_stream_fma.py.jinja'
def transform(source):
    start=source.index('def {{ call.symbol }}__mix(')
    anchor='    for _fm_token in tl.range(_fm_cta, {{ call.token_active }}, _fm_ctas):'
    pos=source.index(anchor,start)
    end=source.index('\n\n@triton.jit(noinline=True)\ndef {{ call.symbol }}__collapse',pos)
    original=source[pos:end]
    condition='{% if call.variant == "cooperative_stream_fma" and call.hc == 4 and call.projection_splits == 8 %}\n'
    guard='    _fm_mix_warp = tl.inline_asm_elementwise("mov.u32 $0, %tid.x;", constraints="=r", args=[], dtype=tl.int32, is_pure=True, pack=1) // 32\n    if _fm_mix_warp == 0:\n'
    indented=''.join('    '+line if line.strip() and not line.startswith('{%') else line for line in original.splitlines(keepends=True))
    return source[:pos]+condition+guard+indented+'\n{% else %}\n'+original+'\n{% endif %}'+source[end:]
def activate():
    original=Environment.get_template
    def selected_template(self,name,*args,**kwargs):
        if name in {'module.py.jinja',KEY}:
            source=self.loader.get_source(self,KEY)[0]
            if '_fm_mix_warp =' not in source:
                self.loader=ChoiceLoader([DictLoader({KEY:transform(source)}),self.loader])
        return original(self,name,*args,**kwargs)
    Environment.get_template=selected_template

"""Keep K128 accumulation while distributing complete V columns over128 CTAs."""
from pathlib import Path
import hashlib
from triton.flagmega.errors import CodegenError

def activate():
    from triton.flagmega.codegen.triton import renderer,tir_package
    from triton.flagmega.codegen.triton.registry import PackageRendererSpec
    prior=tir_package.describe_tir_package
    def describe(module):
        d=prior(module)
        calls=[c for c in d['render_calls'] if c['family']=='mla_projection' and c['direction']=='expand_value']
        if len(calls)!=1:
            raise CodegenError('Expected one shared full MLA V expansion')
        c=calls[0]
        if not (c['variant']=='mma' and c['heads']==16 and c['head_sharded']
                and c['reduction']==512 and c['output']==256 and c['tile_n']==64
                and c['reduction_tile']==128 and not c['quantized']):
            raise CodegenError('Original V expansion geometry or complete ownership changed')
        c['tile_n']=32
        # The original mla renderer verifies complete replicated source/weight
        # owners and installs the128-owner task loop later in template rendering.
        d['v_expand32']=dict(original_output_columns=256,original_heads=16,
            original_reduction512_K128=True,original_output_and_BF16_order=True,
            old_tasks=64,new_tasks=128,disjoint_output_owners=True,original_270_ABI_windows=True)
        d['implementation_override']['inputs']['v_expand32.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return d
    tir_package.describe_tir_package=describe
    spec=renderer.renderer_registry._specs['bufferized-tir']
    renderer.renderer_registry._specs[spec.name]=PackageRendererSpec(spec.name,spec.kind,spec.matches,describe,spec.render)

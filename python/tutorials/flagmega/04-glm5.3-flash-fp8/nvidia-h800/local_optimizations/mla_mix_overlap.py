"""Move MLA mHC coefficient work into the original idle QKV CTA64.

The mHC collapse/norm reads only projection pre rows and RMS. The coefficients
are first consumed after the later attention O projection. QKV uses CTAs0..63,
so CTA64 can compute identical post/comb/pre coefficients before its original
internal full grid without changing any consumer lifetime or transfer.
"""
from pathlib import Path
import ast, copy, hashlib
from triton.flagmega.errors import CodegenError
OUT=Path(__file__).resolve().parent
def args(text):return [ast.unparse(v) for v in ast.parse('f('+text+')').body[0].value.args]
def activate():
    from triton.flagmega.codegen.triton import renderer,tir_package
    from triton.flagmega.codegen.triton.registry import PackageRendererSpec
    previous=tir_package.describe_tir_package
    def describe(module):
        d=previous(module);fs=list(d['device_functions'])
        for p in d['programs']:fs.extend(p.get('device_functions',()))
        fs=list({id(f):f for f in fs}.values());calls={c['symbol']:c for c in d['render_calls']}
        f,=[f for f in fs if f['function']=='decode_mla_attention_sparse.prefix']
        events=f['schedule']['consumer_events'];region=events[:4]
        if [(e['kind'],e.get('family',e.get('scope'))) for e in region] != [('tir.kernel_call','mhc_pre'),('barrier','grid'),('tir.kernel_call','partitioned_linear'),('barrier','grid')]:
            raise CodegenError('Original full MLA start region changed')
        m,q=calls[region[0]['symbol']],calls[region[2]['symbol']]
        if not (m.get('owner_fused_rms') and m['hc']==4 and m['hidden']==4096 and m['projection_splits']==8 and m['sinkhorn_iterations']==20 and q.get('qkv_concurrent') and q['rows']==1536 and q['tasks']==24):
            raise CodegenError('Original complete MLA/QKV ownership or arithmetic changed')
        if q['qkv_concurrent']['cta_q']!='0<=pid<48' or q['qkv_concurrent']['cta_kv']!='48<=pid<64':
            raise CodegenError('QKV does not leave CTA64 idle')
        mf=args(m['signature']);ma=args(region[0]['arguments'])
        qf=args(q['signature'])
        if len(mf)!=10 or len(ma)!=10 or mf[-1]!='device_dptr' or 'device_dptr' not in qf:
            raise CodegenError('mHC original numerical helper ABI changed')
        m['mhc_deferred_mix_cta']=64
        added=['_qkv_mhc_arg'+str(i) for i in range(9)]
        if set(added)&set(qf):raise CodegenError('Duplicate QKV extension formals')
        q['signature']+=', '+', '.join(added)
        region[2]['arguments']+=', '+', '.join(ma[:9])
        q['qkv_mhc_mix']=dict(symbol=m['symbol']+'__mix',arguments=', '.join([*added,'device_dptr']),cta=64)
        d['mhc_mla_qkv_overlap']=dict(function=f['function'],original_region=copy.deepcopy(region),
            deferred_mix_cta=64,original_q48_kv16_tasks=True,
            original_mhc_internal_grid_and_collapse_retained=True,
            original_qkv_internal_grid_and_norm_retained=True,
            original_outer_two_grids_retained=True,
            original_post_comb_numerics=True,
            original_window_and_270abi_unchanged=True)
        d['implementation_override']['inputs']['mla_mix_overlap.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return d
    tir_package.describe_tir_package=describe
    spec=renderer.renderer_registry._specs['bufferized-tir']
    renderer.renderer_registry._specs[spec.name]=PackageRendererSpec(spec.name,spec.kind,spec.matches,describe,spec.render)


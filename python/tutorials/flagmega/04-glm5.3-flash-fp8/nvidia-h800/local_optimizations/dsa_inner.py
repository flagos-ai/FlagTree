"""Original four DSA prefetch projections with fewer inner MMA iterations."""
from pathlib import Path
import hashlib
from triton.flagmega.errors import CodegenError
OUT=Path(__file__).resolve().parent
def activate():
    from triton.flagmega.codegen.triton import renderer,tir_package
    from triton.flagmega.codegen.triton.registry import PackageRendererSpec
    previous=tir_package.describe_tir_package
    def describe(module):
        d=previous(module)
        calls=[c for c in d['render_calls'] if c['family']=='dense_matmul' and c['variant']=='packed_k_major_mma_prefetch']
        if len(calls)!=4 or any('dsa_indexer_' not in c['call'] for c in calls):
            raise CodegenError('Exactly original four DSA prefetch projections required')
        for c in calls:
            if not (c['local_m_capacity']==1 and c['local_n_capacity']==8 and c['tile_n']==16 and c['block_k']==32 and c['prefetch_k']==2048 and c['prefetch_n']==8 and c['local_k_capacity'] in {1536,4096}):
                raise CodegenError('Original DSA projection shape or prefetch differs')
            c['block_k']=128
        d['dsa_inner_k_tile']=dict(block_k=128,prefetch_k=2048,original32bytegroups_ordered=True)
        d['implementation_override']['inputs']['dsa_inner.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return d
    tir_package.describe_tir_package=describe
    spec=renderer.renderer_registry._specs['bufferized-tir']
    renderer.renderer_registry._specs[spec.name]=PackageRendererSpec(spec.name,spec.kind,spec.matches,describe,spec.render)

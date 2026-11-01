"""Preserve ordered QK64, distribute global reads along contiguous K."""
from pathlib import Path
import hashlib
from jinja2 import ChoiceLoader,DictLoader
from triton.flagmega.errors import CodegenError

OUT=Path(__file__).resolve().parent
KEY="kernels/dsa_paged_attention/platforms/nvidia/sm90/online_mma.py.jinja"


def transform(source):
    declaration="        _fm_dsa_k = tl.arange(0, 16)"
    loop="                    for _fm_dsa_k_start in tl.range(0, {{ call.head_dim }}, 16):"
    reduction="                        _fm_dsa_reduction = _fm_dsa_k_start + _fm_dsa_k"
    if any(source.count(x)!=1 for x in [declaration,loop,reduction]):
        raise CodegenError("DSA QK loop contract changed; wider loads need review")
    source=source.replace(declaration,declaration+"\n        _fm_dsa_qk_k = tl.arange(0, 128)")
    source=source.replace(loop,"                    for _fm_dsa_k_start in tl.range(0, {{ call.head_dim }}, 128):")
    source=source.replace(reduction,"                        _fm_dsa_reduction = _fm_dsa_k_start + _fm_dsa_qk_k")
    pv_loop="                    for _fm_dsa_pv_start in tl.range(0, 64, 16, num_stages=1):"
    pv_token="_fm_dsa_pv_start + _fm_dsa_k"
    pv_shape="(16, 16)), 1)"
    if any(source.count(x)!=1 for x in [pv_loop,pv_token,pv_shape]):
        raise CodegenError("DSA original ordered PV16 fragment contract differs")
    source=source.replace(declaration,declaration+"\n        _fm_dsa_pv_k = tl.arange(0, 32)")
    source=source.replace(pv_loop,pv_loop.replace(", 16, num_stages",", 32, num_stages"))
    source=source.replace(pv_token,"_fm_dsa_pv_start + _fm_dsa_pv_k")
    source=source.replace(pv_shape,"(16, 32)), 1)")
    layouts = """
{{ call.symbol }}__q_load_layout = tl.constexpr(tle.gpu.BlockEncoding([1, 8], [4, 8], [4, 2], [1, 0]))
{{ call.symbol }}__k_load_layout = tl.constexpr(tle.gpu.BlockEncoding([8, 1], [8, 4], [1, 8], [0, 1]))
"""
    anchor="@triton.jit(noinline=True)"
    if source.count(anchor)!=1:
        raise CodegenError("Attention entry layout anchor changed")
    source=source.replace(anchor,layouts+anchor)
    q="_fm_dsa_q = tl.load({{ call.query }} + ({{ call.mma_query_offset }}), _fm_dsa_head[:, None] < ({{ call.head_active }}), other=0.)"
    k="_fm_dsa_key = tl.load({{ call.kv_cache }} + _fm_dsa_base[None, :] + _fm_dsa_reduction[:, None], _fm_dsa_valid[None, :], other=0.)"
    if source.count(q)!=1 or source.count(k)!=1:
        raise CodegenError("Head-sharded QK global pointer construction changed")
    source=source.replace(q,"_fm_dsa_q = tle.gpu.set_layout(tl.load(tle.gpu.set_layout({{ call.query }} + ({{ call.mma_query_offset }}), {{ call.symbol }}__q_load_layout), _fm_dsa_head[:, None] < ({{ call.head_active }}), other=0.), {{ call.symbol }}__q_load_layout)")
    source=source.replace(k,"_fm_dsa_key = tle.gpu.set_layout(tl.load(tle.gpu.set_layout({{ call.kv_cache }} + _fm_dsa_base[None, :] + _fm_dsa_reduction[:, None], {{ call.symbol }}__k_load_layout), _fm_dsa_valid[None, :], other=0.), {{ call.symbol }}__k_load_layout)")
    return source


def activate():
    from triton.flagmega.codegen.triton import renderer,tir_package
    from triton.flagmega.codegen.triton.registry import PackageRendererSpec
    from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
    previous=tir_package.describe_tir_package
    if getattr(previous,"_qk64_trial",False):
        return
    def describe(module):
        descriptor=previous(module)
        receipt=descriptor["implementation_override"]
        receipt["kind"]="persistent-join-and-dsa-qk64-coalesced/v2"
        receipt["inputs"]["qk64.py"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        descriptor["dsa_qk_load_tile"]=128
        descriptor["dsa_pv_k_tile"]=32
        descriptor["dsa_qk_global_load_layout"]="contiguous-k-memory-boundary-v2"
        return descriptor
    describe._qk64_trial=True
    tir_package.describe_tir_package=describe
    spec=renderer.renderer_registry._specs["bufferized-tir"]
    renderer.renderer_registry._specs[spec.name]=PackageRendererSpec(spec.name,spec.kind,spec.matches,describe,spec.render)
    original=TritonTemplateRegistry.render
    def render(self,name,context):
        if name!="module.py.jinja":
            return original(self,name,context)
        calls=[c for c in context["render_calls"] if c["family"]=="dsa_paged_attention" and c["variant"]=="decode" and c["decode_schedule"]=="bf16_mma"]
        if not calls:
            return original(self,name,context)
        if len(calls)!=1 or calls[0]["head_dim"]!=512 or calls[0]["num_query_heads"]!=16 or not calls[0]["head_sharded"]:
            raise CodegenError("QK64 full trial requires its exact selected head-sharded attention")
        source=self.environment.loader.get_source(self.environment,KEY)[0]
        # Apply the accepted inner renderer first: its kernel preparation sets
        # the current template loader. The overlay is then used by its render.
        loader=self.environment.loader
        self.environment.loader=ChoiceLoader([DictLoader({KEY:transform(source)}),loader])
        try:
            return original(self,name,context)
        finally:
            self.environment.loader=loader
    TritonTemplateRegistry.render=render

"""Declare CTA-concurrent call domains before the normal SSA lifetime planner."""
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.passes.functions import function_nodes

KEY='tle.concurrent_call_groups'
_active=False

def declare(module):
    if KEY in module.metadata:
        raise IRVerificationError('CTA concurrency must be declared once before allocation',stage=module.stage)
    groups={}
    for name in ('decode_linear_attention_dense','decode_linear_attention_sparse.prefix'):
        function=module.function_map[name]
        nodes=function_nodes(module,function)
        calls=[node for node in nodes if node.op=='tir.call' and node.attrs.get('callee') in module.kernel_callable_map]
        selected=[]
        for family,projection in (('kda_convolution','self_attn_f_b_proj'),('kda_gated_recurrent_core','self_attn_g_b_proj')):
            matches=[]
            for index,node in enumerate(calls[:-1]):
                dispatch=fm.kernel_dispatch_of(module.kernel_callable_map[node.attrs['callee']])
                if dispatch is not None and dispatch.microkernel.family==family:
                    following=calls[index+1]
                    other=fm.kernel_dispatch_of(module.kernel_callable_map[following.attrs['callee']])
                    if other.microkernel.family!='dense_matmul' or projection not in following.id or not following.effect.is_pure:
                        raise IRVerificationError('Declared KDA concurrency requires its original independent projection',stage=module.stage)
                    matches.append((node,following))
            if len(matches)!=1:
                raise IRVerificationError('Expected one complete independent KDA/projection pair',stage=module.stage)
            first,second=matches[0]
            def depends(value,source,seen=None):
                seen=set() if seen is None else seen
                if value==source:return True
                if value in seen:return False
                seen.add(value)
                return any(depends(parent,source,seen) for parent in module.node_map[value].inputs)
            if depends(second.id,first.id) or depends(first.id,second.id):
                raise IRVerificationError('CTA concurrency cannot reorder a true SSA dependency',stage=module.stage)
            selected.append((first.id,second.id))
        groups[name]=tuple(selected)
    from dataclasses import replace
    return replace(module,metadata={**module.metadata,KEY:groups})

def extend_use_ends(module,function,nodes,original):
    result=dict(original)
    groups=module.metadata.get(KEY,{}).get(function.name,())
    if not groups:return result
    indexes={node.id:index for index,node in enumerate(nodes)}
    calls={node.id for node in nodes if node.op in ('tir.call','builtin.call','tir.kernel')}
    covered=set()
    for group in groups:
        if (not isinstance(group,(tuple,list)) or len(group)<2 or len(set(group))!=len(group)
                or any(call not in calls or call in covered for call in group)):
            raise IRVerificationError('CTA concurrent lifetime domains must name unique actual calls',stage=module.stage)
        order=[indexes[call] for call in group]
        if order!=sorted(order):
            raise IRVerificationError('CTA concurrent lifetime domain must retain original SSA order',stage=module.stage)
        covered.update(group)
        end=max(order)
        for call in group:result[call]=max(result.get(call,indexes[call]),end)
    return result

def activate():
    global _active
    if _active:return
    from triton.flagmega.passes.tir.bufferize import program_lifetimes
    previous=program_lifetimes.program_use_ends
    def use_ends(module,function,nodes):
        return extend_use_ends(module,function,nodes,previous(module,function,nodes))
    program_lifetimes.program_use_ends=use_ends
    _active=True

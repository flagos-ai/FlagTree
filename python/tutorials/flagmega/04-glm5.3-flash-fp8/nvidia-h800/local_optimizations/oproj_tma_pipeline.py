"""Keep ordered O-proj K16 updates, with four TMA slots and one pending MMA."""
import hashlib
from dataclasses import replace
from pathlib import Path
from triton.flagmega.errors import CodegenError

SCHEDULE='ordered_oproj_K256_four_slots_prefetch3_wait1_early_copy_v2'
BLOCK_K=256
SLOTS=4
PREFETCH=3
LEAD=3
WAIT_GROUP=1


def transform(source):
    # The accepted input-parallel renderer appends a separate template after
    # this complete loop. It owns different scratch and must remain untouched.
    boundary='{% endfor %}'
    first=source.find(boundary)
    if first<0:
        raise CodegenError('Original O-proj render-call loop is missing')
    first+=len(boundary)
    suffix=source[first:]
    source=source[:first]
    original=source
    checks={
        'tle.gpu.alloc([2, 64, {{ call.block_k }}]':1,
        'tle.gpu.alloc([2, 8, {{ call.block_k }}]':1,
        'tle.gpu.alloc_barriers(2,':2,
        '_fm_slot = (_fm_sequence_base + _fm_initial) % 2':2,
        '_fm_slot = _fm_sequence % 2':2,
        '_fm_phase = (_fm_sequence // 2) % 2':2,
        'tle.gpu.wgmma_wait(0, _fm_accumulator)':2,
        'if _fm_step + 2 <':2,
    }
    for text,count in checks.items():
        if source.count(text)!=count:
            raise CodegenError('Original ordered O-proj TMA template changed: '+text)
    slots='{{ call.oproj_slots | default(2) }}'
    lead='{{ call.oproj_lead | default(2) }}'
    wait='{{ call.oproj_wait_group | default(0) }}'
    source=source.replace('tle.gpu.alloc([2, 64,','tle.gpu.alloc(['+slots+', 64,')
    source=source.replace('tle.gpu.alloc([2, 8,','tle.gpu.alloc(['+slots+', 8,')
    source=source.replace('tle.gpu.alloc_barriers(2,','tle.gpu.alloc_barriers('+slots+',')
    source=source.replace('[2, call.part_k','[call.oproj_prefetch | default(2), call.part_k')
    source=source.replace('[2, call.accumulation_k','[call.oproj_prefetch | default(2), call.accumulation_k')
    source=source.replace('_fm_slot = (_fm_sequence_base + _fm_initial) % 2','_fm_slot = (_fm_sequence_base + _fm_initial) % '+slots)
    source=source.replace('_fm_slot = _fm_sequence % 2','_fm_slot = _fm_sequence % '+slots)
    source=source.replace('_fm_phase = (_fm_sequence // 2) % 2','_fm_phase = (_fm_sequence // '+slots+') % 2')
    source=source.replace('tle.gpu.wgmma_wait(0, _fm_accumulator)','tle.gpu.wgmma_wait('+wait+', _fm_accumulator)')
    lines=[];refill_indent=None
    for line in source.splitlines():
        stripped=line.lstrip();indent=len(line)-len(stripped)
        if refill_indent is not None and stripped and indent<=refill_indent:
            refill_indent=None
        if 'if _fm_step + 2 <' in line:
            refill_indent=indent
            lines.append(line.replace('_fm_step + 2','_fm_step + '+lead))
            lines.extend(['{% if call.oproj_wait_group | default(0) %}',
                          ' '*(indent+4)+'_fm_refill_slot = (_fm_sequence + '+lead+') % '+slots,
                          '{% endif %}'])
            continue
        if refill_indent is not None:
            line=line.replace('_fm_slot','{% if call.oproj_wait_group | default(0) %}_fm_refill_slot{% else %}_fm_slot{% endif %}')
            line=line.replace('_fm_step + 2','_fm_step + '+lead)
        if '_fm_result = tl.reshape(tl.gather(_fm_accumulator,' in line:
            # Finish the accumulator before its original split-result gather.
            lines.extend(['{% if call.oproj_wait_group | default(0) %}',
                          ' '*indent+'_fm_accumulator = tle.gpu.wgmma_wait(0, _fm_accumulator)',
                          '{% endif %}'])
        lines.append(line)
    # The first DMA now precedes the previous-output consumer poll. Its input
    # tensors and this CTA's shared arena are independent of that output window.
    # Sequence identity skips initial0 only in the first part/split, including
    # MLA's two original FP32 splits. Every later DMA remains in its old place.
    guarded=[];initial_indent=None
    for line in lines:
        stripped=line.lstrip();indent=len(line)-len(stripped)
        if initial_indent is not None and stripped and indent<=initial_indent:
            initial_indent=None
        if 'for _fm_initial in tl.static_range(' in line:
            guarded.append(line);initial_indent=indent
            guarded.extend(['{% if call.oproj_prefetch_before_counter | default(false) %}',
                            ' '*(indent+4)+'if _fm_initial != 0 or _fm_sequence_base != _fm_prefetched_sequence:',
                            '{% endif %}'])
            continue
        if initial_indent is not None:
            line='{% if call.oproj_prefetch_before_counter | default(false) %}    {% endif %}'+line
        guarded.append(line)
    changed='\n'.join(guarded)+'\n'
    selector='call.family == "partitioned_linear" and call.variant == "wgmma_local"'
    if original.count(selector)!=1 or changed.count(selector)!=1:
        raise CodegenError('Original physical O-proj template selector changed')
    # Select inside the existing loop so its imports, comments and whitespace
    # are emitted once. Unselected calls render their exact original source.
    loop='{% for call in render_calls | default(()) %}'
    if original.count(loop)!=1 or changed.count(loop)!=1:
        raise CodegenError('Original O-proj render-call loop changed')
    body_start=original.index(loop)+len(loop)
    changed_start=changed.index(loop)+len(loop)
    body_end=original.index(boundary)
    changed_end=changed.index(boundary)
    return (original[:body_start]
            +'{% if call.oproj_slots | default(none) %}'
            +changed[changed_start:changed_end]
            +'{% else %}'+original[body_start:body_end]+'{% endif %}'
            +original[body_end:]+suffix)


_active=False

def activate():
    global _active
    if _active:return
    from triton.flagmega.codegen.triton import renderer,tir_package
    previous=tir_package.describe_tir_package
    def describe(module):
        result=previous(module)
        calls=[c for c in result['render_calls'] if c.get('owner_matched_oproj')]
        if len(calls)!=3 or not str(module.metadata.get('numerical_contract','')).startswith('glm5-vllm-0.31-tp4/'):
            raise CodegenError('Ordered O-proj pipeline requires all three complete GLM TP4 families')
        for call in calls:
            if (call['family']!='partitioned_linear' or call['variant']!='wgmma_local'
                    or call['rows']!=4096 or call['tasks']!=64 or call['block_k']!=512
                    or call['local_shared_bytes']!=147456 or call['local_weight_offset']!=0
                    or call['local_activation_offset']!=131072
                    or call['accumulation_splits'] not in (1,2)):
                raise CodegenError('Original complete O-proj M64/N8/K512 arena contract changed')
            call.update(block_k=BLOCK_K,oproj_slots=SLOTS,oproj_prefetch=PREFETCH,
                        oproj_lead=LEAD,oproj_wait_group=WAIT_GROUP,
                        oproj_prefetch_before_counter=True)
        result['ordered_oproj_TMA_pipeline']=dict(schedule=SCHEDULE,
            selected=[c['call'] for c in calls],block_k=BLOCK_K,slots=SLOTS,
            prefetch=PREFETCH,lead=LEAD,pending_MMA_groups=WAIT_GROUP,
            first_independent_DMA_before_previous_consumed_acquire=True,
            first_part_split_initial0_copy_skipped_exactly_once=True,
            original_K16_updates_FP32_split_merges_and_BF16_part_boundaries=True,
            original_M4096_N8_math_and_TP4_publication=True,
            original147456_byte_arena_and_alias_offsets=True,
            recycle_proof='slot(i+3)%4=slot(i-1)%4; wait1 completes i-1 before copy; final wait0 before BF16')
        result['implementation_override']['inputs']['oproj_tma_pipeline.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return result
    tir_package.describe_tir_package=describe
    spec=renderer.renderer_registry._specs['bufferized-tir']
    renderer.renderer_registry._specs[spec.name]=replace(spec,describe=describe)
    _active=True

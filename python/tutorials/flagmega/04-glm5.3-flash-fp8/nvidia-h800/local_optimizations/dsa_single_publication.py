"""Publish completed DSA results before a single full TP4 rendezvous.

The first local grid retains Q's source lifetime: early key/gate publication
overwrites the same scratch that Q reads. After this grid, all three byte-exact
publish helpers finish before the merged device rendezvous. Q and weight boxing
then read disjoint, fully published source ranges before the original indexer.
"""
from pathlib import Path
import copy, hashlib
from triton.flagmega.errors import CodegenError

HERE = Path(__file__).resolve().parent

def transform_definition(definition):
    if definition['function'] != 'decode_mla_attention_sparse.prefix':
        return None
    events = definition['schedule']['consumer_events']
    qi = [i for i,e in enumerate(events) if e.get('family') == 'dense_matmul'
          and 'dsa_indexer_q.vectorized.compute' == e.get('call', '').split('decode_mla_attention_sparse_')[-1]]
    if len(qi) != 1:
        raise CodegenError('Expected one original concurrent DSA projection')
    i = qi[0]
    region = events[i:i+9]
    expected = [('tir.kernel_call', 'dense_matmul'), ('barrier', 'device'),
                ('tir.kernel_call', 'distributed_boxing'),
                ('tir.kernel_call', 'dense_matmul'), ('tir.kernel_call', 'dense_matmul'),
                ('tir.kernel_call', 'dense_matmul'), ('barrier', 'device'),
                ('tir.kernel_call', 'distributed_boxing'), ('barrier', 'block')]
    if [(e['kind'], e.get('family', e.get('scope'))) for e in region] != expected:
        raise CodegenError('Original full DSA publication/boxing region changed')
    for e, label in zip(region[3:6], ('key', 'gate', 'weights')):
        if f'dsa_indexer_{label}.vectorized.compute' not in e['call'] or not e['symbol'].endswith('__publish'):
            raise CodegenError('Expected original deferred byte publication helper')
    if 'reshard153' not in region[2]['call'] or 'reshard159' not in region[7]['call']:
        raise CodegenError('Original Q/weights boxing consumer changed')
    original = copy.deepcopy(region)
    local = copy.deepcopy(region[1])
    local.update(scope='grid', after=[region[0]['call']],
                 before=region[3]['call'], hazards=['READ->WRITE', 'WRITE->READ'],
                 ranges=[dict(kind='synchronization_range', storage='workspace',
                              physical_id='workspace:@decode_mla_attention_sparse.prefix',
                              offset=256, nbytes=3072, mode='read'),
                         dict(kind='synchronization_range', storage='persistent_window',
                              physical_id='persistent_window:dsa_deferred_results',
                              offset=0, nbytes=768, mode='read')])
    rendezvous = copy.deepcopy(region[1])
    rendezvous['after'] = [region[0]['call'], *[e['call'] for e in region[3:6]]]
    rendezvous['before'] = region[2]['call']
    rendezvous['hazards'] = sorted(set(region[1]['hazards']) | set(region[6]['hazards']))
    rendezvous['ranges'] = [*copy.deepcopy(region[1]['ranges']), *copy.deepcopy(region[6]['ranges'])]
    replacement = [region[0], local, *region[3:6], rendezvous, region[2], region[7], region[8]]
    events[i:i+9] = replacement
    return dict(function=definition['function'], original_region=original,
                replacement_region=copy.deepcopy(replacement),
                publication_device_barriers=1, original_publication_helpers_retained=True,
                original_q_complete_local_grid=True, original_block_completion_retained=True,
                q_source_lifetime=dict(offset=256, nbytes=3072),
                q_boxing_source=dict(offset=11520, nbytes=8192),
                q_boxing_destination=dict(offset=52480, nbytes=8192),
                key_destination=dict(offset=512, nbytes=256),
                gate_destination=dict(offset=256, nbytes=256),
                weights_destination=dict(offset=131328, nbytes=64))

def activate():
    from triton.flagmega.codegen.triton import renderer, tir_package
    from triton.flagmega.codegen.triton.registry import PackageRendererSpec
    previous = tir_package.describe_tir_package
    def describe(module):
        d = previous(module)
        definitions = list(d['device_functions'])
        for p in d['programs']:
            definitions.extend(p.get('device_functions', ()))
        definitions = list({id(v):v for v in definitions}.values())
        receipts = [r for f in definitions if (r := transform_definition(f)) is not None]
        if len(receipts) != 1 or len(d.get('owner_concurrent_dsa', ())) != 1:
            raise CodegenError('Requires the accepted full concurrent DSA implementation')
        resource, = [r for r in d['persistent_window_resources']['resources'] if r['name'] == 'dsa_deferred_results']
        actors = [k['symbol'] for k in d['kernels'] if k['entry_function'] == 'decode_mla_attention_sparse.prefix']
        if resource['nbytes'] != 768 or resource['programs'] != actors or len(actors) != 11:
            raise CodegenError('Original ordered DSA window actors changed')
        d['dsa_single_publication'] = receipts
        d['implementation_override']['inputs']['dsa_single_publication.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return d
    tir_package.describe_tir_package = describe
    spec = renderer.renderer_registry._specs['bufferized-tir']
    renderer.renderer_registry._specs[spec.name] = PackageRendererSpec(spec.name, spec.kind, spec.matches, describe, spec.render)


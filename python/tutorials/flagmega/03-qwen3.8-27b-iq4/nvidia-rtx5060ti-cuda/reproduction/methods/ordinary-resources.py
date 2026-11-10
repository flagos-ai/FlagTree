actual = result['resource_report']
assert len(actual) == 33 and [r['entry_id'] for r in actual] == list(range(33))
assert common_resources is None or actual == common_resources
common_resources = actual
resource_lineage = []
for r, e in zip(actual, entries):
    i = r['entry_id']
    assert i == e['entry_id'] and r['version'] == 3 and (r['record_bytes'] == 80) and (r['reserved'] == 0) and (r['device'] == 0)
    assert r['sm_count'] == 36 and [r['cc_major'], r['cc_minor']] == limits['compute_capability'] and (r['warp_size'] == e['warp_size'] == 32)
    assert r['block_threads'] == e['block_threads'] == (256 if i == 5 else 128) and r['dynamic_shared_bytes'] == e['dynamic_shared_bytes'] == 0
    assert r['block_threads'] <= min(r['max_threads_per_block'], limits['max_threads_per_block'])
    assert r['static_shared_bytes'] + r['dynamic_shared_bytes'] <= limits['max_shared_memory_bytes']
    assert type(r['registers_per_thread']) is int and r['registers_per_thread'] > 0 and (r['registers_per_thread'] * r['block_threads'] <= limits['max_registers_per_sm']) and (r['resident_blocks_per_sm'] > 0)
    blocks = math.prod(e['grid'])
    capacity = r['sm_count'] * r['resident_blocks_per_sm']
    if e['launch_kind'] == 'cooperative':
        assert r['cooperative_supported'] and blocks <= capacity
    resource_lineage.append(dict(r, grid=e['grid'], launch_kind=e['launch_kind'], grid_blocks=blocks, resident_capacity=capacity, original_admission_passed=True))
glu_rows = []
api = read(F / 'O-native-api-candidate.json')
compiled = checked(api['static_resources'])
assert compiled['library'] == side['artifact']['library']
for parent_id in [10, 26]:
    join = next((d for d in mapping['definitions'] if d['parent'] == parent_id))
    i = join['successor']
    e = entries[i]
    assert join['disposition'] == 'same_semantic_calls_actual_replanned_ABI' and e['grid'] == [4352, 1, 1]
    resource = compiled['functions'][e['symbol']]
    assert resource['SHARED'] == resource['LOCAL'] == resource['STACK'] == 0
    assert actual[i]['registers_per_thread'] == resource['REG']
    assert [actual[i][k] for k in ['static_shared_bytes', 'dynamic_shared_bytes', 'local_bytes']] == [0, 0, 0]
    assert actual[i]['resident_blocks_per_sm'] > 0
    capacity_from_registers = limits['max_registers_per_sm'] // (actual[i]['registers_per_thread'] * actual[i]['block_threads'])
    assert actual[i]['resident_blocks_per_sm'] <= capacity_from_registers
    glu_rows.append(dict(parent_entry=parent_id, current_entry=i, actual_registers=actual[i]['registers_per_thread']))
assert [r['current_entry'] for r in glu_rows] == [10, 25]
gdn = actual[5]
gdn_entry = entries[5]
gdn_compiled = compiled['functions'][gdn_entry['symbol']]
assert gdn_entry['grid'] == [48, 1, 1] and gdn_entry['scratch_nbytes'] == 52 and (gdn['block_threads'] == 256)
assert gdn_compiled['LOCAL'] == gdn_compiled['STACK'] == 0 and gdn['local_bytes'] == 0
assert gdn['registers_per_thread'] == gdn_compiled['REG']
assert 48 <= gdn['sm_count'] * gdn['resident_blocks_per_sm']
assert gdn['resident_blocks_per_sm'] <= limits['max_registers_per_sm'] // (gdn['registers_per_thread'] * gdn['block_threads'])
assert actual[1]['local_bytes'] == actual[14]['local_bytes'] == 0

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Eager diagnostic lifetime and durable acknowledgement; no CUDA import."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import struct
import threading

U64_MAX=(1<<64)-1
def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def loaded_identity(artifact,manifest,library):
    root=Path(artifact).resolve()
    def sha(relative):
        p=(root/relative).resolve()
        if not p.is_relative_to(root):raise ValueError('Artifact identity path escapes root.')
        return hashlib.sha256(p.read_bytes()).hexdigest()
    if json.loads((root/'artifact.json').read_text())!=manifest:raise ValueError('Loaded artifact manifest changed.')
    c=manifest['codegen'];p=c['operator_profile']
    if Path(library).resolve()!=(root/c['library']).resolve():raise ValueError('Wrong loaded diagnostic library.')
    result={'artifact_manifest_sha256':sha('artifact.json'),'source_sha256':sha(c['source']),
            'library_sha256':sha(c['library']),'final_IR_sha256':sha(manifest['final_ir']),
            'recipe_sha256':digest(manifest['selections']),'model_ABI_sha256':digest(manifest['abi']),
            'profile_contract_sha256':digest(p),'implementation_sha256':p['implementation_sha256'],
            'target':manifest['target'],'grid':c['grid'],'block_threads':c['block_threads'],
            'max_registers':c.get('build_options', {}).get('max_registers_per_thread')}
    if result['source_sha256']!=c['source_sha256'] or result['library_sha256']!=c['library_sha256']:
        raise ValueError('Diagnostic source/library declared hash mismatch.')
    return result


def _validate_v2_contract(contract, *, grid=None, block_threads=None):
    events, ctas = contract.get('event_count'), contract.get('grid_total')
    if (type(events) is not int or events <= 0 or type(ctas) is not int or ctas <= 0
            or ctas > (1 << 31) - 1 or events * ctas * 32 > (1 << 63) - 1):
        raise ValueError('Invalid diagnostic event/grid bounds.')
    launch_grid = contract.get('launch_grid')
    threads = contract.get('block_threads')
    if (launch_grid != [ctas, 1, 1] or any(type(n) is not int for n in launch_grid)
            or type(threads) is not int or not 0 < threads <= 1024 or threads % 32
            or contract.get('warp_size') != 32):
        raise ValueError('Invalid flat complete-warp diagnostic launch.')
    if grid is not None and list(grid) != launch_grid:
        raise ValueError('Diagnostic grid differs from the loaded manifest.')
    if block_threads is not None and block_threads != threads:
        raise ValueError('Diagnostic block threads differ from the loaded manifest.')
    expected = {
        'schema': 'flagmega.cpp-operator-profile/v2',
        'launch_abi': 'flagmega_launch_profile/v1',
        'extra_trace_shape_u64': [events, ctas, 4],
        'scalar_strides': [ctas * 4, 4, 1],
        'record_bytes': 32, 'trace_alignment_bytes': 8,
        'trace_nbytes': events * ctas * 32, 'counter_bits': 64,
        'trace_fields': ['start_tick', 'end_tick', 'epoch', 'tag'],
        'runtime_mode': 'diagnostic_eager', 'accepted_timing_eligible': False,
    }
    if any(not isinstance(contract.get(key), list)
           or any(type(n) is not int for n in contract[key])
           for key in ('extra_trace_shape_u64', 'scalar_strides')):
        raise ValueError('Diagnostic shape/strides must contain integer dimensions.')
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError('Invalid diagnostic v2 shape/strides/bytes/ABI.')
    inventories = ('site_inventory', 'sync_site_inventory', 'invocation_manifest', 'sync_manifest')
    for name in inventories:
        rows = contract.get(name)
        if not isinstance(rows, list) or digest(rows) != contract.get(name + '_sha256'):
            raise ValueError('Diagnostic inventory missing or its digest differs.')
    sites = contract['site_inventory']
    sync_sites = contract['sync_site_inventory']
    site_map = {}
    for kind, rows in (('operator', sites), ('schedule_barrier', sync_sites)):
        for row in rows:
            identifier, count = row.get('site_id'), row.get('expected_invocations_per_token')
            if (not isinstance(identifier, str) or not identifier or identifier in site_map
                    or type(count) is not int or count <= 0):
                raise ValueError('Invalid/duplicate diagnostic site or multiplicity.')
            if kind == 'schedule_barrier' and row.get('scope') not in ('grid', 'cta'):
                raise ValueError('Unknown diagnostic synchronization scope.')
            site_map[identifier] = (kind, count, row.get('scope'))
    if not sites:
        raise ValueError('Diagnostic trace has no operators.')
    counts = {name: 0 for name in site_map}
    indices = set()
    for kind, rows in (('operator', contract['invocation_manifest']),
                       ('schedule_barrier', contract['sync_manifest'])):
        for row in rows:
            index = row.get('event_index')
            identifier = row.get('site_id')
            if (type(index) is not int or not 0 <= index < events or index in indices
                    or row.get('kind') != kind or row.get('expected_tag') != str(index)
                    or row.get('CTA_slot_formula') != f'event_index*{ctas}+cta_index'
                    or identifier not in site_map or site_map[identifier][0] != kind
                    or not isinstance(row.get('call_path'), str) or not row['call_path']):
                raise ValueError('Invalid diagnostic event/tag/site census.')
            if kind == 'schedule_barrier' and row.get('scope') != site_map[identifier][2]:
                raise ValueError('Diagnostic synchronization scope differs from its site.')
            indices.add(index)
            counts[identifier] += 1
    if len(indices) != events or any(counts[name] != site_map[name][1] for name in counts):
        raise ValueError('Diagnostic event coverage/multiplicity differs from its static census.')
    if contract.get('extra_CTA_barriers_per_CTA_per_token') != 2 * len(contract['invocation_manifest']):
        raise ValueError('Diagnostic instrumentation barrier census differs.')


class ProfileSession:
    def __init__(self,contract, *, grid=None, block_threads=None):
        self.version = 2 if contract.get('profile_mode') == 'cta-globaltimer-v2' else 1
        if self.version == 2:
            _validate_v2_contract(contract, grid=grid, block_threads=block_threads)
        elif (contract.get('profile_mode')!='cta-globaltimer-v1' or contract.get('launch_abi')!='flagmega_launch_profile/v1'
                or contract.get('extra_trace_shape_u64')!=[1978,96,4] or contract.get('trace_nbytes')!=6076416):
            raise ValueError('Invalid diagnostic trace ABI.')
        self.contract=copy.deepcopy(contract);self.epoch=0;self.stream=None;self.pending=False;self.lock=threading.Lock()
        self.failed_raw=None;self.completed=False;self.known_complete=True;self.poisoned=False
        self.failure=None;self.failure_archived=False;self.raw=None
    def reserve(self,stream,*,pointer,capacity,capturing=False):
        with self.lock:
            if capturing:raise ValueError('Profiling v1 is eager only.')
            if self.pending or self.poisoned:raise ValueError('Pending/failed profile cannot reuse trace.')
            if self.stream is not None and self.stream!=stream:raise ValueError('Runtime belongs to one stream.')
            if pointer<=0 or pointer%8 or capacity<self.contract['trace_nbytes']:raise ValueError('Invalid trace alignment/capacity.')
            if self.epoch==U64_MAX:raise ValueError('Profile epoch exhausted.')
            self.stream=stream;self.epoch+=1;self.pending=True;self.completed=False;self.raw=None
            return self.epoch
    def complete(self,raw):
        if not self.pending or not self.known_complete:raise ValueError('No safely completed launch.')
        self.raw=raw
        if len(raw)!=self.contract['trace_nbytes']:raise ValueError('Incomplete trace buffer.')
        ctas = self.contract['grid_total'] if self.version == 2 else 96
        previous_ends = [0] * ctas
        for i,(start,end,epoch,tag) in enumerate(struct.iter_unpack('<QQQQ',raw)):
            if epoch!=self.epoch or tag!=i//ctas or end<start:raise ValueError('Stale/incomplete/nonmonotone slot.')
            if self.version == 2 and start < previous_ends[i % ctas]:
                raise ValueError('Diagnostic per-CTA event order is nonmonotone.')
            previous_ends[i % ctas] = end
        self.completed=True
        return raw
    def fail(self,error,*,raw=None,stage='unknown',failure_sink=None):
        self.poisoned=True;self.failed_raw=raw;self.raw=raw
        self.failure={'error_type':type(error).__name__,'error':str(error),'stage':stage,
                      'epoch':str(self.epoch),'stream':str(self.stream),
                      'known_GPU_completion':self.known_complete,'raw_available':raw is not None,
                      'raw_nbytes':None if raw is None else len(raw),
                      'raw_sha256':None if raw is None else hashlib.sha256(raw).hexdigest(),
                      'archive_error':None,'durable_archive':None}
        if callable(failure_sink):
            try:
                ack=failure_sink(self.epoch,raw,error,dict(self.failure))
                self._check_ack(ack,failed=True)
                self.failure['durable_archive']=ack;self.failure_archived=True
            except BaseException as archival_error:
                self.failure['archive_error']=f'{type(archival_error).__name__}: {archival_error}'
                if hasattr(error,'add_note'):error.add_note('Failure archival also failed: '+self.failure['archive_error'])
    def capture(self,enqueue,synchronize,copy_raw,*,failure_sink=None):
        raw=None;stage='enqueue';self.known_complete=False
        try:
            enqueue()
            stage='synchronize';synchronize();self.known_complete=True
            stage='copy';raw=copy_raw()
            stage='validate';return self.complete(raw)
        except BaseException as error:
            if not self.known_complete:
                try:synchronize();self.known_complete=True
                except BaseException as sync_error:
                    if hasattr(error,'add_note'):error.add_note('Owned stream completion failed: '+str(sync_error))
            if self.known_complete and raw is None:
                try:raw=copy_raw()
                except BaseException as copy_error:
                    if hasattr(error,'add_note'):error.add_note('Failure trace copy unavailable: '+str(copy_error))
            self.fail(error,raw=raw,stage=stage,failure_sink=failure_sink)
            raise
    def _check_ack(self,ack,*,failed=False):
        if not isinstance(ack,dict) or set(ack)!={'path','nbytes','sha256'}:raise ValueError('Durable index pin required.')
        p=Path(ack['path']);data=p.read_bytes()
        if len(data)!=ack['nbytes'] or hashlib.sha256(data).hexdigest()!=ack['sha256']:raise ValueError('Durable index bytes changed.')
        doc=json.loads(data)
        if doc.get('schema')!='flagmega.provisional-profile-archive/v1':raise ValueError('Wrong archival acknowledgement schema.')
        if failed:
            if doc.get('status')!='failed':raise ValueError('Failure archive status missing.')
        elif doc.get('status') not in ('capture_pending','complete_raw') or doc.get('failed_frames'):raise ValueError('Failed archive cannot acknowledge success.')
        rows=doc.get('failed_frames',[]) if failed else doc.get('frames',[])
        row=next((r for r in rows if r.get('launch_epoch')==str(self.epoch)),None)
        if row is None or row.get('known_GPU_completion')!=self.known_complete:raise ValueError('Archive epoch/completion mismatch.')
        if self.raw is not None:
            bp=row.get('binary')
            if not isinstance(bp,dict):raise ValueError('Raw bytes not durably retained.')
            data=Path(bp['path']).read_bytes()
            if len(data)!=len(self.raw) or hashlib.sha256(data).hexdigest()!=hashlib.sha256(self.raw).hexdigest():raise ValueError('Archived raw differs.')
    def consume(self,epoch,ack):
        with self.lock:
            if self.poisoned or not self.pending or not self.completed or epoch!=self.epoch:raise ValueError('No valid matching completed trace.')
            self._check_ack(ack)
            self.pending=False;self.completed=False;self.raw=None
    @property
    def safe_release(self):
        return self.known_complete and (not self.pending or self.poisoned and self.failure_archived)

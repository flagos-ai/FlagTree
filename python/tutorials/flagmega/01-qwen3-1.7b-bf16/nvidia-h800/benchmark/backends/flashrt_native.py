#!/usr/bin/env python3
"""Matched public FlashRT-derived frontend, with CPU-only source preflight.

Timed calls use prepare_generation ->63 decode_step calls, original GPU feedback,
native sampling and one-token pinned D2H. NUM/export is a separate untimed mode.
No native module, Torch or device API is imported by preflight.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import traceback
from device_identity import (prepare_identity, verify_completion_identity,
                             verify_tensor_devices, source_contract)
from reference_rounds import (WORK, common_arguments, check_arguments, reference_prompts,
    controls, bind_gpu, StepTimer, step_arrays, validate_run, aggregate, sha, write_new)

INPUTS = WORK / '.local/agents/dispatch/inspect_flashrt_successful_native_modules/runnable-inputs-001.json'
INPUTS_SHA = 'db41cf05fc4adffafb92f6c71b6009ef907c2c527938a59ca7a3ea6dcbc1a812'
FRONTEND = 'Qwen3_1_7BTorchFrontendH800'
RUNTIME = 'Qwen3DenseBf16Runtime'


def pin(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def checked(pin_record):
    path = Path(pin_record['path'])
    size = pin_record.get('bytes', pin_record.get('nbytes'))
    if sha(path) != pin_record['sha256'] or (size is not None and path.stat().st_size != size):
        raise ValueError('Pinned native/source bytes changed: ' + str(path))
    return path


def source_binding(native_inputs=INPUTS):
    if sha(native_inputs) != INPUTS_SHA:
        raise ValueError('Require the independently inspected current native runtime inputs')
    data = json.loads(Path(native_inputs).read_text())
    manifest = json.loads(checked(data['source_manifest']).read_text())
    sources = [record['working_source'] for record in manifest['files']]
    for record in sources + data['actual_native_module_pins']:
        checked(record)
    root = Path(data['PYTHONPATH']).resolve()
    provenance = json.loads(checked(data['native_provenance']).read_text())
    original = provenance['git_identities']['frozen_original']
    # These are the unchanged imported Python routines/package initialization,
    # not another model route. Bind their bytes to the pinned upstream objects.
    imports = ['flash_rt/__init__.py','flash_rt/_extensions.py','flash_rt/frontends/__init__.py',
        'flash_rt/frontends/torch/__init__.py','flash_rt/models/__init__.py','flash_rt/models/qwen3/__init__.py',
        'flash_rt/frontends/torch/_qwen3_4b_bf16_runtime.py','flash_rt/frontends/torch/_qwen3_4b_bf16_weights.py',
        'flash_rt/models/qwen3/prefill_bf16.py','flash_rt/models/qwen3/_bf16_attention.py']
    for relative in imports:
        frozen = subprocess.check_output(['git','-C',original['root'],'show',original['commit']+':'+relative])
        path = root / relative
        import hashlib
        if sha(path)!=hashlib.sha256(frozen).hexdigest():raise ValueError('Imported upstream Python source changed: '+relative)
        sources.append(pin(path))
    public = root / 'flash_rt/frontends/torch/qwen3_1_7b_h800.py'
    runtime = root / 'flash_rt/frontends/torch/_qwen3_dense_bf16_runtime.py'
    api = source_contract(public, FRONTEND, methods=['__init__','prepare_generation','decode_step','generate_ids','reset','clear_graphs'],
                          calls=['prepare_generation','decode_step','generate_ids','reset','clear_graphs'])
    state = source_contract(runtime, RUNTIME, methods=['prepare_generation','decode_step','generate_ids','_reset_buffers','_sample'],
                            attributes=['_graphs','_token','_out_ids','_k','_v'], calls=['replay','commit_decode','check_current_logits','zero_','copy_'])
    torch_graphs = Path(importlib.metadata.distribution('torch').locate_file('torch/cuda/graphs.py'))
    graph = source_contract(torch_graphs, 'CUDAGraph', methods=['replay'], calls=['replay'])
    if data['module'] != 'flash_rt.frontends.torch.qwen3_1_7b_h800' or data['class'] != FRONTEND:
        raise ValueError('Public native frontend identity changed')
    return {'inputs': pin(native_inputs), 'root': str(root), 'module': data['module'], 'class': data['class'],
        'checkpoint': data['checkpoint'], 'label': data['derived_reference_label'],
        'manifest': data['source_manifest'], 'files': [pin(record['path']) for record in sources],
        'native_modules': data['actual_native_module_pins'], 'required_core_symbols': data['required_core_symbols'],
        'FA2_symbol': data['required_attention_public_symbol'], 'constructor': data['constructor_policy'],
        'AST_contracts': [api,state,graph], 'torch_graph_source': pin(torch_graphs),
        'private_observations': 'Pinned runtime._stream/_graphs/_pointer_identity/_k/_v/_token are diagnostics, not a public KV export API',
        'GPU_proof': 'native load/attributes/full NUM/KV/eager-graph/stream/replay/isolation/perturbation pending'}


def loaded_module(module, expected):
    path = Path(module.__file__).resolve()
    if path != Path(expected['path']).resolve() or sha(path) != expected['sha256']:
        raise RuntimeError('Loaded module differs from pinned native origin: ' + module.__name__)
    return {'module': module.__name__, **pin(path)}


def state_record(runtime, prompt_length, outputs):
    state = runtime.state
    actual = (state.cache_length, state.output_count, state.last_consumed_position)
    expected = (prompt_length + outputs - 1, outputs, prompt_length + outputs - 2)
    if actual != expected:
        raise RuntimeError('Actual native cache/output/position state differs: ' + repr(actual))
    return {'cache_length': actual[0], 'output_count': actual[1], 'last_consumed_position': actual[2]}


class NativeReplayProbe:
    """Scoped graph-instance replay observation; no model/class or math edits.

    Each original bound method is invoked exactly once and its original instance
    attribute state is restored. Observer validation/dispatch overhead is included
    in HOST and can create DEVICE launch idle; perturbation is not inferred here.
    """
    def __init__(self, runtime, torch, timer, prompt_length):
        self.runtime, self.torch, self.timer = runtime, torch, timer
        self.stream = runtime._stream
        self.pointer_identity = runtime._pointer_identity
        self.family = runtime._graphs
        self.active, self.samples, self.installed = False, [], []
        self.prompt_length = prompt_length
        try:
            for slot in range(1,64):
                key = runtime.graph_key(prompt_length + slot - 1, slot)
                graph = self.family.get(key)
                if type(graph) is not torch.cuda.CUDAGraph or not hasattr(graph, '__dict__'):
                    raise RuntimeError('Native prepared Torch CUDAGraph instance observer ABI unavailable')
                present, previous, original = 'replay' in vars(graph), vars(graph).get('replay'), graph.replay
                if not callable(original) or getattr(original, '__self__', None) is not graph:
                    raise RuntimeError('Require original native graph bound replay method')
                def replay(*args, _key=key, _graph=graph, _original=original, **kwargs):
                    if self.active:
                        self.calls += 1
                        if (_key != self.expected_key or _graph is not self.expected_graph or self.calls != 1 or
                                self.torch.cuda.current_stream(self.runtime.device).cuda_stream != self.stream.cuda_stream):
                            raise RuntimeError('Actual native replay key/graph/count/stream differs')
                    return _original(*args, **kwargs)
                graph.replay = replay
                self.installed.append((key,graph,present,previous,original,replay))
        except BaseException as error:
            try:self.close()
            except BaseException as cleanup:error.observer_cleanup_error=str(cleanup)
            raise

    def begin(self):
        self.active, self.calls, self.copy_complete = True, 0, False
        if self.timer.start is not None:
            self.timer.start.record(self.stream)
        state = self.runtime.state
        self.position, self.slot = state.cache_length, state.output_count
        self.expected_key = self.runtime.graph_key(self.position, self.slot)
        self.expected_graph = self.family.get(self.expected_key)
        if (self.runtime._graphs is not self.family or self.runtime._pointer_identity != self.pointer_identity or
                self.position != self.prompt_length + self.slot - 1 or not 1 <= self.slot < 64 or
                self.expected_graph is None):
            raise RuntimeError('Prepared graph/state/pointers changed; timed capture/fallback is forbidden')

    def copied_token(self):
        self.copy_complete = True
        if self.timer.end is not None:
            self.timer.end.record(self.stream)

    def finish(self):
        if (self.calls != 1 or not self.copy_complete or self.runtime._pointer_identity != self.pointer_identity or
                self.runtime._graphs is not self.family or self.family.get(self.expected_key) is not self.expected_graph or
                self.runtime.state.cache_length != self.position + 1 or self.runtime.state.output_count != self.slot + 1):
            raise RuntimeError('Native replay/one-token D2H/state advancement observation incomplete')
        self.samples.append({'position':self.position,'output_slot':self.slot,'replay_calls':self.calls,
            'graph_id':id(self.expected_graph),'pointer_identity':list(self.pointer_identity),
            'actual_stream_handle':self.stream.cuda_stream,'event_end_after_one_token_D2H':True})

    def close(self):
        self.active = False
        errors = []
        for _,graph,present,previous,original,observer in reversed(self.installed):
            if vars(graph).get('replay') is not observer:
                errors.append('Native graph replay observer was replaced')
            if present: graph.replay = previous
            elif 'replay' in vars(graph): del graph.replay
            restored = graph.replay
            if (getattr(restored,'__self__',None),getattr(restored,'__func__',None)) != (
                    getattr(original,'__self__',None),getattr(original,'__func__',None)):
                errors.append('Native graph replay identity not restored')
        self.installed.clear()
        if errors: raise RuntimeError('; '.join(errors))


def timed_request(frontend, prompt, timer, torch, cpu_token):
    runtime = frontend.runtime
    frontend.reset(); runtime._stream.synchronize()
    frontend.prepare_generation(prompt, max_new_tokens=64)
    runtime._stream.synchronize()
    state_record(runtime,len(prompt),1)
    generated = [int(runtime.output_ids[0].item())]  # first output untimed
    steps = []
    probe = NativeReplayProbe(runtime,torch,timer,len(prompt))
    primary = None
    try:
        with torch.cuda.stream(runtime._stream):
            for _ in range(63):
                def operation():
                    frontend.decode_step()  # no external/reference/CPU token input
                    cpu_token.copy_(runtime.output_ids[-1:],non_blocking=True)
                    probe.copied_token()
                    return cpu_token
                token, sample = timer.measure(operation,lambda value:int(value.item()),probe=probe)
                generated.append(token); steps.append(sample)
        final = state_record(runtime,len(prompt),64)
        if runtime.output_ids.detach().cpu().tolist() != generated or int(runtime._token.item()) != generated[-1]:
            raise RuntimeError('Delivered outputs/device feedback differ from native output slots')
        return {'prompt_tokens':prompt,'token_ids':generated,'consumed_token_ids':prompt+generated[:-1],
            'native_state':final,'native_graph_coverage':probe.samples,
            'history_attribution':'consumed IDs derived from native delivered slots; decode_step has no token argument; source-pinned native _sample owns feedback; no separate per-input GPU value capture',
            **step_arrays(steps)}
    except BaseException as error:
        primary=error
        raise
    finally:
        try:probe.close()
        except BaseException as cleanup:
            if primary is None:raise
            primary.observer_cleanup_error=str(cleanup)


def tensor_admission(frontend, torch):
    runtime, dims = frontend.runtime, frontend.dims
    geometry = (dims.num_layers,dims.hidden,dims.intermediate,dims.vocab_size,dims.num_q_heads,dims.num_kv_heads,dims.head_dim)
    if geometry != (28,2048,6144,151936,16,8,128): raise RuntimeError('Full native Qwen3-1.7B geometry differs')
    weights = [runtime.weights.embed,runtime.weights.lm_head,runtime.weights.final_norm]
    weights += [t for layer in runtime.weights.layers for t in layer.values()]
    if len(runtime.weights.layers)!=28 or any(t.dtype!=torch.bfloat16 for t in weights):
        raise RuntimeError('All28 native BF16 weight tensors required')
    for tensor in (runtime._k,runtime._v):
        if tuple(tensor.shape)!=(28,2112,8,128) or tensor.dtype!=torch.bfloat16 or not tensor.is_contiguous():
            raise RuntimeError('All28 full-capacity BF16 native KV layout differs')
    if (runtime._k.untyped_storage().data_ptr()==runtime._v.untyped_storage().data_ptr() or
            runtime.weights.embed.data_ptr()!=runtime.weights.lm_head.data_ptr() or
            tuple(runtime.logits.shape)!=(1,151936) or runtime.logits.dtype!=torch.bfloat16):
        raise RuntimeError('Independent K/V, tied full BF16 head/logits contract differs')
    return {'geometry':geometry,'tensor_devices':verify_tensor_devices(weights+[runtime._k,runtime._v,runtime.logits,runtime._token,runtime._out_ids]),
        'KV_shape':[28,2112,8,128],'KV_dtype':'bfloat16','independent_KV_storage':True,'tied_head_alias':True}


def snapshot_NUM(frontend, torch, prompt_length):
    runtime = frontend.runtime
    state = state_record(runtime,prompt_length,64)
    runtime._stream.synchronize()
    if not runtime.last_generation or runtime.last_generation.get('all_output_logits_checked_finite') is not True:
        raise RuntimeError('NUM requires public generate_ids(check_logits=True) for every complete row')
    snapshot = {'K':runtime._k.detach().cpu().clone(),'V':runtime._v.detach().cpu().clone(),
        'output_ids':runtime.output_ids.detach().cpu().clone(),'last_logits':runtime.logits.detach().cpu().clone(),'state':state}
    length = state['cache_length']; layers=[]
    for layer in range(28):
        k,v = snapshot['K'][layer],snapshot['V'][layer]
        finite = bool(torch.isfinite(k).all().item() and torch.isfinite(v).all().item())
        written = bool(torch.count_nonzero(k[:length]).item() and torch.count_nonzero(v[:length]).item())
        tail_zero = bool(torch.count_nonzero(k[length:]).item()==0 and torch.count_nonzero(v[length:]).item()==0)
        if not (finite and written and tail_zero): raise RuntimeError('Actual native KV finite/prefix/reset-tail failed at layer'+str(layer))
        layers.append({'layer':layer,'finite':finite,'written_prefix_nonzero':written,'unwritten_tail_zero':tail_zero})
    return snapshot, {'native_state':state,'layers':layers,'all64_full_vocab_rows_finite':True,
        'finite_attribution':'Public generate_ids(check_logits=True), source-pinned full151936 BF16 row check at y0 and every63 decode outputs',
        'KV_attribution':'Read-only private pinned runtime._k/_v; actual full tensors exported outside timing; cross-framework KV value comparison pending'}


def NUM_request(frontend,eager,prompt,expected,torch,raw_path=None,export=False):
    outputs = frontend.generate_ids(prompt,max_new_tokens=64,check_logits=True).tolist()
    if len(outputs)!=64: raise RuntimeError('Public fixed-count output64 differs')
    snapshot, proof = snapshot_NUM(frontend,torch,len(prompt))
    eager_outputs = eager.generate_ids(prompt,max_new_tokens=64,check_logits=True).tolist()
    eager_snapshot, eager_proof = snapshot_NUM(eager,torch,len(prompt))
    parity={'own64':outputs==eager_outputs,'all28_full_KV':bool(torch.equal(snapshot['K'],eager_snapshot['K']) and torch.equal(snapshot['V'],eager_snapshot['V'])),
            'final_full_logits':bool(torch.equal(snapshot['last_logits'],eager_snapshot['last_logits']))}
    mismatch = next((i for i,(a,b) in enumerate(zip(outputs,expected)) if a!=b),None)
    result = {'prompt_tokens':prompt,'token_ids':outputs,'consumed_token_ids':prompt+outputs[:-1],**proof,
        'NUM':{'first10_decode_indices':[1,11],'first10_exact':outputs[1:11]==expected[1:11],
               'first_output_exact':outputs[0]==expected[0],'all64_exact':mismatch is None,'first_mismatch':mismatch},
        'native_eager_graph_parity':parity,'eager_token_ids':eager_outputs,'eager_state':eager_proof['native_state']}
    if raw_path is not None and (export or not all(parity.values())):
        with Path(raw_path).open('xb') as stream: torch.save(snapshot,stream)
        result['raw_snapshot'] = pin(raw_path)
        if not all(parity.values()):
            eager_path=Path(raw_path).with_suffix('.eager-failed.pt')
            with eager_path.open('xb') as stream:torch.save(eager_snapshot,stream)
            result['eager_failed_raw_snapshot']=pin(eager_path)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    common_arguments(parser)
    parser.add_argument('--native-inputs',type=Path,default=INPUTS)
    parser.add_argument('--mode',choices=['timing','NUM'],default='timing')
    args = parser.parse_args(argv); check_arguments(args,backend='flashrt')
    if args.mode=='NUM' and args.cohort!='ordinary_no_events': parser.error('NUM exports are untimed and use no harness events')
    binding = source_binding(args.native_inputs)
    checkpoint, inputs, reference = reference_prompts(args)
    if str(args.checkpoint.resolve())!=binding['checkpoint']: raise ValueError('Pinned native checkpoint differs from independent reference')
    report = {'schema':'flagmega.model-measurement/v1' if args.mode=='timing' else 'flagmega.native-NUM-admission/v1',
        'backend':'flashrt','label':binding['label'],'status':'CPU-preflight','measurement_id':args.measurement_id,
        'cohort':args.cohort,'mode':args.mode,'round_index':args.round_index,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'script':pin(__file__),'checkpoint':checkpoint,'reference':reference,'controls':controls(args),'native_source':binding,
        'GPU_initialized':False,'scenarios':[],
        'workload':{'batch':1,'concurrency':1,'dtype':'bfloat16','kv_dtype':'bfloat16','layers':28,'vocab':151936,
            'prompt_lengths':args.prompt_lengths,'decode_tokens':64,'sampling':'native BF16 argmax lowest-index ties; fixed64 includes EOS'},
        'boundary':{'host':'63 public decode_step+original graph+greedy/own-device feedback+pinned one-token D2H+stream sync+host delivery intervals',
                    'device':'preallocated events begin before per-step observer/key lookup and public full28+LMhead+nativegreedy/state, end after one-token pinned D2H; includes host launch idle; not pure graph replay'},
        'observer_policy':'Scoped native graph-instance replay observer forwarding original once and restoring attributes; validation/dispatch overhead retained; actual graph compatibility and perturbation pending',
        'NUM_export_policy':'Separate NUM mode checks all64 full-vocab rows, native graph/eager parity and exports actual all28 K/V outside timing; timing mode does not run these diagnostics'}
    if args.preflight_only: write_new(args.output,report); return 0
    frontends=[];primary=None;cleanup_errors=[]
    try:
        identity = prepare_identity(args,'flashrt')
        import torch
        bind_gpu(torch,args.physical_gpu_uuid,identity=identity)
        module = importlib.import_module(binding['module']);public_source=next(p for p in binding['files'] if p['path'].endswith('/qwen3_1_7b_h800.py'))
        loaded = [loaded_module(module,public_source)]
        for name,expected in zip(('flash_rt.flash_rt_kernels','flash_rt.flash_rt_fa2'),binding['native_modules']):
            native = importlib.import_module(name);loaded.append(loaded_module(native,expected))
            symbols = binding['required_core_symbols'] if name.endswith('kernels') else [binding['FA2_symbol']]
            if any(not callable(getattr(native,s,None)) for s in symbols): raise RuntimeError('Actual native public callable ABI differs: '+name)
        cls = getattr(module,binding['class'])
        def load(graphs):
            frontend=cls(str(args.checkpoint),device='cuda:0',max_seq=2112,cuda_graphs=graphs,graph_cache_size=63,
                         rope_inv_freq='fp32',autotune_gemms=False)
            frontends.append(frontend);tensor_admission(frontend,torch);return frontend
        frontend=load(True)
        report.update(GPU_initialized=True,loaded_source=loaded,device_identity=identity,
                      native_tensor_devices=tensor_admission(frontend,torch),identity_after_native_load=verify_completion_identity(torch,identity))
        if args.mode=='NUM':
            eager=load(False);raw_dir=args.output.with_suffix('.raw');raw_dir.mkdir(exist_ok=False)
            for length in args.prompt_lengths:
                prompt=inputs[length]['prompt'];scenario={'prompt_length':length,'runs':[]};report['scenarios'].append(scenario)
                for _ in range(2): frontend.generate_ids(prompt,max_new_tokens=64)
                for repeat in range(3):
                    run=NUM_request(frontend,eager,prompt,inputs[length]['expected'],torch,raw_dir/f'prompt{length}-repeat{repeat}-all28.pt',export=repeat==0)
                    run['repeat_index']=repeat;scenario['runs'].append(run)
                    if not all(run['native_eager_graph_parity'].values()):raise RuntimeError('Actual eager/graph own64/fullKV/final-logit parity differs')
        else:
            with torch.cuda.stream(frontend.runtime._stream):
                timer=StepTimer(torch,args.cohort,synchronize=frontend.runtime._stream.synchronize)
            cpu_token=torch.empty(1,dtype=torch.int64,device='cpu',pin_memory=True)
            for length in args.prompt_lengths:
                prompt=inputs[length]['prompt']
                for _ in range(2): timed_request(frontend,prompt,timer,torch,cpu_token)
                scenario={'prompt_length':length,'runs':[]};report['scenarios'].append(scenario)
                for repeat in range(3):
                    run=timed_request(frontend,prompt,timer,torch,cpu_token);run['repeat_index']=repeat
                    validate_run(run,prompt,inputs[length]['expected']);scenario['runs'].append(run)
                aggregate(scenario)
        report.update(status='measured' if args.mode=='timing' else 'NUM_completed',
            NUM_passed=all(r['NUM']['first10_exact'] for s in report['scenarios'] for r in s['runs']),
            all64_exact=all(r['NUM']['all64_exact'] for s in report['scenarios'] for r in s['runs']),
            identity_at_completion=verify_completion_identity(torch,identity),
            acceptance='pending independent derived-reference NUM/state/stream/graph/isolation/timing/rotation review; no win claim')
        if source_binding(args.native_inputs)!=binding: raise RuntimeError('Native/frontend source changed during cohort')
    except BaseException as error:
        primary=error;report.update(error=str(error),traceback=traceback.format_exc(),observer_cleanup_error=getattr(error,'observer_cleanup_error',None))
    finally:
        for frontend in reversed(frontends):
            try: frontend.clear_graphs();frontend.reset();frontend.runtime._stream.synchronize()
            except BaseException as error: cleanup_errors.append(str(error))
    report['cleanup_errors']=cleanup_errors
    if primary or cleanup_errors:
        report['status']='failed'
        if primary is None: report['error']='Native frontend cleanup failed: '+'; '.join(cleanup_errors)
        write_new(args.output.with_suffix('.failed.json'),report)
        if primary: raise primary
        raise RuntimeError(report['error'])
    write_new(args.output,report);return 0


if __name__=='__main__': raise SystemExit(main())

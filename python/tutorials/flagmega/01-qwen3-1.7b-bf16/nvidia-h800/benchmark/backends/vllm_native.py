#!/usr/bin/env python3
"""Unchanged native vLLM own-history requests with separate HOST/DEVICE cohorts.

DEVICE events bind the synchronous in-process worker entry and native one-token
D2H completion queue, not the outer engine.step stream or pure graph replay.
CPU preflight never imports vLLM/Torch or initializes a GPU.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import traceback
import time
from device_identity import (prepare_identity, verify_completion_identity, verify_tensor_devices,
                             source_contract, loaded_origin, normalize_uuid)

from reference_rounds import (common_arguments, check_arguments, reference_prompts, controls,
    bind_gpu, StepTimer, step_arrays, validate_run, aggregate, sha, write_new, gpu_inventory)

SOURCE = Path('/root/repos/vllm')
PINNED_COMMIT = '493bd8323b64ad778f802b41ddd3109740910bc8'


def source_binding():
    head = subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
    if head != PINNED_COMMIT:
        raise ValueError('Use the pinned native vLLM493 source')
    paths = ['vllm/v1/worker/gpu_model_runner.py','vllm/executor/uniproc_executor.py',
             'vllm/worker/worker_base.py','vllm/v1/engine/llm_engine.py','vllm/v1/engine/core.py',
             'vllm/v1/sample/sampler.py','vllm/engine/arg_utils.py',
             'vllm/platforms/interface.py','vllm/platforms/cuda.py',
             'vllm/config/compilation.py','vllm/forward_context.py','vllm/v1/cudagraph_dispatcher.py',
             'vllm/compilation/cuda_graph.py','vllm/v1/worker/gpu_input_batch.py',
             'vllm/v1/worker/block_table.py','vllm/v1/core/kv_cache_utils.py']
    runner = (SOURCE/paths[0]).read_text()
    # Fail before any GPU init if the private synchronous binding has changed.
    for clause in ('def execute_model(', 'def _bookkeeping_sync(', 'def _to_list(',
                   'pinned.copy_(sampled_token_ids, non_blocking=True)',
                   'self.transfer_event.record()', 'self.transfer_event.synchronize()',
                   'req_state.output_token_ids.extend(sampled_ids)'):
        if clause not in runner:
            raise ValueError('Native worker/D2H/history ABI changed: '+clause)
    contracts = [source_contract(SOURCE/'vllm/platforms/interface.py', 'Platform',
                    methods=['device_id_to_physical_device_id'], calls=['int']),
        source_contract(SOURCE/'vllm/platforms/cuda.py', 'NvmlCudaPlatform',
                    methods=['get_device_uuid','get_device_capability'], calls=['device_id_to_physical_device_id']),
        source_contract(SOURCE/paths[0], 'GPUModelRunner', methods=['load_model','execute_model'],
                    attributes=['nncase_config','nncase_executor','cudagraph_dispatcher'],
                    calls=['CUDAGraphWrapper','dispatch','set_forward_context']),
        source_contract(SOURCE/'vllm/v1/cudagraph_dispatcher.py', 'CudagraphDispatcher',
                    methods=['initialize_cudagraph_keys','dispatch'], attributes=['non_uniform','cudagraph_keys']),
        source_contract(SOURCE/'vllm/compilation/cuda_graph.py', 'CUDAGraphWrapper',
                    methods=['__call__'], attributes=['batch_descriptor','concrete_cudagraph_entries','output'], calls=['replay'])]
    torch_graphs=Path(importlib.metadata.distribution('torch').locate_file('torch/cuda/graphs.py'))
    contracts.append(source_contract(torch_graphs,'CUDAGraph',methods=['replay'],calls=['replay']))
    return {'root':str(SOURCE),'commit':head,'version':importlib.metadata.version('vllm'), 'AST_contracts':contracts,
        'files':[{'path':str(SOURCE/p),'sha256':sha(SOURCE/p)} for p in paths]+[{'path':str(torch_graphs),'sha256':sha(torch_graphs)}],
        'dirty':subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'],text=True).strip(),
        'worker_path':'engine.model_executor.driver_worker.worker.model_runner',
        'device_path':'execute_model entry -> _prepare_inputs/fullmodel/compute_logits/_sample -> _to_list pinned.copy_ -> transfer_event.record proxy end',
        'gpu_path_proven':'CPU source contract only; actual runtime stream/coverage admission pending'}


class NativeGraphProbe:
    """Observe original native dispatch and captured graph replay, never __call__.

    Instance replay/dispatch bindings are restored even after a failed interval.
    Observer checks add declared Python overhead to both refreshed cohorts.
    CPU protocol tests of this observer do not establish real CUDA replay.
    """
    def __init__(self, runner, full_mode, descriptor, wrapper_class, graph_class, context):
        self.runner, self.full, self.descriptor, self.context = runner, full_mode, descriptor, context
        self.dispatcher, self.wrapper = runner.cudagraph_dispatcher, runner.model
        self.key = descriptor(1, False)
        if type(self.wrapper) is not wrapper_class or self.wrapper.runtime_mode != full_mode:
            raise RuntimeError('Require original native FULL wrapper')
        if runner.vllm_config.nncase_config is not None or runner.nncase_executor is not None:
            raise RuntimeError('Native comparison forbids alternate nncase engine')
        self.entries = self.wrapper.concrete_cudagraph_entries
        self.entry = self.entries.get(self.key)
        if self.entry is None or self.entry.batch_descriptor != self.key or self.entry.cudagraph is None:
            raise RuntimeError('Original selected FULL entry must already be captured')
        self.graph = self.entry.cudagraph
        if type(self.graph) is not graph_class: raise RuntimeError('Require actual native CUDA graph class')
        self.active = False; self.samples = []; self.installed = []
        self._readiness()
        self.original_dispatch = self.dispatcher.dispatch; self.original_replay = self.graph.replay
        try:
            self._install(self.dispatcher, 'dispatch', self._dispatch)
            self._install(self.graph, 'replay', self._replay)
        except BaseException:
            self.close(); raise

    def _install(self, obj, name, value):
        present = name in vars(obj); original = vars(obj).get(name)
        setattr(obj, name, value); self.installed.append((obj, name, present, original))

    def _readiness(self):
        d = self.dispatcher
        if (self.runner.model is not self.wrapper or self.runner.cudagraph_dispatcher is not d or
            d.cudagraph_mode != self.full or not d.keys_initialized or
            d.compilation_config.cudagraph_mode != self.full or d.compilation_config.cudagraph_capture_sizes != [1] or
            d.cudagraph_keys[self.full] != {self.key} or self.wrapper.runtime_mode != self.full or
            self.wrapper.concrete_cudagraph_entries is not self.entries or self.entries.get(self.key) is not self.entry or
            self.entry.batch_descriptor != self.key or self.entry.cudagraph is not self.graph):
            raise RuntimeError('Native configured FULL keys/wrapper/entry/graph identity changed')

    def _dispatch(self, *args, **kwargs):
        if not self.active: return self.original_dispatch(*args, **kwargs)
        started = time.perf_counter_ns()
        self._readiness(); self.dispatch_calls += 1
        inp = args[0] if args else kwargs['batch_descriptor']
        if self.dispatch_calls != 1 or inp != self.descriptor(1, True):
            raise RuntimeError('Timed native INPUT must be exactly uniform B1 one-token decode')
        result = self.original_dispatch(*args, **kwargs)
        mode, selected = result
        if mode != self.full or selected != inp.non_uniform or selected != self.key:
            raise RuntimeError('Native returned mode/SELECTED key differs from registered mixed FULL key')
        self.selected = selected
        self.observer_ns += time.perf_counter_ns()-started
        return result

    def _replay(self, *args, **kwargs):
        if not self.active: return self.original_replay(*args, **kwargs)
        started = time.perf_counter_ns()
        self._readiness(); self.replay_calls += 1
        ctx = self.context()
        if (self.dispatch_calls != 1 or self.replay_calls != 1 or self.selected != self.key or
            ctx.batch_descriptor != self.selected or ctx.cudagraph_runtime_mode != self.full):
            raise RuntimeError('Actual native replay/context does not use the pre-existing SELECTED FULL entry')
        # Preserve original replay arguments, return value, and entry output.
        self.observer_ns += time.perf_counter_ns()-started
        return self.original_replay(*args, **kwargs)

    def begin(self):
        if self.active: raise RuntimeError('Overlapping native graph observation')
        self._readiness(); self.dispatch_calls = self.replay_calls = self.observer_ns = 0
        self.selected = None; self.active = True

    def finish(self):
        try:
            self._readiness()
            if (self.dispatch_calls, self.replay_calls) != (1, 1):
                raise RuntimeError('Timed native interval lacks exactly one FULL dispatch/replay')
            self.samples.append({'INPUT': {'num_tokens':1,'uniform_decode':True},
                'SELECTED': {'num_tokens':1,'uniform_decode':False}, 'mode':'FULL',
                'dispatch_calls':1,'replay_calls':1,'wrapper_id':id(self.wrapper),
                'entry_id':id(self.entry),'graph_id':id(self.graph), 'observer_checks_ns':self.observer_ns})
        finally: self.active = False

    def close(self):
        for obj, name, present, value in reversed(self.installed):
            if present: setattr(obj, name, value)
            else: delattr(obj, name)
        self.installed.clear(); self.active = False


class WorkerProbe:
    """Instrument one actual synchronous native worker without changing math.

    Preserve the runner's original transfer event and instance/class method identity.
    The event end queues immediately after native pinned D2H, before native sync.
    """
    def __init__(self, runner, timer):
        if timer.start is None or runner.use_async_scheduling:
            raise ValueError('Worker probe requires event cohort and synchronous native scheduler')
        self.runner,self.timer=runner,timer
        self.active=False; self.samples=[]
        self.original_event=runner.transfer_event
        self.had_instance_method='execute_model' in vars(runner)
        self.original_instance_method=vars(runner).get('execute_model')
        self.original_method=runner.execute_model
        owner=self
        class TransferProxy:
            def record(self,*args,**kwargs):
                result=owner.original_event.record(*args,**kwargs)
                if owner.active:
                    stream=owner.timer.torch.cuda.current_stream()
                    owner.ends+=1
                    if owner.ends!=1 or owner.starts!=1 or stream.cuda_stream!=owner.stream_id:
                        raise RuntimeError('Native sampled D2H and worker start lack same-stream coverage')
                    owner.timer.end.record(stream)
                return result
            def __getattr__(self,name):return getattr(owner.original_event,name)
        self.proxy=TransferProxy()
        def execute(*args,**kwargs):
            if owner.active:
                scheduler=args[0] if args else kwargs['scheduler_output']
                if scheduler.total_num_scheduled_tokens!=1:
                    raise RuntimeError('Timed native worker must execute exactly one B1 decode token')
                stream=owner.timer.torch.cuda.current_stream()
                owner.starts+=1; owner.stream_id=stream.cuda_stream
                if owner.starts!=1:raise RuntimeError('Multiple native workers in one decode interval')
                owner.timer.start.record(stream)
            return owner.original_method(*args,**kwargs)
        runner.transfer_event=self.proxy
        runner.execute_model=execute

    def begin(self):
        if self.active:raise RuntimeError('Overlapping native worker event reuse')
        self.starts=self.ends=0;self.stream_id=None;self.active=True

    def finish(self):
        if (self.starts,self.ends)!=(1,1):raise RuntimeError('Missing native full-worker/D2H event coverage')
        self.samples.append({'worker_calls':self.starts,'native_D2H_end_calls':self.ends,'cuda_stream':self.stream_id})
        self.active=False

    def close(self):
        self.runner.transfer_event=self.original_event
        if self.had_instance_method:self.runner.execute_model=self.original_instance_method
        else:del self.runner.execute_model
        self.active=False


def one_output(result):
    if len(result)!=1 or len(result[0].outputs)!=1:
        raise RuntimeError('Expected B1 native token delivery')
    return result[0].outputs[0].token_ids[-1]


def request(engine, runner, prompt, sampling, rid, timer, probe=None, graph_probe=None):
    timer.synchronize()
    engine.add_request(rid,{'prompt_token_ids':prompt},sampling)
    first=engine.step();timer.synchronize()
    generated=[int(one_output(first))];steps=[];result=first
    graph_start=len(graph_probe.samples) if graph_probe else None
    for _ in range(63):
        if not engine.has_unfinished_requests():raise RuntimeError('Native request ended before own64')
        if graph_probe: graph_probe.begin()
        try:
            (result,token),span=timer.measure(engine.step,lambda value:(value,int(one_output(value))),probe=probe)
            if graph_probe: graph_probe.finish()
        finally:
            if graph_probe: graph_probe.active=False
        generated.append(token);steps.append(span)
        if list(result[0].outputs[0].token_ids)!=generated:
            raise RuntimeError('Native delivered cumulative history differs from own trajectory')
    if engine.has_unfinished_requests():raise RuntimeError('Native request exceeds own64')
    # Read native persistent input/history outside timing. It still retains the
    # finished request until the next execute_model update removes it.
    state=runner.requests[rid];idx=runner.input_batch.req_id_to_index[rid]
    actual=runner.input_batch.token_ids_cpu[idx,:len(prompt)+64].tolist()
    if state.output_token_ids!=generated or actual!=prompt+generated:
        raise RuntimeError('Native cached history differs from delivered history')
    graph_rows=graph_probe.samples[graph_start:] if graph_probe else []
    if graph_probe and len(graph_rows)!=63:raise RuntimeError('Native trajectory lacks full63 graph observations')
    return {'prompt_tokens':prompt,'token_ids':generated,'consumed_token_ids':prompt+generated[:-1],
        'native_graph_coverage':graph_rows,
        'native_history':{'output_token_ids':list(state.output_token_ids),'persistent_token_ids':actual,
            'num_tokens':int(runner.input_batch.num_tokens[idx]),
            'num_computed_tokens_at_last_worker_entry':int(state.num_computed_tokens),
            'KV_scope':'no full28 KV copy-out proof from this CPU binding; future admission pending'},**step_arrays(steps)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);common_arguments(parser)
    parser.add_argument('--attention-backend',default='FLASH_ATTN')
    parser.set_defaults(gpu_selector_kind='nvml-index')
    args=parser.parse_args(argv);check_arguments(args,'vllm')
    report={'schema':'flagmega.model-measurement/v1','backend':'vllm','label':'native-vllm',
        'measurement_id':args.measurement_id,'status':'preflight','cohort':args.cohort,'round_index':args.round_index,
        'timestamp_utc':datetime.now(timezone.utc).isoformat(),'script':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},
        'controls':controls(args),'scenarios':[],'GPU_initialized':False,
        'workload':{'batch':1,'concurrency':1,'dtype':'bfloat16','kv_dtype':'bfloat16','layers':28,'vocab':151936,
            'prompt_lengths':args.prompt_lengths,'decode_tokens':64,'prefix_caching':False,'speculative_decoding':False,'sampling':'native greedy'},
        'boundary':{'host':'63 synchronous engine.step scheduler+fullmodel+nativegreedy+native pinned D2H+sync+delivery intervals',
            'device':'preallocated actual worker entry through native pinned one-token D2H enqueue; includes worker preparation/launch idle; excludes outer scheduler/output processing; NOT pure graph replay'},
        'native_config':{'cuda_graph':'FULL capture_sizes=[1], Inductor level3','context':3072,'block_size':256,
            'gpu_blocks':16,'gpu_memory_utilization':.3,'async_scheduling':False,'TP':1}}
    engine=probe=graph_probe=None
    try:
        checkpoint,inputs,reference=reference_prompts(args);binding=source_binding()
        report.update(checkpoint=checkpoint,reference=reference,native_source=binding,version=binding['version'])
        if args.preflight_only:write_new(args.output,report);return 0
        identity = prepare_identity(args, 'vllm')
        os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_USE_V1='1',VLLM_ATTENTION_BACKEND=args.attention_backend)
        import torch
        from vllm import EngineArgs, SamplingParams
        from vllm.v1.engine.llm_engine import LLMEngine
        from vllm.config.compilation import CUDAGraphMode
        from vllm.forward_context import BatchDescriptor, get_forward_context
        from vllm.compilation.cuda_graph import CUDAGraphWrapper
        from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
        from vllm.platforms import current_platform
        bind_gpu(torch,args.physical_gpu_uuid,identity=identity);torch.set_num_threads(1)
        if normalize_uuid(current_platform.get_device_uuid(0)) != normalize_uuid(args.physical_gpu_uuid):
            raise RuntimeError('Actual native platform UUID differs from leased GPU')
        report.update(device_identity=identity,loaded_source=[loaded_origin(c,binding) for c in
            (CUDAGraphWrapper,CudagraphDispatcher,type(current_platform),torch.cuda.CUDAGraph)])
        report.update(GPU_initialized=True,gpu_before=gpu_inventory(),visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
        config=EngineArgs(model=str(args.checkpoint),dtype='bfloat16',kv_cache_dtype='auto',tensor_parallel_size=1,
            max_model_len=3072,max_num_seqs=1,max_num_batched_tokens=3072,block_size=256,num_gpu_blocks_override=16,
            gpu_memory_utilization=.3,enable_prefix_caching=False,enable_chunked_prefill=False,disable_log_stats=True,
            async_scheduling=False,compilation_config={'level':3,'cudagraph_mode':'FULL','cudagraph_capture_sizes':[1]},seed=0)
        engine=LLMEngine.from_engine_args(config)
        runner=engine.model_executor.driver_worker.worker.model_runner
        if runner.use_async_scheduling or runner.model_config.hf_config.num_hidden_layers!=28 or runner.model_config.hf_config.vocab_size!=151936:
            raise RuntimeError('Native full28/synchronous worker contract failed')
        if runner.dtype!=torch.bfloat16 or runner.kv_cache_dtype!=torch.bfloat16:raise RuntimeError('Native BF16 model/KV dtype required')
        report['loaded_source'].append(loaded_origin(type(runner),binding))
        if len(runner.kv_caches)!=28:raise RuntimeError('Require all28 actual native KV tensors')
        report['native_tensor_devices'] = verify_tensor_devices(list(runner.model.parameters())+list(runner.kv_caches))
        report['identity_after_native_load'] = verify_completion_identity(torch,identity)
        graph_probe=NativeGraphProbe(runner,CUDAGraphMode.FULL,BatchDescriptor,CUDAGraphWrapper,
                                    torch.cuda.CUDAGraph,get_forward_context)
        report['observer_policy']='Restored dispatch/replay observers in both cohorts; Python validation overhead included in HOST; no HOST harness events; 63 observations/request; actual GPU proof pending review'
        timer=StepTimer(torch,args.cohort)
        if args.cohort=='event_instrumented':probe=WorkerProbe(runner,timer)
        sampling=SamplingParams(temperature=0,max_tokens=64,ignore_eos=True);counter=0
        for length in args.prompt_lengths:
            prompt=inputs[length]['prompt']
            for _ in range(2):
                counter+=1;request(engine,runner,prompt,sampling,str(counter),timer,probe,graph_probe)
            scenario={'prompt_length':length,'runs':[]};report['scenarios'].append(scenario)
            for repeat in range(3):
                counter+=1;run=request(engine,runner,prompt,sampling,str(counter),timer,probe,graph_probe);run['repeat_index']=repeat
                validate_run(run,prompt,inputs[length]['expected']);scenario['runs'].append(run)
            aggregate(scenario)
        if source_binding()!=binding:raise RuntimeError('Native source changed during measurement')
        if normalize_uuid(current_platform.get_device_uuid(0))!=normalize_uuid(args.physical_gpu_uuid):
            raise RuntimeError('Native platform UUID changed during trajectory')
        report.update(status='measured',NUM_passed=all(r['NUM']['first10_exact'] for s in report['scenarios'] for r in s['runs']),
            all64_exact=all(r['NUM']['all64_exact'] for s in report['scenarios'] for r in s['runs']),
            worker_event_coverage=probe.samples if probe else None,gpu_after=gpu_inventory(),
            native_graph_coverage=graph_probe.samples, identity_at_completion=verify_completion_identity(torch,identity),
            acceptance='pending independent actual stream/kernel/state/isolation/rotated-round review')
        write_new(args.output,report);return 0
    except BaseException as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc());write_new(args.output.with_suffix('.failed.json'),report);raise
    finally:
        if probe:probe.close()
        if graph_probe:graph_probe.close()
        if engine:engine.engine_core.shutdown()


if __name__=='__main__':raise SystemExit(main())

#!/usr/bin/env python3
"""Official SGLang low-level own-history decode, separate HOST/DEVICE cohorts.

CPU preflight imports no backend or Torch. Native forward+sampling and KV state
are unchanged; each of63 timed steps queues one-token D2H before event end and
completes delivery before the next step. Events are preallocated and reused.
"""
from __future__ import annotations
import argparse
from array import array
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import traceback
import time
from device_identity import (prepare_identity, verify_completion_identity, verify_tensor_devices,
                             source_contract, loaded_origin)
from reference_rounds import (common_arguments, check_arguments, reference_prompts, controls,
    bind_gpu, StepTimer, step_arrays, validate_run, aggregate, sha, write_new, gpu_inventory)


def source_binding():
    package=Path(importlib.metadata.distribution('sglang').locate_file('sglang'))
    paths=['bench_one_batch.py','srt/model_executor/model_runner.py','srt/model_executor/cuda_graph_runner.py',
        'srt/managers/schedule_batch.py','srt/layers/sampler.py','srt/server_args.py','srt/model_executor/forward_batch_info.py',
        'srt/mem_cache/memory_pool.py','srt/mem_cache/allocator/base.py']
    text=(package/paths[0]).read_text()
    for clause in ('class _TorchBenchRunner:', 'def decode(input_token_ids, batch, model_runner):',
                   'batch.input_ids = input_token_ids.to(torch.int64)', 'batch.prepare_for_decode()',
                   'model_runner.forward(forward_batch).logits_output', 'model_runner.sample(logits_output, forward_batch)'):
        if clause not in text:raise ValueError('Official native bench binding changed: '+clause)
    contracts=[source_contract(package/'srt/model_executor/model_runner.py','ModelRunner',
        methods=['init_device_graphs','_forward_raw'], attributes=['graph_runner','can_run'],calls=['replay','ModelRunnerOutput']),
        source_contract(package/'srt/model_executor/model_runner.py','ModelRunnerOutput',fields=['can_run_graph','logits_output']),
        source_contract(package/'srt/model_executor/cuda_graph_runner.py','CudaGraphRunner',
            methods=['can_run','replay'],attributes=['graphs'],calls=['replay']),
        source_contract(package/'bench_one_batch.py','_TorchBenchRunner',methods=['decode','extend'],
            attributes=['torch_runner'],calls=['decode','extend']),
        source_contract(package/'srt/mem_cache/allocator/base.py','BaseTokenToKVPoolAllocator',methods=['get_kvcache']),
        source_contract(package/'srt/mem_cache/memory_pool.py','MHATokenToKVPool',methods=['get_kv_buffer'])]
    torch_graphs=Path(importlib.metadata.distribution('torch').locate_file('torch/cuda/graphs.py'))
    contracts.append(source_contract(torch_graphs,'CUDAGraph',methods=['replay'],calls=['replay']))
    return {'root':str(package),'version':importlib.metadata.version('sglang'),'AST_contracts':contracts,
        'files':[{'path':str(package/p),'sha256':sha(package/p)} for p in paths]+[{'path':str(torch_graphs),'sha256':sha(torch_graphs)}],
        'native_path':'_TorchBenchRunner.decode -> prepare_for_decode -> forward(native CUDA graph) -> sample -> adapter pinned D2H',
        'gpu_path_proven':'CPU contract only; native runtime graph/stream/fullKV checks pending'}


def graph_readiness(native, graph_class, captured_graph_class=None):
    graph = native.graph_runner
    if type(graph) is not graph_class or graph is None:
        raise RuntimeError('Require the pinned native CUDA graph_runner class')
    if graph.capture_bs != [1] or 1 not in graph.graphs or graph.graphs[1] is None:
        raise RuntimeError('Native CUDA graph_runner must have pre-captured batch1')
    if not callable(graph.can_run) or not callable(graph.replay):
        raise RuntimeError('Native CUDA graph_runner readiness/replay ABI changed')
    if captured_graph_class is not None and type(graph.graphs[1]) is not captured_graph_class:
        raise RuntimeError('Require original captured native Torch CUDA graph; alternate graph route is unqualified')
    return graph


class NativeForwardProbe:
    """Restored observation of native.forward; no new CUDA events or sync."""
    def __init__(self, native, graph_class, output_class, captured_graph_class=None):
        self.native=native;self.graph=graph_readiness(native,graph_class,captured_graph_class);self.output_class=output_class
        self.captured_graph=self.graph.graphs[1]
        self.active=False;self.samples=[]
        self.present='forward' in vars(native);self.instance=vars(native).get('forward');self.original=native.forward
        native.forward=self.forward

    def forward(self,*args,**kwargs):
        if not self.active:return self.original(*args,**kwargs)
        started=time.perf_counter_ns();self.calls+=1
        batch=args[0] if args else kwargs['forward_batch']
        if (self.native.graph_runner is not self.graph or self.graph.graphs.get(1) is not self.captured_graph or self.calls!=1 or
            not batch.forward_mode.is_decode() or batch.batch_size!=1 or batch.input_ids.numel()!=1):
            raise RuntimeError('Timed native SGLang forward must be one original B1 DECODE')
        self.observer_ns+=time.perf_counter_ns()-started
        output=self.original(*args,**kwargs)
        started=time.perf_counter_ns()
        if type(output) is not self.output_class or output.can_run_graph is not True:
            raise RuntimeError('Actual native SGLang decode did not return graph-replay execution flag')
        if self.native.graph_runner is not self.graph or self.graph.graphs.get(1) is not self.captured_graph:
            raise RuntimeError('Native graph_runner/captured graph changed during forward')
        self.batch=batch;self.observer_ns+=time.perf_counter_ns()-started
        return output

    def begin(self):
        if self.active:raise RuntimeError('Overlapping native SGLang forward observation')
        self.calls=self.observer_ns=0;self.batch=None;self.active=True

    def finish(self):
        try:
            if self.calls!=1 or self.batch is None:raise RuntimeError('Missing/multiple native SGLang decode forward')
            # Already synchronized by StepTimer; GPU state reads and serialized
            # observation rows occur outside both HOST/DEVICE intervals.
            seq=self.batch.seq_lens_cpu if self.batch.seq_lens_cpu is not None else self.batch.seq_lens
            self.samples.append({'forward_calls':1,'mode':'DECODE','batch_size':1,'input_tokens':1,
                'actual_seq_lens':seq.tolist(),'can_run_graph':True,'graph_runner_id':id(self.graph),
                'observer_checks_ns':self.observer_ns})
        finally:self.active=False

    def close(self):
        if self.present:self.native.forward=self.instance
        else:del self.native.forward
        self.active=False


def request(runner, prompt, make_req, timer, torch, cpu_token, graph_probe=None):
    runner.clear();req=make_req(prompt)
    timer.synchronize()
    next_ids,_,batch=runner.extend([req]);timer.synchronize()
    generated=[int(next_ids.item())];steps=[]
    graph_start=len(graph_probe.samples) if graph_probe else None
    for _ in range(63):
        def operation():
            # This is the native runner's previous output tensor, never a
            # reference ID or a host-provided forced token.
            sampled,_=runner.decode(next_ids,batch)
            cpu_token.copy_(sampled,non_blocking=True)
            return sampled,cpu_token
        if graph_probe:graph_probe.begin()
        try:
            (next_ids,token),span=timer.measure(operation,lambda value:(value[0],int(value[1].item())))
            if graph_probe:graph_probe.finish()
        finally:
            if graph_probe:graph_probe.active=False
        generated.append(token);steps.append(span)
    position=int(batch.seq_lens.item())
    if position!=len(prompt)+63:raise RuntimeError('Actual native batch/KV progression differs from own64')
    graph_rows=graph_probe.samples[graph_start:] if graph_probe else []
    if graph_probe and (len(graph_rows)!=63 or [row['actual_seq_lens'] for row in graph_rows]!=[[len(prompt)+i] for i in range(1,64)]):
        raise RuntimeError('Actual native graph sequence progression differs from full63 trajectory')
    runner.cleanup(batch)
    return {'prompt_tokens':prompt,'token_ids':generated,'consumed_token_ids':prompt+generated[:-1],
        'native_graph_coverage':graph_rows,
        'native_state':{'actual_batch_seq_len':position,'request_pool_index':int(batch.req_pool_indices.item()),
            'KV_scope':'full native28 runtime pool; full28 KV copy-out admission pending'},**step_arrays(steps)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);common_arguments(parser)
    parser.add_argument('--attention-backend',default='flashinfer')
    args=parser.parse_args(argv);check_arguments(args,'sglang')
    report={'schema':'flagmega.model-measurement/v1','backend':'sglang','label':'native-sglang',
        'status':'preflight','measurement_id':args.measurement_id,'cohort':args.cohort,'round_index':args.round_index,
        'timestamp_utc':datetime.now(timezone.utc).isoformat(),'script':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},
        'controls':controls(args),'GPU_initialized':False,'scenarios':[],
        'workload':{'batch':1,'concurrency':1,'layers':28,'vocab':151936,'dtype':'bfloat16','kv_dtype':'bfloat16',
            'prompt_lengths':args.prompt_lengths,'decode_tokens':64,'sampling':'native greedy','prefix_caching':False,'speculative_decoding':False},
        'boundary':{'host':'63 official batch prepare+fullmodel+sampler+pinned D2H+sync+host delivery intervals',
            'device':'preallocated events around native batch prepare/fullmodel/sampler and one-token pinned D2H; includes in-window launch idle; NOT pure graph replay'},
        'native_config':{'cuda_graph':'enabled capture_bs=[1]','attention_backend':args.attention_backend,
            'TP':1,'context':3072,'mem_fraction_static':.3,'disable_radix_cache':True,'disable_overlap_schedule':True}}
    runner=graph_probe=None
    try:
        checkpoint,inputs,reference=reference_prompts(args);binding=source_binding()
        report.update(checkpoint=checkpoint,reference=reference,native_source=binding,version=binding['version'])
        if args.preflight_only:write_new(args.output,report);return 0
        identity=prepare_identity(args,'sglang')
        import torch
        from sglang.bench_one_batch import load_model, _TorchBenchRunner
        from sglang.srt.model_executor.model_runner import ModelRunner, ModelRunnerOutput
        from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
        from sglang.srt.entrypoints.engine import _set_envs_and_config
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.srt.server_args import PortArgs,ServerArgs
        bind_gpu(torch,args.physical_gpu_uuid,identity=identity);torch.set_num_threads(1)
        report.update(device_identity=identity,loaded_source=[loaded_origin(c,binding) for c in
            (_TorchBenchRunner,ModelRunner,ModelRunnerOutput,CudaGraphRunner,torch.cuda.CUDAGraph)])
        server=ServerArgs(model_path=str(args.checkpoint),tokenizer_path=str(args.checkpoint),dtype='bfloat16',kv_cache_dtype='bfloat16',
            tp_size=1,context_length=3072,max_running_requests=1,mem_fraction_static=.3,disable_radix_cache=True,
            disable_overlap_schedule=True,disable_cuda_graph=False,cuda_graph_bs=[1],cuda_graph_max_bs=1,
            attention_backend=args.attention_backend,random_seed=0)
        _set_envs_and_config(server)
        report.update(GPU_initialized=True,gpu_before=gpu_inventory(),visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
        runner,_=load_model(server,PortArgs.init_new(server),gpu_id=0,tp_rank=0)
        native=runner.torch_runner
        if type(runner) is not _TorchBenchRunner or type(native) is not ModelRunner:
            raise RuntimeError('Require original native TorchBenchRunner/ModelRunner classes')
        if (native.model_config.num_hidden_layers,native.model_config.vocab_size)!=(28,151936) or native.dtype!=torch.bfloat16 or native.kv_cache_dtype!=torch.bfloat16:
            raise RuntimeError('Full28 BF16 model/KV contract failed')
        graph_probe=NativeForwardProbe(native,CudaGraphRunner,ModelRunnerOutput,torch.cuda.CUDAGraph)
        pool=native.token_to_kv_pool_allocator.get_kvcache()
        report['loaded_source'].append(loaded_origin(type(pool),binding))
        kv=[t for layer in range(28) for t in pool.get_kv_buffer(layer)]
        report['native_tensor_devices']=verify_tensor_devices(list(native.model.parameters())+kv)
        report['identity_after_native_load']=verify_completion_identity(torch,identity)
        report['observer_policy']='Restored native.forward execution-flag observer in both cohorts; Python validation overhead included in HOST; no HOST harness events; 63 observations/request; actual GPU proof pending review'
        timer=StepTimer(torch,args.cohort,synchronize=runner.synchronize)
        cpu_token=torch.empty(1,dtype=torch.int64,device='cpu',pin_memory=True);counter=0
        def make_req(prompt):
            nonlocal counter
            counter+=1
            sampling=SamplingParams(temperature=0,max_new_tokens=64,ignore_eos=True)
            req=Req(rid=str(counter),origin_input_text='',origin_input_ids=array('q',prompt),sampling_params=sampling)
            req.full_untruncated_fill_ids=req.origin_input_ids;req.fill_len=len(prompt);req.logprob_start_len=-1
            req.set_extend_input_len(req.fill_len-len(req.prefix_indices))
            return req
        for length in args.prompt_lengths:
            prompt=inputs[length]['prompt']
            for _ in range(2):request(runner,prompt,make_req,timer,torch,cpu_token,graph_probe)
            scenario={'prompt_length':length,'runs':[]};report['scenarios'].append(scenario)
            for repeat in range(3):
                run=request(runner,prompt,make_req,timer,torch,cpu_token,graph_probe);run['repeat_index']=repeat
                validate_run(run,prompt,inputs[length]['expected']);scenario['runs'].append(run)
            aggregate(scenario)
        if source_binding()!=binding:raise RuntimeError('Native source changed during measurement')
        report.update(status='measured',NUM_passed=all(r['NUM']['first10_exact'] for s in report['scenarios'] for r in s['runs']),
            all64_exact=all(r['NUM']['all64_exact'] for s in report['scenarios'] for r in s['runs']),
            native_graph_coverage=graph_probe.samples,identity_at_completion=verify_completion_identity(torch,identity),
            gpu_after=gpu_inventory(),acceptance='pending independent native runtime/stream/state/isolation/rotated-round review')
        write_new(args.output,report);return 0
    except BaseException as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc());write_new(args.output.with_suffix('.failed.json'),report);raise
    finally:
        if graph_probe:graph_probe.close()
        if runner:
            runner.synchronize();runner.clear()


if __name__=='__main__':raise SystemExit(main())

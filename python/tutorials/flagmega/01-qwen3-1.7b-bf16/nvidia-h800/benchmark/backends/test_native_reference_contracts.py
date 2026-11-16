"""CPU protocol regressions; fixtures are not CUDA objects or GPU identity proof."""
import ast
import enum
import importlib.abc
import os
from pathlib import Path
import sys
import types
from typing import NamedTuple, Optional
import unittest
from unittest.mock import patch


class NoGPUImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'vllm', 'sglang', 'pynvml', 'triton'}:
            raise AssertionError('CPU contract test attempted backend import: '+fullname)


sys.meta_path.insert(0, NoGPUImports())
import reference_rounds as shared
import vllm_native as vllm_adapter
import sglang_native as sg_adapter
import device_identity as identity_policy

def no_cuda_load(event, args):
    if event == 'ctypes.dlopen':raise AssertionError('CPU test attempted driver/NVML loading')
sys.addaudithook(no_cuda_load)

UUID = 'GPU-5b5e235a-1234-5678-9012-123456789abc'
PCI = '00000000:65:00.0'


def proof():
    return {'schema': 'flagmega.device-identity/v3', 'physical_gpu_uuid': UUID,
            'nvml_index': 3, 'pci_bus_id': PCI, 'selector_kind': 'nvml-index',
            'cuda_visible_device': '3', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}


class FakeCuda:
    def set_device(self, value): self.current = value
    def device_count(self): return 1
    def current_device(self): return self.current
    def get_device_properties(self, value): return types.SimpleNamespace(uuid=UUID)


class NativeRegression(unittest.TestCase):
    def test_numeric_native_selector_shared_binder(self):
        fake = types.SimpleNamespace(cuda=FakeCuda())
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='3', CUDA_DEVICE_ORDER='PCI_BUS_ID'):
            if hasattr(shared, 'validate_visibility'):
                identity = proof()
                identity['cuda'] = {'uuid': UUID, 'pci_bus_id': PCI, 'logical_device': 0,
                                    'visible_count': 1, 'compute_capability': [9, 0]}
                shared.bind_gpu(fake, UUID, identity=identity)
            else:
                # Execute the original production binder, not a copied predicate.
                shared.bind_gpu(fake, UUID)
        self.assertEqual(fake.cuda.current, 0)

    def test_actual_sglang_graph_field(self):
        native = types.SimpleNamespace(graph_runner=types.SimpleNamespace(graphs={1: object()},capture_bs=[1],
            can_run=lambda b:True,replay=lambda b:object()))
        if hasattr(sg_adapter, 'graph_readiness'):
            sg_adapter.graph_readiness(native, type(native.graph_runner))
        else:
            # Execute the exact original main readiness AST without loading a backend.
            tree = ast.parse(Path(sg_adapter.__file__).read_text())
            node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                        and isinstance(n.test, ast.Compare)
                        and any(isinstance(a, ast.Attribute) and a.attr == 'cuda_graph_runner'
                                for a in ast.walk(n.test)))
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                         sg_adapter.__file__, 'exec'), {'native': native})

    def test_vllm_actual_replay_gate_exists(self):
        self.assertTrue(hasattr(vllm_adapter, 'NativeGraphProbe'),
                        'Production adapter lacks selected-entry/actual replay gate')


def native_classes():
    """Execute unchanged native classes/call AST under explicit CPU protocol globals."""
    root=Path('/root/repos/vllm/vllm')
    env={'enum':enum,'NamedTuple':NamedTuple,'Optional':Optional,'VllmConfig':object,
         'logger':types.SimpleNamespace(warning_once=lambda _:None)}
    for name,path in [('CUDAGraphMode','config/compilation.py'),('BatchDescriptor','forward_context.py'),
                      ('CudagraphDispatcher','v1/cudagraph_dispatcher.py')]:
        node=next(n for n in ast.parse((root/path).read_text()).body if isinstance(n,ast.ClassDef) and n.name==name)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(root/path),'exec'),env)
    wrapper=next(n for n in ast.parse((root/'compilation/cuda_graph.py').read_text()).body
                 if isinstance(n,ast.ClassDef) and n.name=='CUDAGraphWrapper')
    call=next(n for n in wrapper.body if isinstance(n,ast.FunctionDef) and n.name=='__call__')
    exec(compile(ast.Module(body=[call],type_ignores=[]),str(root/'compilation/cuda_graph.py'),'exec'),env)
    return env


def graph_fixture():
    env=native_classes();M,B,D=(env[n] for n in ('CUDAGraphMode','BatchDescriptor','CudagraphDispatcher'))
    config=types.SimpleNamespace(cudagraph_mode=M.FULL,cudagraph_capture_sizes=[1],is_attention_compiled_piecewise=lambda:True)
    dispatcher=D(types.SimpleNamespace(compilation_config=config,scheduler_config=types.SimpleNamespace(max_num_seqs=1)))
    dispatcher.initialize_cudagraph_keys(M.FULL,1)
    class Graph:
        def __init__(self):self.hits=[]
        def replay(self,*args,**kwargs):self.hits.append((args,kwargs));return 'native replay result'
    graph=Graph();key=B(1,False);value=object()
    entry=types.SimpleNamespace(batch_descriptor=key,cudagraph=graph,output=value)
    class Wrapper:pass
    Wrapper.__call__=env['__call__']
    wrapper=Wrapper();wrapper.runtime_mode=M.FULL;wrapper.is_debugging_mode=False
    wrapper.concrete_cudagraph_entries={key:entry}
    wrapper.runnable=lambda *a,**k:(_ for _ in ()).throw(AssertionError('eager/capture path'))
    ctx=types.SimpleNamespace(batch_descriptor=key,cudagraph_runtime_mode=M.FULL)
    env['get_forward_context']=lambda:ctx
    runner=types.SimpleNamespace(model=wrapper,cudagraph_dispatcher=dispatcher,nncase_executor=None,
                                vllm_config=types.SimpleNamespace(nncase_config=None))
    return types.SimpleNamespace(M=M,B=B,runner=runner,dispatcher=dispatcher,wrapper=wrapper,
                                 graph=graph,entry=entry,value=value,context=ctx,Graph=Graph,Wrapper=Wrapper)


def probe(f):return vllm_adapter.NativeGraphProbe(f.runner,f.M.FULL,f.B,f.Wrapper,f.Graph,lambda:f.context)


class VllmGraphContract(unittest.TestCase):
    def test_real_native_dispatch_wrapper_replay_and_restoration(self):
        f=graph_fixture();original=f.dispatcher.dispatch.__func__;replay=f.graph.replay.__func__
        p=probe(f)
        try:
            for _ in range(63):
                p.begin();mode,key=f.dispatcher.dispatch(f.B(1,True));f.context.batch_descriptor=key
                self.assertEqual(mode,f.M.FULL);self.assertEqual(key,f.B(1,False))
                self.assertIs(f.wrapper(object()),f.value);p.finish()
            self.assertEqual(len(f.graph.hits),63);self.assertEqual(len(p.samples),63)
            self.assertNotIn('__call__',vars(f.wrapper))
        finally:p.close()
        self.assertIs(f.dispatcher.dispatch.__func__,original);self.assertIs(f.graph.replay.__func__,replay)
        self.assertNotIn('dispatch',vars(f.dispatcher));self.assertNotIn('replay',vars(f.graph))

    def test_input_selected_and_context_negative_controls(self):
        for case in ('nonuniform','size','NONE','PIECEWISE','invented_uniform','wrong_selected_size','wrong_context'):
            with self.subTest(case=case):
                f=graph_fixture()
                if case in ('NONE','PIECEWISE','invented_uniform','wrong_selected_size'):
                    original=f.dispatcher.dispatch
                    def dispatch(inp):
                        if case=='NONE':return f.M.NONE,None
                        if case=='PIECEWISE':return f.M.PIECEWISE,inp.non_uniform
                        return f.M.FULL,f.B(1,True) if case=='invented_uniform' else f.B(2,False)
                    f.dispatcher.dispatch=dispatch
                p=probe(f)
                try:
                    p.begin()
                    with self.assertRaises(RuntimeError):
                        inp=f.B(1,False) if case=='nonuniform' else f.B(2,True) if case=='size' else f.B(1,True)
                        f.dispatcher.dispatch(inp)
                        if case=='wrong_context':
                            f.context.batch_descriptor=f.B(1,True)
                            # Exercise the real instance replay observer's context
                            # gate without executing a fabricated capture path.
                            f.graph.replay()
                        else:f.wrapper()
                finally:p.close()

    def test_initial_readiness_negative_controls(self):
        for case in ('mode','keys','uninitialized','capture_size','entry_missing','uncaptured','entry_descriptor','wrapper_class','graph_class','nncase'):
            with self.subTest(case=case):
                f=graph_fixture()
                if case=='mode':f.dispatcher.cudagraph_mode=f.M.FULL_DECODE_ONLY
                elif case=='keys':f.dispatcher.cudagraph_keys[f.M.FULL].add(f.B(1,True))
                elif case=='uninitialized':f.dispatcher.keys_initialized=False
                elif case=='capture_size':f.dispatcher.compilation_config.cudagraph_capture_sizes=[1,2]
                elif case=='entry_missing':f.wrapper.concrete_cudagraph_entries.clear()
                elif case=='uncaptured':f.entry.cudagraph=None
                elif case=='entry_descriptor':f.entry.batch_descriptor=f.B(1,True)
                elif case=='wrapper_class':f.Wrapper=object
                elif case=='graph_class':f.Graph=object
                elif case=='nncase':f.runner.nncase_executor=object()
                with self.assertRaises(RuntimeError):probe(f)
                self.assertNotIn('dispatch',vars(f.dispatcher));self.assertNotIn('replay',vars(f.graph))

    def test_replay_count_identity_and_restore_on_error(self):
        for case in ('missing','duplicate','entry_replaced','graph_replaced','wrapper_replaced','replay_raises'):
            with self.subTest(case=case):
                f=graph_fixture()
                if case=='replay_raises':
                    f.graph.replay=lambda:(_ for _ in ()).throw(ValueError('native error'))
                original=vars(f.graph).get('replay');present='replay' in vars(f.graph);p=probe(f)
                try:
                    p.begin();f.dispatcher.dispatch(f.B(1,True))
                    with self.assertRaises((RuntimeError,ValueError)):
                        if case=='missing':p.finish()
                        else:
                            if case=='entry_replaced':f.wrapper.concrete_cudagraph_entries[f.B(1,False)]=object()
                            if case=='graph_replaced':f.entry.cudagraph=object()
                            if case=='wrapper_replaced':f.runner.model=object()
                            if case in ('entry_replaced','graph_replaced','wrapper_replaced'):p.finish()
                            else:f.wrapper()
                            if case=='duplicate':f.wrapper()
                finally:p.close()
                self.assertEqual('replay' in vars(f.graph),present)
                if present:self.assertIs(vars(f.graph)['replay'],original)

    def test_neighbor_native_full_decode_only_key_remains_uniform(self):
        env=native_classes();M,B,D=(env[n] for n in ('CUDAGraphMode','BatchDescriptor','CudagraphDispatcher'))
        config=types.SimpleNamespace(cudagraph_mode=M.FULL_DECODE_ONLY,cudagraph_capture_sizes=[1],is_attention_compiled_piecewise=lambda:True)
        d=D(types.SimpleNamespace(compilation_config=config,scheduler_config=types.SimpleNamespace(max_num_seqs=1)))
        d.initialize_cudagraph_keys(M.FULL_DECODE_ONLY,1)
        self.assertEqual(d.dispatch(B(1,True)),(M.FULL,B(1,True)))


class FakeSeq:
    def tolist(self):return [129]

class SGGraph:
    capture_bs=[1]
    def __init__(self):self.graphs={1:object()}
    def can_run(self,b):return True
    def replay(self,b):return object()

class SGOutput:
    def __init__(self,flag=True):self.can_run_graph=flag

class SGNative:
    def __init__(self):self.graph_runner=SGGraph();self.hits=[];self.output=SGOutput()
    def forward(self,*a,**k):self.hits.append((a,k));return self.output

def sg_batch():
    return types.SimpleNamespace(forward_mode=types.SimpleNamespace(is_decode=lambda:True),batch_size=1,
        input_ids=types.SimpleNamespace(numel=lambda:1),seq_lens_cpu=FakeSeq())


class SGLangGraphContract(unittest.TestCase):
    def test_original_output_arguments_and_restoration(self):
        n=SGNative();orig=n.forward.__func__;p=sg_adapter.NativeForwardProbe(n,SGGraph,SGOutput)
        try:
            for _ in range(63):
                b=sg_batch();p.begin();self.assertIs(n.forward(b,pp_proxy_tensors=None),n.output);p.finish()
                self.assertIs(n.hits[-1][0][0],b);self.assertEqual(n.hits[-1][1],{'pp_proxy_tensors':None})
            self.assertEqual(len(p.samples),63)
        finally:p.close()
        self.assertIs(n.forward.__func__,orig);self.assertNotIn('forward',vars(n))

    def test_readiness_rejections(self):
        for case in ('none','wrong_class','uncaptured','bs','missing_can_run'):
            with self.subTest(case=case):
                n=SGNative()
                if case=='none':n.graph_runner=None
                elif case=='wrong_class':n.graph_runner=object()
                elif case=='uncaptured':n.graph_runner.graphs={}
                elif case=='bs':n.graph_runner.capture_bs=[1,2]
                elif case=='missing_can_run':n.graph_runner.can_run=None
                with self.assertRaises(RuntimeError):sg_adapter.NativeForwardProbe(n,SGGraph,SGOutput)
                self.assertNotIn('forward',vars(n))
        with self.assertRaises(RuntimeError):sg_adapter.graph_readiness(SGNative(),SGGraph,SGGraph)

    def test_execution_rejections_restore_existing_instance_method(self):
        for case in ('flag_false','not_decode','size','tokens','multiple','missing','graph_changed','capture_changed','native_error'):
            with self.subTest(case=case):
                n=SGNative();original=n.forward
                if case=='native_error':original=lambda *a,**k:(_ for _ in ()).throw(ValueError('native error'))
                n.forward=original;p=sg_adapter.NativeForwardProbe(n,SGGraph,SGOutput);b=sg_batch()
                try:
                    p.begin()
                    if case=='flag_false':n.output.can_run_graph=False
                    if case=='not_decode':b.forward_mode.is_decode=lambda:False
                    if case=='size':b.batch_size=2
                    if case=='tokens':b.input_ids.numel=lambda:2
                    if case=='graph_changed':n.graph_runner=SGGraph()
                    if case=='capture_changed':n.graph_runner.graphs[1]=object()
                    with self.assertRaises((RuntimeError,ValueError)):
                        if case=='missing':p.finish()
                        else:
                            n.forward(b)
                            if case=='multiple':n.forward(b)
                finally:p.close()
                self.assertIs(vars(n)['forward'],original)

    def test_full63_request_consumes_original_own_sample_tensors(self):
        class Tensor:
            def __init__(self,value):self.value=value
            def item(self):return self.value
            def tolist(self):return [self.value]
            def numel(self):return 1
        n=SGNative();p=sg_adapter.NativeForwardProbe(n,SGGraph,SGOutput)
        class Bench:
            def clear(self):self.previous=None
            def extend(self,reqs):
                self.previous=Tensor(0)
                b=types.SimpleNamespace(seq_lens=Tensor(128),req_pool_indices=Tensor(1))
                return self.previous,None,b
            def decode(self,own,b):
                if own is not self.previous:raise AssertionError('Native own output tensor replaced/teacher-forced')
                b.seq_lens.value+=1
                f=sg_batch();f.input_ids=own;f.seq_lens_cpu=b.seq_lens;n.forward(f)
                self.previous=Tensor(own.value+1)
                return self.previous,None
            def cleanup(self,b):self.cleaned=True
        class CpuToken(Tensor):
            def copy_(self,value,non_blocking):
                if not non_blocking:raise AssertionError('Pinned delivery enqueue policy changed')
                self.value=value.value
        cuda=types.SimpleNamespace(synchronize=lambda:None,Event=lambda **k:(_ for _ in ()).throw(AssertionError('ordinary events')))
        torch=types.SimpleNamespace(cuda=cuda);timer=shared.StepTimer(torch,'ordinary_no_events');bench=Bench();prompt=list(range(128))
        try:run=sg_adapter.request(bench,prompt,lambda x:x,timer,torch,CpuToken(0),p)
        finally:p.close()
        self.assertEqual(run['token_ids'],list(range(64)))
        self.assertEqual(run['consumed_token_ids'],prompt+list(range(63)))
        self.assertEqual(len(run['decode_step_ms']),63);self.assertEqual(len(run['native_graph_coverage']),63)
        self.assertTrue(bench.cleaned)


class SelectorContract(unittest.TestCase):
    def test_numeric_and_uuid_neighbors(self):
        policy=proof();identity_policy.validate_visibility(policy,{'CUDA_VISIBLE_DEVICES':'3','CUDA_DEVICE_ORDER':'PCI_BUS_ID'})
        cuda={'uuid':UUID,'pci_bus_id':'0000:65:00.0','visible_count':1,'logical_device':0,'compute_capability':[9,0]}
        identity_policy.validate_cuda_identity(policy,cuda)
        other={**policy,'backend':'sglang','selector_kind':'uuid','cuda_visible_device':UUID}
        identity_policy.validate_visibility(other,{'CUDA_VISIBLE_DEVICES':UUID,'CUDA_DEVICE_ORDER':'PCI_BUS_ID'})

    def test_selector_proof_and_environment_negative_controls(self):
        for case in ('bad_uuid','MIG','negative','multiple','empty','index','PCI','uuid','order','override','count','sm'):
            with self.subTest(case=case):
                p=proof();env={'CUDA_VISIBLE_DEVICES':'3','CUDA_DEVICE_ORDER':'PCI_BUS_ID'}
                cuda={'uuid':UUID,'pci_bus_id':PCI,'visible_count':1,'logical_device':0,'compute_capability':[9,0]}
                if case=='bad_uuid':p['physical_gpu_uuid']='GPU-alias'
                if case=='MIG':p['physical_gpu_uuid']='MIG-'+UUID
                if case=='negative':p['cuda_visible_device']='-1'
                if case=='multiple':p['cuda_visible_device']='3,2'
                if case=='empty':p['cuda_visible_device']=''
                if case=='index':p['nvml_index']=2
                if case=='PCI':cuda['pci_bus_id']='0000:64:00.0'
                if case=='uuid':cuda['uuid']='GPU-00000000-0000-0000-0000-000000000000'
                if case=='order':env['CUDA_DEVICE_ORDER']='FASTEST_FIRST'
                if case=='override':env['CUDA_VISIBLE_DEVICES']=UUID
                if case=='count':cuda['visible_count']=2
                if case=='sm':cuda['compute_capability']=[8,0]
                with self.assertRaises(ValueError):
                    identity_policy.validate_visibility(p,env);identity_policy.validate_cuda_identity(p,cuda)

    def test_missing_actual_runtime_proof_before_driver_loading(self):
        a=types.SimpleNamespace(gpu_selector_kind='nvml-index',cuda_visible_device='3',physical_gpu_uuid=UUID,device_identity_proof=None)
        with patch.dict(os.environ,{},clear=True),self.assertRaises(ValueError):identity_policy.prepare_identity(a,'vllm')

    def test_backend_policy_no_numeric_fallback(self):
        for backend in ('sglang','tilefoundry','flagmega','llama.cpp'):
            with self.subTest(backend=backend),self.assertRaises(ValueError):identity_policy.selector_policy(backend,UUID,'nvml-index','3')


class SourceAndTimingContract(unittest.TestCase):
    def test_fresh_identity_import_performs_no_driver_loading(self):
        import subprocess
        source=Path(identity_policy.__file__).resolve()
        code="import sys,runpy\ndef audit(event,args):\n if event=='ctypes.dlopen':raise RuntimeError('driver loading during CPU import')\nsys.addaudithook(audit)\nrunpy.run_path(sys.argv[1])\n"
        result=subprocess.run([sys.executable,'-B','-c',code,str(source)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_source_bindings_and_native_loaded_origin(self):
        # Metadata/source parsing only; imports remain guarded.
        binding=vllm_adapter.source_binding()
        self.assertEqual(binding['commit'],vllm_adapter.PINNED_COMMIT)
        self.assertGreaterEqual(len(binding['AST_contracts']),5)
        import importlib.metadata
        site='/root/miniconda3/envs/sglang_bench/lib/python3.10/site-packages'
        distribution=next(d for d in importlib.metadata.Distribution.discover(path=[site])
                          if d.metadata['Name']=='sglang')
        # Read actual other-env distribution metadata; no installed backend import.
        distributions={d.metadata['Name']:d for d in importlib.metadata.Distribution.discover(path=[site])}
        with patch.object(sg_adapter.importlib.metadata,'distribution',side_effect=lambda name:distributions[name]),\
             patch.object(sg_adapter.importlib.metadata,'version',side_effect=lambda name:distributions[name].version):
            sg=sg_adapter.source_binding()
        self.assertGreaterEqual(len(sg['AST_contracts']),6)
        with self.assertRaises(ValueError):identity_policy.loaded_origin(SGNative,sg)

    def test_ordinary_and_device_delivery_order(self):
        for cohort in shared.COHORTS:
            log=[]
            class Event:
                def record(self,stream):log.append('event')
                def elapsed_time(self,other):log.append('query');return 1.0
            class Cuda:
                def Event(self,**kwargs):
                    if cohort=='ordinary_no_events':raise AssertionError('ordinary CUDA event allocation')
                    return Event()
                def current_stream(self):return object()
                def synchronize(self):log.append('sync')
            timer=shared.StepTimer(types.SimpleNamespace(cuda=Cuda()),cohort);log.clear()
            def operation():log.extend(['fullmodel+sampling','pinned_D2H']);return 'own token'
            def delivery(v):log.append('delivery');return v
            value,span=timer.measure(operation,delivery)
            self.assertEqual(value,'own token');self.assertLess(log.index('pinned_D2H'),log.index('sync'))
            self.assertLess(log.index('sync'),log.index('delivery'))
            if cohort=='event_instrumented':
                self.assertEqual(log,['event','fullmodel+sampling','pinned_D2H','event','sync','delivery','query'])
                self.assertIn('device_ms',span)
            else:self.assertNotIn('device_ms',span)

    def test_vllm_device_native_D2H_stream_boundary_and_restoration(self):
        for different_stream in (False,True):
            with self.subTest(different_stream=different_stream):
                log=[]
                class Event:
                    def __init__(self,name):self.name=name
                    def record(self,*a,**k):log.append(self.name)
                    def elapsed_time(self,other):return 1.0
                class Cuda:
                    def __init__(self):self.events=0;self.calls=0
                    def Event(self,**kwargs):self.events+=1;return Event('start' if self.events==1 else 'end')
                    def current_stream(self):
                        self.calls+=1
                        return types.SimpleNamespace(cuda_stream=8 if different_stream and self.calls%2==0 else 7)
                    def synchronize(self):log.append('sync')
                cuda=Cuda();timer=shared.StepTimer(types.SimpleNamespace(cuda=cuda),'event_instrumented');log.clear();cuda.calls=0
                class Runner:
                    use_async_scheduling=False
                    def __init__(self):self.transfer_event=Event('native_transfer')
                    def execute_model(self,scheduler):log.append('pinned_D2H');self.transfer_event.record();return 'native own output'
                runner=Runner();original_event=runner.transfer_event;original=runner.execute_model.__func__
                worker=vllm_adapter.WorkerProbe(runner,timer)
                try:
                    call=lambda:runner.execute_model(types.SimpleNamespace(total_num_scheduled_tokens=1))
                    if different_stream:
                        with self.assertRaises(RuntimeError):timer.measure(call,probe=worker)
                    else:
                        value,span=timer.measure(call,lambda v:(log.append('delivery'),v)[1],probe=worker)
                        self.assertEqual(value,'native own output')
                        self.assertEqual(log,['start','pinned_D2H','native_transfer','end','sync','delivery'])
                        self.assertEqual(len(worker.samples),1);self.assertIn('device_ms',span)
                finally:worker.close()
                self.assertIs(runner.transfer_event,original_event);self.assertIs(runner.execute_model.__func__,original)
                self.assertNotIn('execute_model',vars(runner))

    def test_monitor_preserves_numeric_environment(self):
        import importlib.util
        path=Path(shared.__file__).parents[1]/'run_monitored_backend_v3.py'
        spec=importlib.util.spec_from_file_location('monitor_v3',path);monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
        config={'backend':'vllm','GPU_UUID':UUID,'gpu_selector_kind':'nvml-index','cuda_visible_device':'3',
            'environment':{'CUDA_VISIBLE_DEVICES':'3','CUDA_DEVICE_ORDER':'PCI_BUS_ID'}}
        actual={'uuid':UUID,'nvml_index':3,'pci_bus_id':PCI}
        p,env=monitor.apply_selector(config,actual,{'CUDA_VISIBLE_DEVICES':UUID})
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'3');self.assertEqual(p['physical_gpu_uuid'],UUID)
        with self.assertRaises(ValueError):monitor.apply_selector(config,{**actual,'nvml_index':2},{})

    def test_planner_rotation_and_pending_identity(self):
        result=shared.plan(shared.WORK,shared.WORK,UUID,3)
        self.assertEqual(result['rounds'],3)
        for job in result['jobs']:
            self.assertFalse(job['GPU_access']);self.assertFalse(job['runnable_after_UUID_binding'])
            self.assertEqual(job['environment']['CUDA_VISIBLE_DEVICES'],'3' if job['backend']=='vllm' else UUID)
            self.assertEqual(set(map(int,job['argv'][job['argv'].index('--prompt-lengths')+1:job['argv'].index('--round-index')])),set(shared.LENGTHS))
        self.assertNotEqual([j['backend'] for j in result['jobs'] if j['round']==0],
                            [j['backend'] for j in result['jobs'] if j['round']==1])


if __name__ == '__main__': unittest.main()

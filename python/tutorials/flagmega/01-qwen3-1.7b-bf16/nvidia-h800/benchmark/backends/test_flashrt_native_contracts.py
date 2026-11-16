"""CPU protocol fixtures; no native load, CUDA tensors, NUM or performance proof."""
import ast
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
import flashrt_native as adapter

ROOT = adapter.WORK / '.local/cache/reference-candidates/FlashRT-a007ef0-h800-qwen3-1_7b-v1'


def source_fixture():
    runtime_path = ROOT / 'flash_rt/frontends/torch/_qwen3_dense_bf16_runtime.py'
    tree = ast.parse(runtime_path.read_text())
    selected = [n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and
                n.name in ('generation_extent','DecodeState','validate_input_ids')]
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==adapter.RUNTIME)
    methods = [n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in
               ('prepare_generation','decode_step','graph_key','output_ids','logits')]
    for method in methods:
        if method.name not in ('output_ids','logits'):method.decorator_list=[]
    cache_tree = ast.parse((ROOT/'flash_rt/models/qwen3/cache_bf16.py').read_text())
    selected.append(next(n for n in cache_tree.body if isinstance(n,ast.ClassDef) and n.name=='CacheLayout'))
    environment={'dataclass':dataclass,'torch':SimpleNamespace(Tensor=type('NeverTorchTensor',(),{})),
                 'operator':__import__('operator'),'__name__':__name__}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(runtime_path),'exec'),environment)
    namespace={}
    for method in methods:
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(runtime_path),'exec'),environment,namespace)
    return environment,namespace


class CPUBuffer:
    def __init__(self, values, log, indices=None, pinned=False):
        self.values,self.log,self.indices,self.pinned=values,log,list(range(len(values)))if indices is None else indices,pinned
    def __getitem__(self,key):
        indices=self.indices[key];return CPUBuffer(self.values,self.log,[indices]if isinstance(indices,int)else indices)
    def item(self):
        self.log.append('delivery');return self.values[self.indices[0]]
    def tolist(self):return [self.values[i]for i in self.indices]
    def detach(self):return self
    def cpu(self):return self
    def copy_(self,other,non_blocking=False):
        if self.pinned:self.log.append('D2H')
        for i,v in zip(self.indices,other.tolist()):self.values[i]=v
        return self


class CPUStream:
    def __init__(self,log):self.log,self.cuda_stream=log,12345
    def synchronize(self):self.log.append('sync')


class CPUEvent:
    def __init__(self,log,name):self.log,self.name=log,name
    def record(self,stream):self.log.append(self.name)
    def elapsed_time(self,other):return 777.0  # explicit CPU sentinel, never a GPU measurement


class CPUCuda:
    def __init__(self,log):self.log=log;self.events=[];self.active=CPUStream(log);self.CUDAGraph=CPUGraph
    def current_stream(self,*args):return self.active
    @contextmanager
    def stream(self,stream):
        previous=self.active;self.active=stream
        try:yield
        finally:self.active=previous
    def Event(self,**kwargs):
        event=CPUEvent(self.log,'start'if not self.events else'end');self.events.append(event);return event


class CPUGraph:
    def __init__(self,runtime,position,slot):self.runtime,self.position,self.slot=runtime,position,slot
    def replay(self):self.runtime.run_native_fixture(self.position,self.slot)


class CPURuntime:
    def __init__(self,prompt_length=128,torch=None):
        env,methods=source_fixture()
        for name,method in methods.items():setattr(type(self),name,method)
        self.state=env['DecodeState']();self.device='CPU_protocol_fixture_only';self.cuda_graphs=True
        self._lock=__import__('threading').RLock();self._stream=torch.cuda.active
        self.log=torch.cuda.log;self._graphs={};self._pointer_identity=(101,202,303)
        self._out_ids=CPUBuffer([-1]*64,self.log);self._token=CPUBuffer([0],self.log)
        self._logits=object();self.prefills=0;self.captures=0;self.consumed=[];self.KV=[[0]*2112 for _ in range(28)]
        self.output_override=None
    @contextmanager
    def _execution(self):yield
    def _restore_prompt(self,values):
        for layer in self.KV:layer[:]=[0]*2112;layer[:len(values)]=[1]*len(values)
        self._out_ids.values[:]=[-1]*64;self._out_ids.values[0]=7;self._token.values[0]=7
        self.state.prefilled(len(values));self._prompt=tuple(values);self.prefills+=1
    def _ensure_graph(self,position,slot):
        key=self.graph_key(position,slot)
        if key in self._graphs:return self._graphs[key],False
        self.run_native_fixture(position,slot)  # capture/warm mutations must be restored
        self._graphs[key]=CPUGraph(self,position,slot);self.captures+=1
        return self._graphs[key],True
    def run_native_fixture(self,position,slot):
        self.log.append('native');old=self._token.values[0];self.consumed.append(old)
        new=(old*3+slot)%151936
        if slot==11:new=151645  # EOS must not terminate the protocol
        self._token.values[0]=new;self._out_ids.values[slot]=new
        for layer in self.KV:layer[position]=2
    def reset(self):
        self.log.append('reset');self.state.reset()
    def clear_graphs(self):self._graphs.clear()


def frontend_fixture(torch):
    runtime=CPURuntime(torch=torch)
    tree=ast.parse((ROOT/'flash_rt/frontends/torch/qwen3_1_7b_h800.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name==adapter.FRONTEND)
    env={adapter.RUNTIME:lambda *a,**k:runtime,'QWEN3_1_7B_H800':'CPU_source_fixture_only'}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'actual_public_frontend_fixture','exec'),env)
    return env[adapter.FRONTEND](),runtime


class FlashRTContracts(unittest.TestCase):
    def run_request(self,cohort,length=128):
        log=[];torch=SimpleNamespace(cuda=CPUCuda(log));frontend,runtime=frontend_fixture(torch)
        timer=adapter.StepTimer(torch,cohort,synchronize=runtime._stream.synchronize)
        cpu_token=CPUBuffer([0],log,pinned=True);log.clear()
        result=adapter.timed_request(frontend,[5]*length,timer,torch,cpu_token)
        return result,runtime,torch,log

    def test_module_import_is_strict_CPU_only(self):
        script="import sys;sys.path.insert(0,sys.argv[1]);import flashrt_native;assert not any(x in sys.modules for x in ['torch','triton','flash_rt','pynvml']);print('CPU-only import')"
        result=subprocess.run([sys.executable,'-B','-S','-c',script,str(Path(adapter.__file__).parent)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_public_source_owns64_feedback_and_full_five_growth(self):
        for length in (32,128,250,1024,2048):
            result,runtime,torch,log=self.run_request('ordinary_no_events',length)
            self.assertEqual(len(result['token_ids']),64);self.assertEqual(len(result['decode_step_ms']),63)
            self.assertEqual(result['consumed_token_ids'],[5]*length+result['token_ids'][:-1])
            self.assertEqual(runtime.consumed[-63:],result['token_ids'][:-1])
            self.assertEqual(result['native_state'],{'cache_length':length+63,'output_count':64,'last_consumed_position':length+62})
            self.assertIn(151645,result['token_ids']);self.assertFalse(torch.cuda.events)
            self.assertTrue(all('replay'not in vars(graph)for graph in runtime._graphs.values()))

    def test_actual_prepare_source_restores_all28_after_capture(self):
        log=[];torch=SimpleNamespace(cuda=CPUCuda(log));frontend,runtime=frontend_fixture(torch)
        frontend.prepare_generation([3]*128,max_new_tokens=64)
        self.assertEqual(runtime.captures,63);self.assertEqual(runtime.prefills,2)
        self.assertEqual(runtime.state.output_count,1);self.assertEqual(runtime._token.item(),7)
        self.assertTrue(all(layer[128:]==[0]*(2112-128)for layer in runtime.KV))

    def test_DEVICE_pair_reused_and_end_after_D2H_before_sync_delivery(self):
        result,runtime,torch,log=self.run_request('event_instrumented')
        self.assertEqual(len(torch.cuda.events),2);self.assertEqual(len(result['device_decode_step_ms']),63)
        position=log.index('start');timed=log[position:]
        for step in range(63):self.assertEqual(timed[step*6:step*6+6],['start','native','D2H','end','sync','delivery'])
        self.assertEqual(len(result['native_graph_coverage']),63)

    def test_missing_prepared_graph_has_no_eager_fallback(self):
        log=[];torch=SimpleNamespace(cuda=CPUCuda(log));frontend,runtime=frontend_fixture(torch)
        frontend.prepare_generation([3]*128,max_new_tokens=64);runtime._graphs.pop(runtime.graph_key(128,1))
        timer=adapter.StepTimer(torch,'ordinary_no_events',runtime._stream.synchronize)
        with self.assertRaisesRegex(RuntimeError,'observer ABI unavailable'):adapter.NativeReplayProbe(runtime,torch,timer,128)
        self.assertTrue(all('replay'not in vars(graph)for graph in runtime._graphs.values()))

    def test_wrong_actual_replay_stream_fails_and_restores_instances(self):
        log=[];torch=SimpleNamespace(cuda=CPUCuda(log));frontend,runtime=frontend_fixture(torch)
        frontend.prepare_generation([3]*128,max_new_tokens=64);timer=adapter.StepTimer(torch,'ordinary_no_events',runtime._stream.synchronize)
        probe=adapter.NativeReplayProbe(runtime,torch,timer,128);probe.begin();torch.cuda.active.cuda_stream=54321
        probe.stream=SimpleNamespace(cuda_stream=12345)
        with self.assertRaisesRegex(RuntimeError,'stream differs'):frontend.decode_step()
        probe.close();self.assertTrue(all('replay'not in vars(graph)for graph in runtime._graphs.values()))

    def test_pointer_change_fails_without_accepting_timed_capture(self):
        log=[];torch=SimpleNamespace(cuda=CPUCuda(log));frontend,runtime=frontend_fixture(torch)
        frontend.prepare_generation([3]*128,max_new_tokens=64);timer=adapter.StepTimer(torch,'ordinary_no_events',runtime._stream.synchronize)
        probe=adapter.NativeReplayProbe(runtime,torch,timer,128);runtime._pointer_identity=(404,505,606)
        with self.assertRaisesRegex(RuntimeError,'pointers changed'):probe.begin()
        probe.close()

    def test_native_decode_exception_still_restores_replay_attributes(self):
        log=[];torch=SimpleNamespace(cuda=CPUCuda(log));frontend,runtime=frontend_fixture(torch)
        timer=adapter.StepTimer(torch,'ordinary_no_events',runtime._stream.synchronize)
        original=runtime.run_native_fixture
        def fail_after_warm(position,slot):
            if runtime.state.output_count>1:raise RuntimeError('actual CPU fixture payload error')
            return original(position,slot)
        runtime.run_native_fixture=fail_after_warm
        with self.assertRaisesRegex(RuntimeError,'payload error'):
            adapter.timed_request(frontend,[5]*128,timer,torch,CPUBuffer([0],log,pinned=True))
        self.assertTrue(all('replay'not in vars(graph)for graph in runtime._graphs.values()))

    def test_NUM_is_public_finite_generation_and_distinct_from_timing(self):
        tree=ast.parse(Path(adapter.__file__).read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='NUM_request')
        calls=[n for n in ast.walk(function)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='generate_ids']
        self.assertEqual(len(calls),2)
        self.assertTrue(all(any(k.arg=='check_logits'and isinstance(k.value,ast.Constant)and k.value.value is True for k in n.keywords)for n in calls))
        self.assertFalse(any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='measure'for n in ast.walk(function)))


if __name__=='__main__':unittest.main()

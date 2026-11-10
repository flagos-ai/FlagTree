import ast
import ctypes
import gc
import sys
import threading
import types
import weakref
from contextlib import contextmanager
from pathlib import Path

import pytest
from triton.flagmega.codegen.cpp.program_abi import PlanResult, PlanStage, ProgramRow
from triton.flagmega.errors import RuntimeContractError
from triton.flagmega.runtime.cpp_programs import CppCounterEpochFrame, CounterEpochLaunchError, CppProgramFrame


class Tensor:
    def __init__(self, address=4096, size=128): self.address, self.size = address, size
    def data_ptr(self): return self.address
    def numel(self): return self.size
    def element_size(self): return 1


class Stream:
    def __init__(self, handle=17): self.cuda_stream, self.waits, self.fail = handle, 0, False
    def synchronize(self):
        self.waits += 1
        if self.fail: raise RuntimeError('wait failed')


class Library:
    def __init__(self):
        self.counter_epoch_plan = {'frame_nbytes':100, 'rows':[{'entry_id':i} for i in range(3)]}
        self.manifest = {'entries':[{'entry_id':i,'scratch_nbytes':100,'signature':[{'tag':1,'min_nbytes':64,'alignment':4}]} for i in range(3)],
                         'invocations':[{'entry_id':i,'ordinal':i,'call_path':[], 'argument_bindings':[{'kind':'argument','source':0}]} for i in range(3)]}
        self.calls=[]; self.callback=None; self.failure=None
    def validate_counter_epoch_plan(self, cells, rows, **kw):
        if len(rows)!=3 or any(row.reserved for row in rows):
            raise RuntimeContractError('mock preflight rejected rows')
    def launch_counter_epoch_plan(self, cells, rows, result, stream, **kw):
        if self.callback: self.callback()
        self.calls.append(([c[0].payload for c in cells], stream))
        if self.failure: raise self.failure
    def launch(self, entry, cells, stream, **kw):
        self.calls.append((entry['entry_id'], cells[0].payload))
        if self.callback and entry['entry_id']==0:self.callback()


def frame(lib=None, roots=None, **kw):
    return CppCounterEpochFrame(lib or Library(), roots or [Tensor()], device=0, scratch=8192, scratch_bytes=100, keepalive=kw.get('keepalive',()))


class Graph:
    def __init__(self, cuda):self.cuda=cuda;self.replays=[];self.resets=0;self.fail_reset=False;self.fail_replay=False
    def replay(self):
        self.replays.append(self.cuda.active)
        if self.fail_replay:raise RuntimeError('replay failed')
    def reset(self):
        self.resets+=1
        if self.fail_reset:raise RuntimeError('reset failed')


class CUDA:
    def __init__(self):self.active=99;self.last_graph=None;self.fail_capture=False;self.capturing=False
    def CUDAGraph(self):self.last_graph=Graph(self);return self.last_graph
    @contextmanager
    def stream(self, stream):
        old=self.active;self.active=stream.cuda_stream
        try:yield
        finally:self.active=old
    @contextmanager
    def graph(self, graph, stream):
        with self.stream(stream):
            yield
            if self.fail_capture:raise RuntimeError('capture failed')
    def is_current_stream_capturing(self):return self.capturing


def torch_mock(monkeypatch):
    torch=types.SimpleNamespace(cuda=CUDA());monkeypatch.setitem(sys.modules,'torch',torch);return torch


def test_actual_legacy_rebinding_counterexample_and_new_reentrant_guard():
    lib=Library();old=CppProgramFrame(lib,[Tensor()],device=0,scratch=8192,scratch_bytes=100)
    lib.callback=lambda:old.bind([Tensor(16384)])
    old.enqueue(17)
    assert lib.calls==[(0,4096),(1,16384),(2,16384)]
    lib=Library();new=frame(lib);roots=new.roots;stream=Stream();caught=[]
    def callback():
        with pytest.raises(RuntimeContractError,match='busy or reentered'):
            new.submit([Tensor(16384)],stream)
        caught.append(True)
    lib.callback=callback
    new.submit(roots,stream)
    assert caught and lib.calls==[([4096,4096,4096],17)]
    new.synchronize();new.release()


def test_deterministic_two_host_threads_cannot_interleave_bind_enqueue():
    lib=Library();f=frame(lib);stream=Stream();entered=threading.Event();finish=threading.Event();errors=[]
    def callback():entered.set();assert finish.wait(5)
    lib.callback=callback
    def submit():
        try:f.submit(f.roots,stream)
        except BaseException as error:errors.append(error)
    thread=threading.Thread(target=submit);thread.start();assert entered.wait(5)
    try:
        with pytest.raises(RuntimeContractError,match='busy'):
            f.submit([Tensor(16384)],stream)
    finally:finish.set();thread.join(5)
    assert not thread.is_alive() and not errors and lib.calls==[([4096]*3,17)]
    f.synchronize();f.release()


def test_scratch_alias_and_root_overlap_are_excluded():
    f=frame()
    with pytest.raises(RuntimeContractError,match='aliases'):
        CppCounterEpochFrame(Library(),[Tensor()],device=0,scratch=8240,scratch_bytes=100,keepalive=())
    f.release()
    with pytest.raises(RuntimeContractError,match='overlaps'):
        frame(roots=[Tensor(8180,128)])


def test_owned_stream_no_unmanaged_capture_and_late_binding_errors():
    f=frame();s=Stream()
    with pytest.raises(RuntimeContractError,match='unmanaged'):
        f.submit(f.roots,s,capturing=True)
    assert not f.pending and not f.library.calls
    with pytest.raises(RuntimeContractError,match='span/alignment'):
        f.submit([Tensor(4097)],s)
    assert not f.pending and not f.library.calls
    f.submit(f.roots,s)
    with pytest.raises(RuntimeContractError,match='one owned stream'):
        f.submit(f.roots,Stream(18))
    f.synchronize();f.release()


def test_graph_owns_roots_and_uses_owned_stream_with_actual_retirement(monkeypatch):
    torch=torch_mock(monkeypatch);f=frame();s=Stream();g=f.capture(f.roots,s,torch);raw=torch.cuda.last_graph
    assert f.captured and f.pending and len(f.library.calls)==1
    with pytest.raises(RuntimeContractError,match='pinned'):
        f.submit([Tensor(16384)],s)
    with pytest.raises(RuntimeContractError,match='switch streams'):g.replay(stream=Stream(18))
    g.replay();g.replay();assert raw.replays==[17,17] and torch.cuda.active==99
    f.synchronize();assert not f.pending and f.captured
    with pytest.raises(RuntimeContractError,match='retained'):f.require_releasable()
    g.close();assert raw.resets==1 and s.waits==2 and not f.captured
    with pytest.raises(RuntimeContractError,match='retired'):g.replay()
    f.require_releasable();f.release()


def test_failed_completion_and_graph_reset_keep_quarantine(monkeypatch):
    torch=torch_mock(monkeypatch);f=frame();s=Stream();g=f.capture(f.roots,s,torch);raw=torch.cuda.last_graph
    s.fail=True
    with pytest.raises(RuntimeError,match='wait failed'):g.close()
    assert f.pending and f.captured and raw.resets==0
    s.fail=False;raw.fail_reset=True
    with pytest.raises(RuntimeError,match='reset failed'):g.close()
    assert f.pending and f.captured
    raw.fail_reset=False;g.close();f.release()


@pytest.mark.parametrize('stage',[PlanStage.RESET,PlanStage.LAUNCH])
def test_partial_failure_poison_survives_completion_and_requires_fresh_frame(stage):
    lib=Library();f=frame(lib);s=Stream();lib.failure=CounterEpochLaunchError(71,PlanResult(1,32,8,stage,71,8,0,0))
    with pytest.raises(CounterEpochLaunchError) as caught:f.submit(f.roots,s)
    assert caught.value.ordinal==8 and caught.value.stage==stage and f.pending
    s.fail=True
    with pytest.raises(RuntimeError):f.synchronize()
    assert f.pending
    s.fail=False;f.synchronize();assert not f.pending and f.poisoned is caught.value
    with pytest.raises(RuntimeContractError,match='fresh frame'):f.submit(f.roots,s)
    f.release();new=frame();new.submit(new.roots,s);new.synchronize();new.release()


def test_failed_capture_retires_actual_graph_before_release(monkeypatch):
    torch=torch_mock(monkeypatch);torch.cuda.fail_capture=True;f=frame();s=Stream()
    with pytest.raises(RuntimeError,match='capture failed'):f.capture(f.roots,s,torch)
    assert f.pending and f.captured and f.poisoned
    with pytest.raises(RuntimeContractError,match='retained'):f.require_releasable()
    f.retire_graphs();assert torch.cuda.last_graph.resets==1
    assert f.poisoned and not f.pending and not f.captured
    f.release()


def test_pending_frame_and_graph_strongly_keep_every_resource(monkeypatch):
    torch=torch_mock(monkeypatch);root=Tensor();module=Tensor(12000);scratch=Tensor(8192,100)
    f=frame(roots=[root],keepalive=(module,scratch));s=Stream();g=f.capture(f.roots,s,torch)
    refs=[weakref.ref(x) for x in (f,root,module,scratch)];del root,module,scratch,f;gc.collect()
    assert all(x() is not None for x in refs)
    g.close();refs[0]().release();del g;gc.collect()
    assert all(x() is None for x in refs)


def test_pending_eager_frame_survives_owner_drop_until_real_sync():
    root=Tensor();f=frame(roots=[root],keepalive=(root,));s=Stream();f.submit(f.roots,s)
    ref=weakref.ref(f);rootref=weakref.ref(root);del root,f;gc.collect()
    assert ref() is not None and rootref() is not None
    survivor=ref();survivor.synchronize();survivor.release();del survivor;gc.collect()
    assert ref() is None and rootref() is None


def test_module_completion_rejects_boolean_graph_retirement(monkeypatch):
    # Execute the real method AST without importing unrelated model backends.
    from triton.flagmega.runtime import cpp_programs
    source=Path(cpp_programs.__file__).with_name('cpp_module.py')
    tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='_CppLaunchMixin')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='complete_cpp_programs')
    ns={'RuntimeContractError':RuntimeContractError};exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),ns)
    torch_mock(monkeypatch);f=frame();s=Stream();f.submit(f.roots,s);owner=types.SimpleNamespace(_cpp_program_frame=f)
    with pytest.raises(RuntimeContractError,match='boolean'):
        ns['complete_cpp_programs'](owner,s,graph_released=True)
    assert f.pending and s.waits==0
    ns['complete_cpp_programs'](owner,s);assert not f.pending and s.waits==1
    f.release()


def test_owned_replay_cannot_escape_via_unmanaged_outer_capture(monkeypatch):
    torch=torch_mock(monkeypatch);f=frame();g=f.capture(f.roots,Stream(),torch);raw=torch.cuda.last_graph
    torch.cuda.capturing=True
    with pytest.raises(RuntimeContractError,match='unmanaged'):g.replay()
    with pytest.raises(RuntimeContractError,match='outside capture'):g.close()
    assert not raw.replays and raw.resets==0
    torch.cuda.capturing=False;g.close();f.release()


def test_same_capture_prefix_plan_feedback_order_and_extra_root_lifetime(monkeypatch):
    torch=torch_mock(monkeypatch);lib=Library();f=frame(lib);s=Stream();events=[]
    lib.callback=lambda:events.append(('plan',torch.cuda.active))
    extra=Tensor(24000);ref=weakref.ref(extra)
    before=lambda:events.append(('slot_mapping_copy',torch.cuda.active))
    after=lambda:events.append(('input_ids_feedback',torch.cuda.active))
    before_ref=weakref.ref(before);after_ref=weakref.ref(after)
    g=f.capture(f.roots,s,torch,before=before,after=after,keepalive=(extra,))
    del extra,before,after;gc.collect()
    assert events==[('slot_mapping_copy',17),('plan',17),('input_ids_feedback',17)]
    assert ref() is not None and before_ref() is not None and after_ref() is not None
    g.close();gc.collect();assert ref() is None and before_ref() is None and after_ref() is None
    f.release()


def test_entire_preflight_precedes_capture_and_prefix(monkeypatch):
    torch=torch_mock(monkeypatch);f=frame();events=[];f.rows[-1].reserved=1
    with pytest.raises(RuntimeContractError,match='preflight'):
        f.capture(f.roots,Stream(),torch,before=lambda:events.append('prefix'))
    assert not events and torch.cuda.last_graph is None and not f.pending
    f.rows[-1].reserved=0;f.release()


@pytest.mark.parametrize('position',['before','after'])
def test_feedback_failure_quarantines_actual_capture_and_extra_roots(monkeypatch,position):
    torch=torch_mock(monkeypatch);f=frame();extra=Tensor(24000);ref=weakref.ref(extra)
    def failure():raise RuntimeError('feedback failed')
    with pytest.raises(RuntimeError,match='feedback failed'):
        f.capture(f.roots,Stream(),torch,keepalive=(extra,),**{position:failure})
    del extra;gc.collect();assert ref() is not None and f.pending and f.poisoned and f.captured
    f.retire_graphs();gc.collect();assert ref() is None
    assert f.poisoned is not None;f.release()


@pytest.mark.parametrize('scratch',[0,True,2**64+8192,2**64-4])
def test_scratch_rejects_noncanonical_or_wrapping_native_pointer(scratch):
    with pytest.raises(RuntimeContractError,match='aligned capacity'):
        CppCounterEpochFrame(Library(),[Tensor()],device=0,scratch=scratch,scratch_bytes=100,keepalive=())

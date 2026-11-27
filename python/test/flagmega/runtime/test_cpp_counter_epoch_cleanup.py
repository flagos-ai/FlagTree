"""Cleanup failures keep a frame poisoned while allowing actual cleanup retry."""
import ast
import gc
import types
import weakref
from pathlib import Path

import pytest
if __package__:
    from .test_cpp_counter_epoch import Tensor, Stream, Library, frame, torch_mock
else:
    from test_cpp_counter_epoch import Tensor, Stream, Library, frame, torch_mock
from triton.flagmega.codegen.cpp.program_abi import PlanResult, PlanStage
from triton.flagmega.errors import RuntimeContractError
from triton.flagmega.runtime import cpp_programs


def module_method(name):
    # Exercise the real public integration method without unrelated backends.
    source = Path(cpp_programs.__file__).with_name('cpp_module.py')
    if not source.is_file():
        # A private delta checkout intentionally contains no unchanged module.
        import os
        source = Path(os.environ['FLAGMEGA_TEST_CPP_MODULE_SOURCE'])
    cls = next(node for node in ast.parse(source.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == '_CppLaunchMixin')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {'RuntimeContractError': RuntimeContractError}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[name]


@pytest.mark.parametrize('path,fault', [
    ('frame_completion', 'sync'),
    ('module_completion', 'sync'),
    ('frame_graph_completion', 'sync'),
    ('frame_retirement', 'sync'),
    ('wrapper_close', 'sync'),
    ('frame_retirement', 'reset'),
    ('wrapper_close', 'reset'),
    ('module_retirement', 'reset'),
])
def test_cleanup_failure_blocks_all_new_work_but_allows_actual_retry(monkeypatch, path, fault):
    torch = torch_mock(monkeypatch)
    f, stream, extra = frame(), Stream(), Tensor(50000)
    owner = types.SimpleNamespace(_cpp_program_frame=f)
    graph = raw = None
    has_graph = path not in {'frame_completion', 'module_completion'}
    if has_graph:
        graph = f.capture(f.roots, stream, torch, keepalive=(extra,))
        raw = torch.cuda.last_graph
    else:
        f.submit(f.roots, stream)
    extra_ref = weakref.ref(extra)
    if has_graph:
        del extra
    original_roots = f.roots
    operations = {
        'frame_completion': f.synchronize,
        'module_completion': lambda: module_method('complete_cpp_programs')(owner, stream),
        'frame_graph_completion': f.synchronize,
        'frame_retirement': f.retire_graphs,
        'wrapper_close': lambda: graph.close(),
        'module_retirement': lambda: module_method('retire_cpp_program_graphs')(owner),
    }
    if fault == 'sync':
        stream.fail = True
    else:
        raw.fail_reset = True
    try:
        with pytest.raises(RuntimeError, match='wait failed|reset failed') as failure:
            operations[path]()
        error = failure.value
        assert f.poisoned is error
        assert f.pending and id(f) in cpp_programs._epoch_pending_holds
        if has_graph:
            assert f.captured and not graph._closed and graph._graph is raw
            gc.collect()
            assert extra_ref() is not None
        assert any(('synchronize' if fault == 'sync' else 'graph reset') in stage and observed is error
                   for stage, observed in f.cleanup_failures)
        calls, replays = len(f.library.calls), len(raw.replays) if raw else 0
        for operation in (
            lambda: f.submit(original_roots, stream),
            lambda: f._bind([Tensor(60000)]),
            lambda: f.capture(original_roots, stream, torch),
        ):
            with pytest.raises(RuntimeContractError, match='quarantined'):
                operation()
        if graph:
            with pytest.raises(RuntimeContractError, match='quarantined'):
                graph.replay()
        assert f.roots == original_roots and len(f.library.calls) == calls
        assert (len(raw.replays) if raw else 0) == replays
        with pytest.raises(RuntimeContractError, match='retained'):
            f.require_releasable()
        with pytest.raises(RuntimeContractError, match='aliases'):
            frame()
        # Retry the real cleanup, without resetting poison or accepting a bool.
        stream.fail = False
        if raw:
            raw.fail_reset = False
        if graph:
            graph.close()
            assert raw.resets >= 1 and graph._closed
        else:
            f.synchronize()
        assert not f.pending and not f.captured and f.poisoned is error
        with pytest.raises(RuntimeContractError, match='quarantined'):
            f.submit(original_roots, stream)
        if graph:
            with pytest.raises(RuntimeContractError, match='quarantined'):
                graph.replay()
        assert len(f.library.calls) == calls and (len(raw.replays) if raw else 0) == replays
        f.require_releasable()
    finally:
        # Also clean the intentionally failing old-source run, without leaks.
        stream.fail = False
        if raw:
            raw.fail_reset = False
        f.retire_graphs()
        f.release()
    fresh = frame()
    fresh.submit(fresh.roots, stream)
    assert len(fresh.library.calls) == 1
    fresh.synchronize()
    fresh.release()


def test_cleanup_retries_preserve_first_error_and_hold_after_prior_completion(monkeypatch):
    torch = torch_mock(monkeypatch)
    f, stream = frame(), Stream()
    graph = f.capture(f.roots, stream, torch)
    raw = torch.cuda.last_graph
    f.synchronize()
    assert not f.pending and f.captured
    try:
        raw.fail_reset = True
        with pytest.raises(RuntimeError, match='reset failed') as first:
            graph.close()
        assert f.poisoned is first.value and f.pending
        stream.fail = True
        with pytest.raises(RuntimeError, match='wait failed') as second:
            f.retire_graphs()
        assert f.poisoned is first.value and f.poisoned is not second.value
        assert f.cleanup_failures[0][1] is first.value
        assert 'synchronize' in f.cleanup_failures[-1][0] and f.cleanup_failures[-1][1] is second.value
        assert id(f) in cpp_programs._epoch_pending_holds and not graph._closed
        stream.fail = False
        raw.fail_reset = False
        graph.close()
        assert f.poisoned is first.value and not f.pending and not f.captured
        with pytest.raises(RuntimeContractError, match='quarantined'):
            f.submit(f.roots, stream)
    finally:
        stream.fail = raw.fail_reset = False
        f.retire_graphs()
        f.release()


def test_cleanup_failure_does_not_replace_native_ordinal_and_stage():
    lib, stream = Library(), Stream()
    f = frame(lib)
    native_error = cpp_programs.CounterEpochLaunchError(
        71, PlanResult(1, 32, 8, PlanStage.RESET, 71, 8, 0, 0))
    lib.failure = native_error
    try:
        with pytest.raises(cpp_programs.CounterEpochLaunchError):
            f.submit(f.roots, stream)
        stream.fail = True
        with pytest.raises(RuntimeError, match='wait failed'):
            f.synchronize()
        assert f.poisoned is native_error
        assert (f.poisoned.ordinal, f.poisoned.stage, f.poisoned.submitted) == (8, PlanStage.RESET, 8)
        stream.fail = False
        f.synchronize()
        assert f.poisoned is native_error and not f.pending
        with pytest.raises(RuntimeContractError, match='quarantined'):
            f.submit(f.roots, stream)
    finally:
        stream.fail = False
        f.retire_graphs()
        f.release()

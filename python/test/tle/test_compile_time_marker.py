# Copyright 2026- Xcoresigma Technology Co., Ltd

import ast
from pathlib import Path

from triton.experimental.tle.language.dsa.ascend.core import PIPE, SyncSpec
from triton.experimental.tle.language.dsa.core import Workspace
from triton.experimental.tle.language.dsa.ascend.pipe import PipeEndpoint, PipeSlot, PipeWaitResult


class FakeLanguage:

    class dtype:
        pass

    class tuple:
        pass


def load_is_native_nontensor_value():
    # Ascend compilation runs the spec copy of the code generator; that is the
    # one the marker check needs to stay in sync with.
    source_path = Path(__file__).parents[2] / "triton/spec/ascend/compiler/code_generator.py"
    module = ast.parse(source_path.read_text())
    function_node = next(node for node in module.body
                         if isinstance(node, ast.FunctionDef) and node.name == "_is_native_nontensor_value")
    namespace = {"language": FakeLanguage}
    exec(compile(ast.Module(body=[function_node], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["_is_native_nontensor_value"]


_is_native_nontensor_value = load_is_native_nontensor_value()


class ObjectWithCommonAttribute:
    kind = "not-a-tle-descriptor"


class ExplicitCompileTimeValue:
    __triton_compile_time_value__ = True


def test_compile_time_marker_is_required_for_non_tensor_values():
    assert _is_native_nontensor_value(FakeLanguage.dtype())
    assert _is_native_nontensor_value(ExplicitCompileTimeValue())
    assert not _is_native_nontensor_value(ObjectWithCommonAttribute())
    assert not _is_native_nontensor_value(type)


def test_tle_pipe_objects_are_compile_time_values():
    slot = PipeSlot({})
    wait_result = PipeWaitResult(slot)
    endpoint = PipeEndpoint(object(), "writer")
    sync = SyncSpec("cube", "vector", PIPE.PIPE_FIX, PIPE.PIPE_MTE2)
    workspace = Workspace("base", capacity=2, shape=[4, 8], dtype="float32")

    assert _is_native_nontensor_value(slot)
    assert _is_native_nontensor_value(wait_result)
    assert _is_native_nontensor_value(endpoint)
    assert _is_native_nontensor_value(sync)
    assert _is_native_nontensor_value(workspace)

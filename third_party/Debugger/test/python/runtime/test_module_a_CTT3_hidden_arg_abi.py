# SPDX-License-Identifier: MIT
"""CTT-3 + §5.1.1: ``hidden_arg`` / ``prepare_launch_debug_ctrl`` 与当前 launch 注入路径一致。

契约 ID: **CTT-3**, **§5.1.1**；使用轻量测试替身，无需 GPU。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_unit = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_module_a_doc", _unit / "_module_a_doc.py")
_mad = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mad)
__doc__ = _mad.extend_doc(__doc__)

pytest.importorskip("triton.backends.nvidia.driver")
from triton.backends.nvidia import driver as nvidia_driver
from triton.compiler.flagtree_debug import prepare_launch_debug_ctrl


def _launcher_meta(debug_enabled=True, debug_launch_hidden_arg=True):
    return SimpleNamespace(
        tensordesc_meta=None,
        global_scratch_size=0,
        global_scratch_align=1,
        profile_scratch_size=0,
        profile_scratch_align=1,
        num_ctas=1,
        launch_cooperative_grid=False,
        launch_pdl=False,
        debug_enabled=debug_enabled,
        debug_launch_hidden_arg=debug_launch_hidden_arg,
    )


def _launcher_src():
    return SimpleNamespace(
        constants={},
        fn=SimpleNamespace(arg_names=["x"]),
        signature={0: "*fp32"},
    )


def _launch_module():
    module = SimpleNamespace()

    def launch(*args, **kwargs):
        del kwargs
        module.seen = args

    module.launch = launch
    return module


@pytest.mark.module_a
@pytest.mark.module_a_ctt3
def test_module_a_CTT3_prepare_launch_sets_tail_abi_value():
    k = SimpleNamespace()
    k._run = SimpleNamespace()
    k._run.debug_ctrl_ptr = 0
    k._run.debug_launch_hidden_arg = True
    k._debug_ctrl_ptr = 0xABCDEF01
    prepare_launch_debug_ctrl(k, stream=None)
    assert k._run.debug_ctrl_ptr == 0xABCDEF01


@pytest.mark.module_a
@pytest.mark.module_a_ctt3
def test_module_a_CTT3_hidden_arg_is_last_launch_tuple_element(monkeypatch):
    meta = _launcher_meta()
    src = _launcher_src()
    launch_mod = _launch_module()
    prepare_calls = []
    finalize_calls = []

    def prepare_launch(*args):
        prepare_calls.append(args)
        return SimpleNamespace(kernel_args=(0x11223344, ))

    def finalize_launch(*args):
        finalize_calls.append(args)

    monkeypatch.setattr(nvidia_driver, "compile_module_from_src", lambda *args, **kwargs: launch_mod)
    monkeypatch.setattr(nvidia_driver, "library_dirs", lambda: [])
    monkeypatch.setattr(nvidia_driver, "prepare_kernel_launch", prepare_launch)
    monkeypatch.setattr(nvidia_driver, "finalize_prepared_launch", finalize_launch)

    launcher = nvidia_driver.CudaLauncher(src, meta)
    stream = object()
    fn = object()
    launch_metadata = {"grid": (2, 1, 1)}
    launcher(2, 1, 1, stream, fn, object(), launch_metadata, None, None, 99)
    assert launch_mod.seen[-2:] == (0x11223344, 99)
    assert prepare_calls == [(meta, stream, launch_metadata, (99, ))]
    assert len(finalize_calls) == 1


@pytest.mark.module_a
@pytest.mark.module_a_ctt3
def test_module_a_511_prepare_launch_debug_ctrl_updates_real_launcher(monkeypatch):
    """§5.1.1：``prepare_launch_debug_ctrl`` 会把控制句柄写入真实 launcher。"""

    meta = _launcher_meta()
    src = _launcher_src()
    launch_mod = _launch_module()

    compiled = SimpleNamespace()
    compiled.metadata = meta
    compiled._debug_ctrl_ptr = 0x55667788
    compiled._run = None

    monkeypatch.setattr(nvidia_driver, "compile_module_from_src", lambda *args, **kwargs: launch_mod)
    monkeypatch.setattr(nvidia_driver, "library_dirs", lambda: [])

    compiled._init_handles = lambda: setattr(compiled, "_run", nvidia_driver.CudaLauncher(src, meta))
    compiled._init_handles()
    prepare_launch_debug_ctrl(compiled, stream=None)

    assert compiled._run.debug_launch_hidden_arg is True
    assert compiled._run.debug_ctrl_ptr == 0x55667788

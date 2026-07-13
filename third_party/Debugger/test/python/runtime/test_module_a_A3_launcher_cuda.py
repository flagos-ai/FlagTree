# SPDX-License-Identifier: MIT
"""A-3: ``make_launcher`` / ``CudaLauncher`` 在 debug kernel 下走 ``debug_hidden_arg`` 注入路径。

契约 ID: **A-3**；使用轻量测试替身，无需 libcuda/GPU。
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


def _minimal_launch_mod():
    module = SimpleNamespace()

    def launch(*args, **kwargs):
        del kwargs
        module.last_launch_args = args

    module.launch = launch
    return module


def _launcher_src():
    return SimpleNamespace(
        constants={},
        fn=SimpleNamespace(arg_names=["x"]),
        signature={0: "*fp32"},
    )


@pytest.mark.module_a
@pytest.mark.module_a_a3
def test_module_a_A3_make_launcher_includes_debug_ctrl_when_enabled():
    src_off = nvidia_driver.make_launcher({}, {0: "*fp32"}, None, debug_enabled=False)
    assert "debug_hidden_arg" not in src_off

    src_on = nvidia_driver.make_launcher({}, {0: "*fp32"}, None, debug_enabled=True)
    params_line = "void *params[] = { &arg0, &debug_hidden_arg, &global_scratch, &profile_scratch };"
    assert "uint64_t debug_hidden_arg" in src_on
    assert params_line in src_on


@pytest.mark.module_a
@pytest.mark.module_a_a3
def test_module_a_A3_cuda_launcher_injects_debug_hidden_arg_from_prepare_hook(monkeypatch):

    def make_meta(debug_enabled, debug_launch_hidden_arg):
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

    src = _launcher_src()

    launch_mod = _minimal_launch_mod()
    meta_off = make_meta(False, False)
    meta_on = make_meta(True, True)
    prepare_off_calls = []
    finalize_off_calls = []
    prepare_on_calls = []
    finalize_on_calls = []

    monkeypatch.setattr(nvidia_driver, "compile_module_from_src", lambda *args, **kwargs: launch_mod)
    monkeypatch.setattr(nvidia_driver, "library_dirs", lambda: [])

    def prepare_off(*args):
        prepare_off_calls.append(args)
        return None

    def finalize_off(*args):
        finalize_off_calls.append(args)

    monkeypatch.setattr(nvidia_driver, "prepare_kernel_launch", prepare_off)
    monkeypatch.setattr(nvidia_driver, "finalize_prepared_launch", finalize_off)

    launcher_off = nvidia_driver.CudaLauncher(src, meta_off)
    stream = object()
    fn = object()
    packed_metadata = object()
    launch_metadata = {"grid": (4, 1, 1)}

    launcher_off(4, 1, 1, stream, fn, packed_metadata, launch_metadata, None, None, 99)
    assert launch_mod.last_launch_args[-1] == 99
    assert prepare_off_calls == [(meta_off, stream, launch_metadata, (99, ))]
    assert len(finalize_off_calls) == 1

    prepared = SimpleNamespace(kernel_args=(0xBEEF, ))

    def prepare_on(*args):
        prepare_on_calls.append(args)
        return prepared

    def finalize_on(*args):
        finalize_on_calls.append(args)

    monkeypatch.setattr(nvidia_driver, "prepare_kernel_launch", prepare_on)
    monkeypatch.setattr(nvidia_driver, "finalize_prepared_launch", finalize_on)

    launcher_on = nvidia_driver.CudaLauncher(src, meta_on)
    stream = object()
    fn = object()
    packed_metadata = object()
    launch_metadata = {"grid": (4, 1, 1)}

    launcher_on(4, 1, 1, stream, fn, packed_metadata, launch_metadata, None, None, 99)
    assert launcher_on.debug_launch_hidden_arg is True
    assert launch_mod.last_launch_args[-2:] == (0xBEEF, 99)
    assert prepare_on_calls == [(meta_on, stream, launch_metadata, (99, ))]
    assert finalize_on_calls == [(prepared, None)]

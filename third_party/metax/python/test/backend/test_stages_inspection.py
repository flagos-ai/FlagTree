# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import triton
from triton import knobs

import os
import pathlib


def test_inspection(monkeypatch, tmp_path: pathlib.Path):
    stage_name = 'make_ttgir'
    curr_repro_path = tmp_path / ("repro_prefix." + stage_name + ".repro.mlir")
    repro_path = tmp_path / "repro_prefix"

    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    monkeypatch.setenv("TRITON_REPRODUCER_PATH", str(repro_path))

    inspect_stages_hook_called = False
    make_ttgir_wrapper_called = False

    def inspect_stages_hook(self, stages, options, language, capability):
        nonlocal inspect_stages_hook_called
        inspect_stages_hook_called = True

        def make_ttgir_wrapper(src, metadata, options, capability):
            nonlocal make_ttgir_wrapper_called
            make_ttgir_wrapper_called = True
            return self.make_ttgir(src, metadata, options, capability)

        stages["ttgir"] = lambda src, metadata: make_ttgir_wrapper(src, metadata, options, capability)

    @triton.jit
    def k1():
        return

    @triton.jit
    def k2():
        return

    # Run once to get the clean/golden repro dump
    k1[(1, )]()
    assert not inspect_stages_hook_called and not make_ttgir_wrapper_called
    assert os.path.exists(curr_repro_path)
    golden_repro = curr_repro_path.read_text()
    curr_repro_path.unlink()

    # Setup hook and call again, check if hooks got called
    knobs.runtime.add_stages_inspection_hook = inspect_stages_hook
    k2[(1, )]()
    assert inspect_stages_hook_called and make_ttgir_wrapper_called
    assert os.path.exists(curr_repro_path)
    hook_repro = curr_repro_path.read_text()

    # Check that repros match
    assert golden_repro.replace('k1', 'dummy') == hook_repro.replace('k2', 'dummy')

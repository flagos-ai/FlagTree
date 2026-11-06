"""CPU-only checks of unit-dimension views crossing LLVM versions."""
import importlib.util
import os
from pathlib import Path
import subprocess

import pytest

path = Path(__file__).resolve().parents[2] / 'backend/structured_permute_ir.py'
spec = importlib.util.spec_from_file_location('unit_view_late', path)
late = importlib.util.module_from_spec(spec)
spec.loader.exec_module(late)


@pytest.fixture(scope='module', autouse=True)
def assign_npu():
    yield


def _module(shape='1x?x?', strides='512, 725760, 1', output='?x?', output_strides='725760, 1', space='gm', offset='?',
            dtype='f32'):
    source = f'memref<{shape}x{dtype}, strided<[{strides}], offset: {offset}>, #hivm.address_space<{space}>>'
    target = f'memref<{output}x{dtype}, strided<[{output_strides}], offset: {offset}>, #hivm.address_space<{space}>>'
    return f'''module {{
  func.func @probe(%arg0: {source}) -> {target} {{
    %r = "memref.collapse_shape"(%arg0) <{{reassociation = [[0, 1], [2]]}}> : ({source}) -> {target}
    return %r : {target}
  }}
}}
'''


@pytest.mark.parametrize('space,dtype,offset', [('gm', 'f32', '?'), ('ub', 'i32', '7')])
def test_unit_drop_keeps_addressing(space, dtype, offset):
    source = _module(space=space, dtype=dtype, offset=offset)
    result = late.lower_unit_collapses(source)
    assert 'memref.collapse_shape' not in result
    assert 'memref.subview %arg0[0, 0, 0]' in result
    assert source.split('return %r : ')[1] == result.split('return %r : ')[1]
    assert late.lower_unit_collapses(result) == result


def test_static_sizes_and_ssa_collision():
    text = _module(shape='1x5x9', output='5x9')
    result = late.lower_unit_collapses(text)
    assert '[1, 5, 9] [1, 1, 1]' in result
    assert 'memref.dim' not in result
    text = _module().replace('%arg0:', '%flagtree_unit_view_0_size:').replace('%arg0)', '%flagtree_unit_view_0_size)')
    result = late.lower_unit_collapses(text)
    assert '%flagtree_unit_view__0_size = memref.dim' in result


@pytest.mark.parametrize('text', [
    _module(shape='2x?x?'),
    _module(output_strides='123, 1'),
    _module().replace('[[0, 1], [2]]', '[[0], [1, 2]]'),
    _module().replace('reassociation =', 'extra_marker, reassociation =')
])
def test_unproven_collapses_untouched(text):
    assert late.lower_unit_collapses(text) == text


def test_native_verifiers(tmp_path):
    old = Path(os.environ.get('UNIT_VIEW_OLD_OPT', '/usr/local/Ascend/cann-9.0.0/bin/bishengir-opt'))
    new_path = os.environ.get('UNIT_VIEW_NEW_OPT')
    if new_path is None:
        import triton
        new = Path(triton.__file__).resolve().parent / '_C/triton-mlir-opt'
    else:
        new = Path(new_path)
    if not old.is_file() or not new.is_file():
        pytest.skip('Requires the two recorded MLIR toolchains, no NPU required')
    before, after = tmp_path / 'before.mlir', tmp_path / 'after.mlir'
    before.write_text(_module())
    after.write_text(late.lower_unit_collapses(before.read_text()))
    for tool, accepted in [(old, before), (old, after), (new, after)]:
        run = subprocess.run([str(tool), str(accepted), '--allow-unregistered-dialect', '-o', os.devnull],
                             capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
    rejected = subprocess.run([str(new), str(before), '--allow-unregistered-dialect', '-o', os.devnull],
                              capture_output=True, text=True)
    assert rejected.returncode != 0 and 'expected collapsed type' in rejected.stderr

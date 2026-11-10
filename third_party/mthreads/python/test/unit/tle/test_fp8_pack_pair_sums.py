"""FP8 packing may bind pair sums without changing any defined FP lane."""
import re

import pytest
from triton._C.libtriton import ir, llvm, mthreads
from test_tle_utils import mthreads_backend, require_mthreads_libtriton

require_mthreads_libtriton()


def transform(tmp_path, strict=False, volatile=False, extra_lane=False, missing_pair=False, reverse=False, reuse=False,
              final=False, vector_final=False, extra_pair_use=False):
    attributes = 'attributes {passthrough = ["strictfp"]}' if strict else ''
    effects = 'has_side_effects' if volatile else ''
    mask = '[1, 0, -1, -1]' if extra_lane else '[1, -1, -1, -1]'
    second = '%v' if missing_pair else '%s2'
    lhs, rhs = ('%s0', '%v') if reverse else ('%v', '%s0')
    tail = '''%s4 = llvm.shufflevector %b, %nil [2, -1, -1, -1] : vector<4xf32>
    %v4 = llvm.fadd %a, %s4 : vector<4xf32>
    %sum = llvm.extractelement %v4[%i0 : i32] : vector<4xf32>''' if vector_final else '''%lo = llvm.extractelement %a[%i0 : i32] : vector<4xf32>
    %hi = llvm.extractelement %b[%i2 : i32] : vector<4xf32>
    %sum = llvm.fadd %lo, %hi : f32'''
    extra = 'llvm.call @observe_vec(%a) : (vector<4xf32>) -> ()' if extra_pair_use else ''
    path = tmp_path / 'pack-pair.mlir'
    path.write_text(f'''
module {{
  llvm.func @observe(i32)
  llvm.func @observe_vec(vector<4xf32>)
  llvm.func @probe(%v: vector<4xf32>) -> f32 {attributes} {{
    %nil = llvm.mlir.poison : vector<4xf32>
    %i0 = llvm.mlir.constant(0 : i32) : i32
    %i2 = llvm.mlir.constant(2 : i32) : i32
    %p = llvm.inline_asm {effects} "CVT.RTE.DENORM.BST4 $0.e4m3, $1.f32", "=R,R"
        %v : (vector<4xf32>) -> i32
    %s0 = llvm.shufflevector %v, %nil {mask} : vector<4xf32>
    %a = llvm.fadd {lhs}, {rhs} : vector<4xf32>
    %s2 = llvm.shufflevector %v, %nil [-1, -1, 3, -1] : vector<4xf32>
    %b = llvm.fadd %v, {second} : vector<4xf32>
    {tail}
    {extra}
    llvm.call @observe(%p) : (i32) -> ()
    llvm.return %sum : f32
  }}
}}
''')
    context = ir.context()
    ir.load_dialects(context)
    _, backend = mthreads_backend()
    backend.load_dialects(context)
    module = ir.parse_mlir_module(str(path), context)
    llvm_context = llvm.context()
    result = llvm.to_module(module, llvm_context)
    before = str(result)
    count = mthreads.fuse_fp8_pack_pair_sums(result)
    return count, before, str(result)


@pytest.mark.parametrize('reverse', [False, True])
def test_two_pair_operations_and_final_sum_are_preserved(tmp_path, reverse):
    count, before, after = transform(tmp_path, reverse=reverse)
    assert count == 1
    assert before.count(' = fadd float') == after.count(' = fadd float') == 1
    assert 'FOP.ADD $1, $4, $5' in after and 'FOP.ADD $2, $6, $7' in after
    assert '"=&R,=&R,=R,R,R,R,R,R"' in after
    assert 'call { i32, float, float } asm' in after
    assert len(re.findall(r'extractvalue \{ i32, float, float \}', after)) == 3
    assert after.count('call void @observe') == before.count('call void @observe')
    assert 'fast ' not in after and 'alloca ' not in after


@pytest.mark.parametrize('option', ['strict', 'volatile', 'extra_lane', 'missing_pair'])
def test_nonmatching_or_strict_operations_are_unchanged(tmp_path, option):
    count, before, after = transform(tmp_path, **{option: True})
    assert count == 0
    assert before == after


def test_option_default_and_cache():
    _, backend = mthreads_backend()
    assert not backend.parse_options({}).fuse_fp8_pack_pair_sums
    assert backend.parse_options({}).hash() != backend.parse_options({'fuse_fp8_pack_pair_sums': True}).hash()
    for invalid in [1, 'true', None]:
        with pytest.raises(ValueError, match='fuse_fp8_pack_pair_sums must be a bool'):
            backend.parse_options({'fuse_fp8_pack_pair_sums': invalid})

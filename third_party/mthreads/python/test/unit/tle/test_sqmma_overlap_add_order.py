"""Real scalar RNE adds may anchor compiler ordering, never replace GPU waits."""
import pytest
import re
from triton._C.libtriton import ir, llvm, mthreads
from test_tle_utils import mthreads_backend, require_mthreads_libtriton

require_mthreads_libtriton()


def transform(tmp_path, strict=False, int8=False, early_wait=False, width=1, partial=False, shuffled=False,
              observe=False, two=False, limit=0, native_rest=False, canonicalize=True, tied=False):
    suffix = 's8' if int8 else 'e4m3'
    attrs = 'attributes {passthrough = ["strictfp"]}' if strict else ''
    wait = 'llvm.call @llvm.musa.sqmma.wait() : () -> ()'
    name = f'llvm.musa.sqmma.{suffix}.m32n128k128.mma'
    dtype = 'f32' if width == 1 else f'vector<{width}xf32>'
    result_type = 'f32' if partial else dtype
    tail = ''
    if shuffled:
        tail += f'%permuted = llvm.shufflevector %after, %after [3, 2, 1, 0] : {dtype}\n'
    if partial:
        tail += f'%lane = llvm.mlir.constant(1 : i64) : i64\n'
        tail += f'%scalar = llvm.extractelement {"%permuted" if shuffled else "%after"}[%lane : i64] : {dtype}\n'
    tail += f'llvm.return {"%scalar" if partial else "%after"} : {result_type}'
    types = ', '.join(['i32', 'i32', 'vector<32xi32>'] + ['i32'] * 9)
    args = ', '.join(['%d0', '%d1', '%zero'] + ['%c0'] * 9)
    path = tmp_path / 'overlap.mlir'
    path.write_text(f'''module {{
  llvm.func @{name}({types}) -> vector<32xi32>
  llvm.func @llvm.musa.sqmma.wait()
  llvm.func @observe({dtype})
  llvm.func @probe(%d0: i32, %d1: i32, %zero: vector<32xi32>, %x: {dtype}, %y: {dtype}) -> {result_type} {attrs} {{
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %before = llvm.fadd %x, %y : {dtype}
    %mma = llvm.call @{name}({args}) : ({types}) -> vector<32xi32>
    {wait if early_wait else ''}
    %inside = llvm.fadd %before, %x : {dtype}
    {'%second = llvm.fadd %inside, %y : ' + dtype if two else ''}
    {'llvm.call @observe(%inside) : (' + dtype + ') -> ()' if observe else ''}
    {'' if early_wait else wait}
    %after = llvm.fadd {"%second" if two else "%inside"}, %y : {dtype}
    {tail}
  }}
}}''')
    context = ir.context()
    ir.load_dialects(context)
    _, backend = mthreads_backend()
    backend.load_dialects(context)
    module = ir.parse_mlir_module(str(path), context)
    llvm_context = llvm.context()
    result = llvm.to_module(module, llvm_context)
    before = str(result)
    count = mthreads.order_sqmma_overlap_adds(result, limit, tied)
    return count, before, str(result)


def test_only_scalar_add_inside_pending_fp8_region_is_ordered(tmp_path):
    count, before, after = transform(tmp_path)
    assert count == 1
    assert before.count(' = fadd float') == 3
    assert after.count(' = fadd float') == 2
    assert 'asm sideeffect "FOP.ADD $0, $1, $2", "=R,R,R,~{memory}"' in after
    assert before.count('call void @llvm.musa.sqmma.wait') == after.count('call void @llvm.musa.sqmma.wait') == 1
    assert 'fast ' not in after and 'fence ' not in after


@pytest.mark.parametrize('width', [2, 4])
def test_componentwise_vector_adds_keep_their_width(tmp_path, width):
    count, before, after = transform(tmp_path, width=width)
    assert count == 1
    assert before.count(f' = fadd <{width} x float>') == 3
    assert after.count(f' = fadd <{width} x float>') == 2
    assert f'asm sideeffect "FOP.ADD.BST{width} $0, $1, $2"' in after
    assert before.count('call void @llvm.musa.sqmma.wait') == after.count('call void @llvm.musa.sqmma.wait')


@pytest.mark.parametrize('shuffled,expected_lane', [(False, 1), (True, 2)])
def test_only_the_observed_component_needs_a_scalar_instruction(tmp_path, shuffled, expected_lane):
    count, before, after = transform(tmp_path, width=4, partial=True, shuffled=shuffled)
    assert count == 1
    assert 'FOP.ADD.BST4' not in after
    assert after.count('asm sideeffect "FOP.ADD $0, $1, $2"') == 1
    assert len(re.findall(rf'extractelement <4 x float> %\w+, i(?:32|64) {expected_lane}', after)) >= 2


def test_unknown_vector_consumer_demands_every_lane(tmp_path):
    count, before, after = transform(tmp_path, width=4, partial=True, observe=True)
    assert count == 1
    assert 'asm sideeffect "FOP.ADD.BST4 $0, $1, $2"' in after


@pytest.mark.parametrize('limit,expected', [(0, 2), (1, 1), (2, 2)])
def test_partial_ordering_budget_changes_no_addition(tmp_path, limit, expected):
    count, before, after = transform(tmp_path, two=True, limit=limit)
    assert count == expected
    assert (after.count(' = fadd float') + after.count('asm sideeffect "FOP.ADD') + after.count('asm "FOP.ADD')) == 4
    assert after.count('asm sideeffect "FOP.ADD') == expected
    assert before.count('call void @llvm.musa.sqmma.wait') == after.count('call void @llvm.musa.sqmma.wait')


@pytest.mark.parametrize('case', ['strict', 'int8', 'early_wait'])
def test_other_regions_are_unchanged(tmp_path, case):
    count, before, after = transform(tmp_path, **{case: True})
    assert count == 0 and before == after


def test_default_off_and_cache_key():
    _, backend = mthreads_backend()
    base = backend.parse_options({})
    assert not base.order_sqmma_overlap_adds
    assert base.hash() != backend.parse_options({'order_sqmma_overlap_adds': True}).hash()
    with pytest.raises(ValueError, match='must be a bool'):
        backend.parse_options({'order_sqmma_overlap_adds': 1})


@pytest.mark.parametrize('width', [1, 2, 4])
@pytest.mark.parametrize('partial', [False, True])
def test_tied_adds_preserve_operands_and_waits(tmp_path, width, partial):
    if partial and width == 1:
        pytest.skip('Scalar has no component extraction')
    count, before, after = transform(tmp_path, width=width, partial=partial, tied=True)
    assert count == 1
    assert '"=R,0,R,~{memory}"' in after
    assert before.count('call void @llvm.musa.sqmma.wait') == after.count('call void @llvm.musa.sqmma.wait') == 1
    assert 'fast ' not in after and 'fence ' not in after


def test_tied_pure_remainder_keeps_all_original_adds(tmp_path):
    count, before, after = transform(tmp_path, two=True, limit=1, tied=True)
    assert count == 1
    assert 'asm sideeffect "FOP.ADD $0, $1, $2", "=R,0,R,~{memory}"' in after
    assert 'asm "FOP.ADD $0, $1, $2", "=R,0,R"' in after
    assert after.count(' = fadd float') + after.count('asm sideeffect "FOP.ADD') + after.count('asm "FOP.ADD') == 4


@pytest.mark.parametrize('case', ['strict', 'int8', 'early_wait'])
def test_tied_mode_cannot_rewrite_other_regions(tmp_path, case):
    count, before, after = transform(tmp_path, tied=True, **{case: True})
    assert count == 0 and before == after


def test_tie_option_is_explicit_and_cached():
    _, backend = mthreads_backend()
    assert not backend.parse_options({}).tie_sqmma_overlap_adds
    with pytest.raises(ValueError, match='requires order_sqmma_overlap_adds'):
        backend.parse_options({'tie_sqmma_overlap_adds': True})
    base = {'order_sqmma_overlap_adds': True}
    assert backend.parse_options(base).hash() != backend.parse_options({**base, 'tie_sqmma_overlap_adds': True}).hash()
    for invalid in [1, None, 'true']:
        with pytest.raises(ValueError, match='must be a bool'):
            backend.parse_options({**base, 'tie_sqmma_overlap_adds': invalid})


def packed_transform(tmp_path, width=4, dependent=False, memory=False, tied=False, strict=False, int8=False,
                     late_view=False):
    suffix = 's8' if int8 else 'e4m3'
    attrs = 'attributes {passthrough = ["strictfp"]}' if strict else ''
    dtype = f'vector<{width}xf32>'
    types = ', '.join(['i32', 'i32', 'vector<32xi32>'] + ['i32'] * 9)
    args = ', '.join(['%d0', '%d1', '%zero'] + ['%c0'] * 9)
    body = []
    for i in range(width):
        body.append(f'%idx{i} = llvm.mlir.constant({i} : i32) : i32')
        body.append(f'%x{i} = llvm.extractelement %x[%idx{i} : i32] : {dtype}')
        body.append(f'%y{i} = llvm.extractelement %y[%idx{i} : i32] : {dtype}')
        if memory and i == width // 2:
            body.append('llvm.call @observe(f32 %sum0) : (f32) -> ()'.replace('f32 %sum0', '%sum0'))
        a = f'%sum{i-1}' if dependent and i else f'%x{i}'
        if late_view and i:
            body.append(f'%view{i} = llvm.insertelement %sum{i-1}, %x[%idx0 : i32] : {dtype}')
            body.append(f'%pick{i} = llvm.extractelement %view{i}[%idx0 : i32] : {dtype}')
            a = f'%pick{i}'
        body.append(f'%sum{i} = llvm.fadd {a}, %y{i} : f32')
    body.append(f'%nil = llvm.mlir.poison : {dtype}')
    for i in range(width):
        body.append(
            f'%r{i} = llvm.insertelement %sum{i}, {"%nil" if not i else "%r"+str(i-1)}[%idx{i} : i32] : {dtype}')
    body.append('llvm.call @llvm.musa.sqmma.wait() : () -> ()')
    body.append(f'llvm.return %r{width-1} : {dtype}')
    path = tmp_path / 'pack-adds.mlir'
    path.write_text(f"""module {{
      llvm.func @llvm.musa.sqmma.{suffix}.m32n128k128.mma({types}) -> vector<32xi32>
      llvm.func @llvm.musa.sqmma.wait()
      llvm.func @observe(f32)
      llvm.func @probe(%d0: i32, %d1: i32, %zero: vector<32xi32>, %x: {dtype}, %y: {dtype}) -> {dtype} {attrs} {{
        %c0 = llvm.mlir.constant(0 : i32) : i32
        %mma = llvm.call @llvm.musa.sqmma.{suffix}.m32n128k128.mma({args}) : ({types}) -> vector<32xi32>
        {chr(10).join(body)}
      }}
    }}""")
    context = ir.context()
    ir.load_dialects(context)
    _, backend = mthreads_backend()
    backend.load_dialects(context)
    module = ir.parse_mlir_module(str(path), context)
    llvm_context = llvm.context()
    result = llvm.to_module(module, llvm_context)
    mthreads.order_sqmma_overlap_adds(result, 0, tied)
    before = str(result)
    count = mthreads.pack_sqmma_overlap_adds(result)
    return count, before, str(result)


@pytest.mark.parametrize('width,groups', [(2, 1), (3, 1), (4, 1), (8, 2)])
@pytest.mark.parametrize('tied', [False, True])
def test_pack_independent_add_lanes_without_reassociation(tmp_path, width, groups, tied):
    count, before, after = packed_transform(tmp_path, width=width, tied=tied)
    assert count == groups
    assert f'FOP.ADD.BST{4 if width >= 4 else 2}' in after
    assert ('"=R,0,R,~{memory}"' if tied else '"=R,R,R,~{memory}"') in after
    assert before.count('call void @llvm.musa.sqmma.wait') == after.count('call void @llvm.musa.sqmma.wait') == 1
    assert 'fast ' not in after and 'fence ' not in after


@pytest.mark.parametrize('case', ['dependent', 'late_view', 'strict', 'int8'])
def test_packing_cannot_cross_add_dependencies_or_other_regions(tmp_path, case):
    count, before, after = packed_transform(tmp_path, **{case: True})
    assert count == 0 and before == after


def test_pack_stops_at_unknown_memory_call(tmp_path):
    count, before, after = packed_transform(tmp_path, memory=True)
    assert count == 2 and after.count('FOP.ADD.BST2') == 2
    assert before.count('call void @observe') == after.count('call void @observe') == 1


def test_pack_add_option_is_explicit_and_cached():
    _, backend = mthreads_backend()
    assert not backend.parse_options({}).pack_sqmma_overlap_adds
    with pytest.raises(ValueError, match='requires order_sqmma_overlap_adds'):
        backend.parse_options({'pack_sqmma_overlap_adds': True})
    base = {'order_sqmma_overlap_adds': True}
    assert backend.parse_options(base).hash() != backend.parse_options({**base, 'pack_sqmma_overlap_adds': True}).hash()
    for invalid in [0, None, 'true']:
        with pytest.raises(ValueError, match='must be a bool'):
            backend.parse_options({**base, 'pack_sqmma_overlap_adds': invalid})

# buffer store pairing
"""HCU scalar buffer stores combine only when both halfwords can be written."""

import re

import pytest
import triton
from triton._C import libtriton

from . import hcu_test_utils as memory_rules
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource
from triton.experimental.gluon.language.amd.cdna3 import buffer_store

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler required", allow_module_level=True)


@g.jit
def _buffer_store_pairing_store(C, BOUND: gl.constexpr, START: gl.constexpr, STRIDE: gl.constexpr):
    layout: gl.constexpr = gl.DistributedLinearLayout([[2], [1]], [[4], [8], [16], [32], [64], [128]], [], [], [256])
    i = gl.arange(0, 256, layout=layout)
    offset = START + i * STRIDE
    buffer_store((i + 19).to(gl.int16), C, offset, i < BOUND)


@pytest.mark.parametrize(
    "bound,start,stride,alignment,expected",
    [
        (254, 0, 1, 16, "dword"),
        (253, 0, 1, 16, "short"),
        (254, 1, 1, 16, "short"),
        (254, 0, 2, 16, "short"),
        (254, 0, 1, 2, "short"),
        # The last scalar offset is in range but a pair there would cross the
        # end of the gfx936 raw descriptor.
        (256, 0x3FFFFF00, 1, 16, "mixed"),
    ],
)
def test_buffer_store_pairs_preserve_mask_alignment_and_bounds(bound, start, stride, alignment, expected):
    constants = {"BOUND": bound, "START": start, "STRIDE": stride}
    source = GluonASTSource(
        _buffer_store_pairing_store,
        {"C": "*i16", **dict.fromkeys(constants, "constexpr")},
        constants,
        attrs={(0, ): [("tt.divisibility", alignment)]},
    )
    kernel = triton.compile(source, target=GPUTarget("hip", "gfx936", 64), options={"num_warps": 1})
    stores = re.findall(r"^\s*buffer_store_(\w+)\s", kernel.asm["amdgcn"], re.M)
    expected_stores = {
        "dword": ["dword"] * 2,
        "short": ["short"] * 4,
        "mixed": ["dword", "short", "short"],
    }
    assert sorted(stores) == expected_stores[expected]


@g.jit
def _buffer_store_pairing_store_from_origin(C):
    layout: gl.constexpr = gl.DistributedLinearLayout([[2], [1]], [[4], [8], [16], [32], [64], [128]], [], [], [256])
    i = gl.arange(0, 256, layout=layout)
    origin = gl.program_id(0) * 256
    buffer_store(i.to(gl.int16), C, origin + i, i < 254)


def test_buffer_store_pairs_skip_possible_descriptor_boundary():
    source = GluonASTSource(
        _buffer_store_pairing_store_from_origin,
        {"C": "*i16"},
        attrs={(0, ): [("tt.divisibility", 16)]},
    )
    kernel = triton.compile(source, target=GPUTarget("hip", "gfx936", 64), options={"num_warps": 1})
    stores = re.findall(r"^\s*buffer_store_(\w+)\s", kernel.asm["amdgcn"], re.M)
    assert sorted(stores) == ["dword", "short", "short"]


@g.jit
def _buffer_store_address_modes(C):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    index = gl.arange(0, 128, layout=layout)
    # Bound the complete offset for every possible program-id bit pattern.
    origin = (gl.program_id(0) & 3) * 512
    buffer_store((index + 19).to(gl.int16), C, origin + index + 5, index < 117)


@pytest.mark.parametrize("mode", ["direct", "scalar", "scalar-split", "scalar-fold"])
def test_address_strategies_preserve_payload_and_partial_mask(mode):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires HCU device")
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.arch != "gfx936":
        pytest.skip("requires gfx936 buffer address lowering")
    output = torch.full((4096, ), -1, device="cuda", dtype=torch.int16)
    _buffer_store_address_modes[(4, )](output, num_warps=1, buffer_store_address_mode=mode)
    expected = torch.full((4096, ), -1, dtype=torch.int16)
    for program in range(4):
        begin = program * 512 + 5
        expected[begin:begin + 117] = torch.arange(117, dtype=torch.int16) + 19
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


# factor buffer store offsets

from triton._C.libtriton import llvm

_factor_buffer_store_offsets_INTRINSIC = "llvm.amdgcn.raw.ptr.buffer.store.i32"


def _factor_buffer_store_offsets_fixture(
    mask=255,
    delta=12288,
    masked=False,
    distinct=False,
    soffset=0,
    separate_blocks=False,
    payload="i32",
    suffix="i32",
    cache=0,
):
    intrinsic = "llvm.amdgcn.raw.ptr.buffer.store." + suffix

    def store(resource, offset):
        return (f"  call void @{intrinsic}({payload} %value, ptr addrspace(8) %{resource}, "
                f"i32 %{offset}, i32 {soffset}, i32 {cache})\n")

    ir = f"""define amdgpu_kernel void @mm(ptr addrspace(8) %r, ptr addrspace(8) %r2,
                                  i32 %row, i32 %col, {payload} %value, i1 %valid) {{
entry:
  %left = mul i32 %row, 3072
  %right = and i32 %col, {mask}
  %next = add i32 %left, {delta}
  %p0 = or i32 %left, %right
  %p1 = or i32 %next, %right
"""
    if masked:
        ir += "  %s0 = select i1 %valid, i32 %p0, i32 -1\n  %s1 = select i1 %valid, i32 %p1, i32 -1\n"
    ir += store("r", "s0" if masked else "p0")
    if separate_blocks:
        ir += "  br label %next_block\nnext_block:\n"
    ir += store("r2" if distinct else "r", "s1" if masked else "p1")
    return ir + f"""  ret void
}}
declare void @{intrinsic}({payload}, ptr addrspace(8), i32, i32, i32 immarg)
"""


@pytest.mark.parametrize("masked", [False, True])
def test_factors_disjoint_shared_expressions(masked):
    transformed, count = memory_rules.factor_offsets(_factor_buffer_store_offsets_fixture(masked=masked), "gfx936")
    assert count == 2
    stores = [
        line.strip()
        for line in transformed.splitlines()
        if f"call void @{_factor_buffer_store_offsets_INTRINSIC}" in line
    ]
    assert len(stores) == 2
    assert all("i32 %value, ptr addrspace(8) %r" in line and line.endswith("i32 0, i32 0)") for line in stores)
    assert transformed.count("select i1 %valid") == (2 if masked else 0)
    assert transformed.count("i32 -1") == (2 if masked else 0)
    assert "or i32" not in transformed
    assert transformed.count(", 12288") == 1


@pytest.mark.parametrize(
    "arch, kwargs",
    [
        ("gfx936", {"mask": 4095}),
        ("gfx936", {"delta": 12289}),
        ("gfx936", {"distinct": True}),
        ("gfx936", {"separate_blocks": True}),
        ("gfx942", {}),
    ],
)
def test_keeps_unproved_or_unsupported_stores(arch, kwargs):
    source = _factor_buffer_store_offsets_fixture(**kwargs)
    assert memory_rules.factor_offsets(source, arch) == (source, 0)


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("payload,suffix", [
    ("i16", "i16"),
    ("half", "f16"),
    ("float", "f32"),
    ("<2 x i32>", "v2i32"),
    ("<4 x i32>", "v4i32"),
    ("<2 x float>", "v2f32"),
])
def test_address_factoring_does_not_depend_on_payload_type(masked, payload, suffix):
    source = _factor_buffer_store_offsets_fixture(masked=masked, payload=payload, suffix=suffix)
    transformed, count = memory_rules.factor_offsets(source, "gfx936")
    assert count == 2
    # The same modulo-i32 offset equality applies to every payload overload;
    # neither data, descriptor, mask, cache policy nor store order changes.
    intrinsic = "llvm.amdgcn.raw.ptr.buffer.store." + suffix
    stores = [line for line in transformed.splitlines() if f"call void @{intrinsic}" in line]
    assert len(stores) == 2 and all(f"{payload} %value" in line for line in stores)
    assert transformed.count("select i1 %valid") == (2 if masked else 0)
    assert memory_rules.factor_offsets(transformed, "gfx936") == (transformed, 0)


@pytest.mark.parametrize("soffset", [0, 4096, -16])
@pytest.mark.parametrize("cache", [0, 1, 2, 3])
def test_preserves_other_address_and_cache_fields(soffset, cache):
    transformed, count = memory_rules.factor_offsets(_factor_buffer_store_offsets_fixture(soffset=soffset, cache=cache),
                                                     "gfx936")
    assert count == 2
    stores = [
        line for line in transformed.splitlines() if f"call void @{_factor_buffer_store_offsets_INTRINSIC}" in line
    ]
    assert len(stores) == 2
    assert all(line.endswith(f"i32 {soffset}, i32 {cache})") for line in stores)


# uniform resource indices
from importlib import import_module
if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU backend is not built", allow_module_level=True)
_uniform_resource_indices_HIPBackend = import_module("triton.backends.hcu.compiler").HIPBackend


def _uniform_resource_indices_lower(tmp_path, mode='uniform', bits=64, arch='gfx936', kernel=True):
    input_value = 'workgroup'
    count = '    %count = llvm.mlir.constant(1024 : i64) : i64'
    prepare = '    %wide = llvm.zext %id : i32 to i64'
    index = '%wide'
    if mode == 'varying': input_value = 'workitem'
    if bits == 32:
        prepare = ''
        index = '%id'
    elif bits < 32:
        prepare = f'    %narrow = llvm.trunc %id : i32 to i{bits}'
        index = '%narrow'
    elif bits > 64:
        prepare = f'    %wide = llvm.zext %id : i32 to i{bits}'
    if mode == 'undef':
        prepare = f'    %wide = llvm.mlir.undef : i{bits}'
        index = '%wide'
    if mode == 'freeze':
        prepare = f'    %unknown = llvm.mlir.undef : i{bits}\n    %wide = llvm.freeze %unknown : i{bits}'
        index = '%wide'
    if mode == 'arg':
        prepare = ''
        index = '%arg'
        bits = 64
    if mode == 'wide_count': count = '    %count = llvm.mlir.constant(4294967296 : i64) : i64'
    if mode == 'varying_count':
        count = '    %count32 = llvm.call @llvm.amdgcn.workitem.id.x() : () -> i32\n    %count = llvm.zext %count32 : i32 to i64'
    if mode == 'arg_count': count = '    %count = llvm.and %arg, %mask : i64'
    if mode == 'constant':
        prepare = f'    %wide = llvm.mlir.constant(8 : i{bits}) : i{bits}'
        index = '%wide'
    text = f'''module {{
  llvm.func @llvm.amdgcn.workgroup.id.x() -> i32
  llvm.func @llvm.amdgcn.workitem.id.x() -> i32
  llvm.func @llvm.amdgcn.make.buffer.rsrc.p8.p1(!llvm.ptr<1>, i16, i64, i32) -> !llvm.ptr<8>
  llvm.func @fixture(%base: !llvm.ptr<1>, %arg: i64) {{
    %id = llvm.call @llvm.amdgcn.{input_value}.id.x() : () -> i32
    %stride = llvm.mlir.constant(16448 : i16) : i16
    %flags = llvm.mlir.constant(131072 : i32) : i32
    %mask = llvm.mlir.constant(4294967295 : i64) : i64
{prepare}
{count}
    %address = llvm.getelementptr %base[{index}] : (!llvm.ptr<1>, i{bits}) -> !llvm.ptr<1>, i8
    %resource = llvm.call @llvm.amdgcn.make.buffer.rsrc.p8.p1(%address, %stride, %count, %flags)
      : (!llvm.ptr<1>, i16, i64, i32) -> !llvm.ptr<8>
    llvm.return
  }}
}}'''
    path = tmp_path / 'resource.mlir'
    path.write_text(text)
    ctx = libtriton.ir.context()
    libtriton.ir.load_dialects(ctx)
    _uniform_resource_indices_HIPBackend(GPUTarget('hip', 'gfx936', 64)).load_dialects(ctx)
    mod = libtriton.ir.parse_mlir_module(str(path), ctx)
    libtriton.llvm.init_targets()
    llctx = libtriton.llvm.context()
    ll = libtriton.llvm.to_module(mod, llctx)
    libtriton.hcu.attach_target_triple(ll)
    libtriton.llvm.attach_datalayout(ll, libtriton.hcu.TARGET_TRIPLE, 'gfx936', '')
    for function in ll.get_functions():
        if function.name == 'fixture' and kernel: function.set_calling_conv(libtriton.hcu.CALLING_CONV_AMDGPU_KERNEL)
    before = str(ll)
    changed = libtriton.hcu.preserve_uniform_resource_indices(ll, arch)
    after = str(ll)
    again = libtriton.hcu.preserve_uniform_resource_indices(ll, arch)
    assert again == 0
    return changed, before, after


@pytest.mark.parametrize('bits', [1, 16, 32, 64])
def test_preserves_defined_scalar_indices_and_is_idempotent(tmp_path, bits):
    changed, before, after = _uniform_resource_indices_lower(tmp_path, bits=bits)
    assert changed == 1
    assert 'getelementptr i8' in after and 'llvm.amdgcn.readfirstlane' in after
    assert 'hcu.uniform.resource.indices.done' in after
    assert 'ptrtoint' not in after and 'inttoptr' not in after


@pytest.mark.parametrize('mode',
                         ['varying', 'undef', 'freeze', 'arg', 'wide_count', 'varying_count', 'arg_count', 'constant'])
def test_unproved_or_unnecessary_stays_unchanged(tmp_path, mode):
    changed, before, after = _uniform_resource_indices_lower(tmp_path, mode)
    assert changed == 0 and before == after


@pytest.mark.parametrize('arch,kernel', [('gfx942', True), ('gfx936', False)])
def test_requires_the_proved_target_kernel_domain(tmp_path, arch, kernel):
    changed, before, after = _uniform_resource_indices_lower(tmp_path, arch=arch, kernel=kernel)
    assert changed == 0 and before == after


# buffer load byte offsets

from ._helpers import load_fixture
from triton._C.libtriton import hcu

_buffer_load_byte_offsets_BASE = '''
declare i32 @input()
declare i1 @input_pred()
declare i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8), i32, i32, i32 immarg)
declare void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8), ptr addrspace(3), i32 immarg, i32, i32, i32 immarg, i32 immarg)
define void @fixture() {
entry:
  %base = call i32 @input()
  %delta = call i32 @input()
  %pred = call i1 @input_pred()
  %off = add i32 %base, %delta
  %bytes = shl i32 %off, 1
  %masked = select i1 %pred, i32 %bytes, i32 -2147483648
  %value = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) poison, i32 %masked, i32 0, i32 0)
  store i32 %value, ptr addrspace(1) poison
  ret void
}
'''


@pytest.mark.parametrize('case', ['masked_true', 'masked_false', 'unmasked', 'shared_scale'])
def test_preserve_scale_and_idempotence(tmp_path, case):
    text = _buffer_load_byte_offsets_BASE
    if case == 'masked_false':
        text = text.replace('i32 %bytes, i32 -2147483648', 'i32 -2147483648, i32 %bytes')
    elif case == 'unmasked':
        text = text.replace('i32 %masked, i32 0, i32 0)', 'i32 %bytes, i32 0, i32 0)')
    elif case == 'shared_scale':
        text = text.replace(
            '  store i32 %value,',
            '  %other = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) poison, i32 %bytes, i32 0, i32 0)\n  store i32 %value,'
        )
    module, keep = load_fixture(tmp_path, text)
    before = str(module)
    assert hcu.preserve_buffer_load_byte_offsets(module, 'gfx936') == 1
    after = str(module)
    assert len(re.findall(r'freeze i32', after)) == 1
    assert re.search(r'(%[^ ]+) = freeze i32 %off\n  %bytes = shl i32 \1, 1', after)
    assert before.count('call i32 @llvm.amdgcn.raw.ptr.buffer.load') == after.count(
        'call i32 @llvm.amdgcn.raw.ptr.buffer.load')
    assert hcu.preserve_buffer_load_byte_offsets(module, 'gfx936') == 0
    assert str(module) == after


@pytest.mark.parametrize(
    'case',
    ['other_arch', 'other_scale', 'add_offset', 'already_frozen', 'non_kernel', 'to_lds', 'user_call', 'buffer_store'])
def test_unrelated_operations_unchanged(tmp_path, case):
    text = _buffer_load_byte_offsets_BASE
    if case == 'other_scale':
        text = text.replace('%bytes = shl i32 %off, 1', '%bytes = shl i32 %off, 2')
    elif case == 'add_offset':
        text = text.replace('%bytes = shl i32 %off, 1', '%bytes = add i32 %off, 1')
    elif case == 'already_frozen':
        text = text.replace('  %bytes = shl i32 %off, 1', '  %ready = freeze i32 %off\n  %bytes = shl i32 %ready, 1')
    elif case == 'to_lds':
        text = text.replace(
            '  %value = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) poison, i32 %masked, i32 0, i32 0)\n  store i32 %value, ptr addrspace(1) poison',
            '  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) poison, ptr addrspace(3) poison, i32 4, i32 %masked, i32 0, i32 0, i32 0)'
        )
    elif case == 'user_call':
        text = text.replace('@llvm.amdgcn.raw.ptr.buffer.load.i32', '@user_load')
    elif case == 'buffer_store':
        text = text.replace(
            'define void @fixture()',
            'declare void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32, ptr addrspace(8), i32, i32, i32 immarg)\ndefine void @fixture()'
        )
        text = text.replace(
            '  %value = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) poison, i32 %masked, i32 0, i32 0)\n  store i32 %value, ptr addrspace(1) poison',
            '  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 1, ptr addrspace(8) poison, i32 %masked, i32 0, i32 0)'
        )
    module, keep = load_fixture(tmp_path, text)
    if case == 'non_kernel':
        next(fn for fn in module.get_functions() if fn.name == 'fixture').set_calling_conv(0)
    before = str(module)
    assert hcu.preserve_buffer_load_byte_offsets(module, 'gfx928' if case == 'other_arch' else 'gfx936') == 0
    assert str(module) == before


# kernelarg address remat
from triton._C.libtriton import ir, passes
try:
    from triton._C.libtriton import hcu
except ImportError:
    pytest.skip("Requires the HCU backend", allow_module_level=True)

_kernelarg_address_remat_BASE = r"""#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [1, 0]}>
#sM = #ttg.slice<{dim = 1, parent = #blocked}>
#sK = #ttg.slice<{dim = 0, parent = #blocked}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx936", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @outer_factor_inner_unknown_stride(%A: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %O: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %U: i32 {tt.divisibility = 8 : i32}, %stride: i32 {tt.divisibility = 16 : i32}) {
    %m = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #sM>
    %k = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #sK>
    %m1 = tt.expand_dims %m {axis = 1 : i32} : tensor<256xi32, #sM> -> tensor<256x1xi32, #blocked>
    %k1 = tt.expand_dims %k {axis = 0 : i32} : tensor<16xi32, #sK> -> tensor<1x16xi32, #blocked>
    %mf = tt.broadcast %m1 : tensor<256x1xi32, #blocked> -> tensor<256x16xi32, #blocked>
    %kf = tt.broadcast %k1 : tensor<1x16xi32, #blocked> -> tensor<256x16xi32, #blocked>
    %pitch = arith.constant dense<1024> : tensor<256x16xi32, #blocked>
    %sixteen = arith.constant dense<16> : tensor<256x16xi32, #blocked>
    %eight = arith.constant dense<8> : tensor<256x16xi32, #blocked>
    %zero = arith.constant dense<0> : tensor<256x16xi32, #blocked>
    %stridef = tt.splat %stride : i32 -> tensor<256x16xi32, #blocked>
    %row = arith.muli %mf, %stridef : tensor<256x16xi32, #blocked>
    %v0 = arith.addi %row, %kf : tensor<256x16xi32, #blocked>
    %us = tt.splat %U : i32 -> tensor<256x16xi32, #blocked>
    %sum = arith.addi %us, %v0 : tensor<256x16xi32, #blocked>
    %kb = arith.andi %kf, %eight : tensor<256x16xi32, #blocked>
    %mask = arith.cmpi eq, %kb, %zero : tensor<256x16xi32, #blocked>
    %init = arith.constant dense<-3.000000e+00> : tensor<256x16xbf16, #blocked>
    %dst = ttg.local_alloc %init : (tensor<256x16xbf16, #blocked>) -> !ttg.memdesc<256x16xbf16, #shared, #smem, mutable>
    ttg.local_barrier
    %t = hcug.buffer_load_to_local %A[%sum] into %dst {contiguity = 8 : i32} : <bf16>[tensor<256x16xi32, #blocked>] -> <256x16xbf16, #shared, #smem, mutable>
    %w = ttg.async_wait %t {num = 0 : i32}
    ttg.local_barrier
    %got = ttg.local_load %dst token %w : !ttg.memdesc<256x16xbf16, #shared, #smem, mutable> -> tensor<256x16xbf16, #blocked>
    %orow = arith.muli %mf, %sixteen : tensor<256x16xi32, #blocked>
    %oi = arith.addi %orow, %kf : tensor<256x16xi32, #blocked>
    %op = tt.splat %O : !tt.ptr<bf16> -> tensor<256x16x!tt.ptr<bf16>, #blocked>
    %optrs = tt.addptr %op, %oi : tensor<256x16x!tt.ptr<bf16>, #blocked>, tensor<256x16xi32, #blocked>
    tt.store %optrs, %got : tensor<256x16x!tt.ptr<bf16>, #blocked>
    ttg.local_dealloc %dst : !ttg.memdesc<256x16xbf16, #shared, #smem, mutable>
    tt.return
  }
}
"""


def _kernelarg_address_remat_budget_fixture():
    # Keep the existing 64-value rematerialization budget effective even when
    # the address also contains an allowed kernel argument. All row values
    # stay in [0,255]; the long nonlinear DAG is deliberately not cheap.
    ty = "tensor<256xi32, #sM>"
    lines = [
        "    %budget_five = arith.constant dense<5> : " + ty,
        "    %budget_xor = arith.constant dense<91> : " + ty,
        "    %budget_mask = arith.constant dense<255> : " + ty,
    ]
    previous = "%m"
    for i in range(40):
        mul, xor, bounded = ("%budget_" + name + str(i) for name in ["mul", "xor", "and"])
        lines.extend([
            "    " + mul + " = arith.muli " + previous + ", %budget_five : " + ty,
            "    " + xor + " = arith.xori " + mul + ", %budget_xor : " + ty,
            "    " + bounded + " = arith.andi " + xor + ", %budget_mask : " + ty,
        ])
        previous = bounded
    return _kernelarg_address_remat_BASE.replace("    %m1 = tt.expand_dims %m {",
                                                 "\n".join(lines) + "\n    %m1 = tt.expand_dims " + previous + " {")


def _kernelarg_address_remat_fixture(kind):
    text = _kernelarg_address_remat_BASE
    if kind == 'budget': return _kernelarg_address_remat_budget_fixture()
    if kind == 'original_kernel':
        return text
    # Mixed LLVM/TTGIR avoids FuncOpConversion rewriting the marker we test.
    # Pointer casts keep the original typed BUFFER op, while function arguments
    # have legal LLVM ABI types. No inline assembly or hardware flags involved.
    public = kind != 'private_marked'
    marker = {'unmarked': '', 'marker_zero': ', nvvm.kernel = 0 : i1'}.get(kind, ', nvvm.kernel = 1 : i1')
    extra = '' if public else ', %shared: !llvm.ptr<3>, %scratch: !llvm.ptr<1>, %profile: !llvm.ptr<1>'
    header = ('llvm.func @fixture(%Araw: !llvm.ptr<1> {tt.divisibility = 16 : i32}, '
              '%Oraw: !llvm.ptr<1> {tt.divisibility = 16 : i32}, '
              '%U: i32 {tt.divisibility = 8 : i32}, %stride: i32 {tt.divisibility = 16 : i32}' + extra +
              ') attributes {sym_visibility = "' + ('public' if public else 'private') + '"' + marker + '} {')
    line = next(l for l in text.splitlines() if 'tt.func public' in l)
    casts = '''
    %A = builtin.unrealized_conversion_cast %Araw : !llvm.ptr<1> to !tt.ptr<bf16>
    %O = builtin.unrealized_conversion_cast %Oraw : !llvm.ptr<1> to !tt.ptr<bf16>'''
    text = text.replace(line, '  ' + header + casts).replace('    tt.return', '    llvm.return')
    if kind in ['marked_i64', 'marked_i16']:
        width = 64 if kind == 'marked_i64' else 16
        text = text.replace('%stride: i32', '%stride: i' + str(width))
        op = 'trunc' if width == 64 else 'zext'
        text = text.replace(casts, casts + '\n    %stride32 = llvm.' + op + ' %stride : i' + str(width) + ' to i32')
        text = text.replace('tt.splat %stride :', 'tt.splat %stride32 :')
    if kind == 'nonentry_block':
        text = text.replace(casts, '''
    cf.br ^next(%stride : i32)
  ^next(%carried: i32):''' + casts)
        text = text.replace('tt.splat %stride :', 'tt.splat %carried :')
    if kind == 'scalar_load':
        text = text.replace(casts, casts + '\n    %loaded = llvm.load %Oraw : !llvm.ptr<1> -> i32')
        text = text.replace('tt.splat %stride :', 'tt.splat %loaded :')
    if kind == 'scalar_call':
        text = text.replace(casts, casts + '\n    %called = llvm.call @opaque() : () -> i32')
        text = text.replace('tt.splat %stride :', 'tt.splat %called :')
        text = text.replace('  llvm.func @fixture', '  llvm.func @opaque() -> i32\n  llvm.func @fixture')
    return text


def _kernelarg_address_remat_lower(tmp_path, text):
    src = tmp_path / 'fixture.mlir'
    src.write_text(text)
    ctx = ir.context()
    ir.load_dialects(ctx)
    hcu.load_dialects(ctx)
    mod = ir.parse_mlir_module(str(src), ctx)
    pm = ir.pass_manager(ctx)
    hcu.passes.ttgpuir.add_update_async_wait_count(pm, 'gfx936')
    passes.convert.add_scf_to_cf(pm)
    hcu.passes.ttgpuir.add_allocate_shared_memory(pm)
    hcu.passes.ttgpuir.add_to_llvmir(pm, 'gfx936', True)
    passes.convert.add_cf_to_llvmir(pm)
    passes.convert.add_arith_to_llvmir(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    pm.run(mod, 'kernelarg_remat_fixture')
    out = str(mod)
    (tmp_path / 'lowered.mlir').write_text(out)
    # Count actual instructions, not declarations of the intrinsic.
    count = sum('bpermute' in line and ('llvm.call' in line or 'rocdl.ds' in line) for line in out.splitlines())
    assert 'raw.ptr.buffer.load.lds' in out or 'raw_ptr_buffer_load_lds' in out
    return count


@pytest.mark.parametrize('kind,expected', [
    ('original_kernel', 0),
    ('marked', 0),
    ('unmarked', 1),
    ('marker_zero', 1),
    ('private_marked', 1),
    ('nonentry_block', 1),
    ('scalar_load', 1),
    ('scalar_call', 1),
    ('marked_i64', 0),
    ('marked_i16', 0),
    ('budget', 1),
])
def test_kernel_entry_guard(tmp_path, kind, expected):
    assert _kernelarg_address_remat_lower(tmp_path, _kernelarg_address_remat_fixture(kind)) == expected


# buffer resource native abi
from triton.backends.hcu import compiler_hcu
from triton.runtime.errors import HSACOError

_buffer_resource_native_abi_INTRINSIC = "@llvm.amdgcn.make.buffer.rsrc.p8.p1"


def _buffer_resource_native_abi_resource_module(count):
    return f"""
declare ptr addrspace(8) {_buffer_resource_native_abi_INTRINSIC}(ptr addrspace(1), i16, i64, i32)

define ptr addrspace(8) @resource(ptr addrspace(1) %base, i32 %count) {{
  %wide = zext i32 %count to i64
  %resource = call ptr addrspace(8) {_buffer_resource_native_abi_INTRINSIC}(ptr addrspace(1) %base, i16 0, i64 {count}, i32 0)
  ret ptr addrspace(8) %resource
}}
"""


@pytest.mark.parametrize("count", ["%wide", "4294967295"])
def test_clang18_buffer_resource_count_is_narrowed_when_proved(count):
    source = _buffer_resource_native_abi_resource_module(count)
    converted = llvm.adapt_hcu_buffer_resources_for_clang18(source)
    declaration = next(line for line in converted.splitlines()
                       if line.startswith("declare") and _buffer_resource_native_abi_INTRINSIC in line)
    assert ", i16, i32, i32)" in declaration
    call = next(line for line in converted.splitlines()
                if f"call ptr addrspace(8) {_buffer_resource_native_abi_INTRINSIC}" in line)
    assert ", i64 " not in call


def test_clang18_buffer_resource_count_cannot_drop_nonzero_bits():
    with pytest.raises(ValueError, match="32-bit buffer resource count"):
        llvm.adapt_hcu_buffer_resources_for_clang18(_buffer_resource_native_abi_resource_module("4294967296"))


def test_clang18_buffer_resource_rejection_is_backend_compile_failure(monkeypatch):
    monkeypatch.setattr(compiler_hcu, "_hcu_clang_version", lambda: "clang version 18.0.0")
    backend = compiler_hcu.HIPBackend(GPUTarget("hip", "gfx936", 64))
    source = _buffer_resource_native_abi_resource_module("4294967296") + """
define amdgpu_kernel void @kernel() {
  ret void
}
"""
    with pytest.raises(HSACOError, match="32-bit buffer resource count"):
        backend.make_amdgcn(source, {}, backend.parse_options({}), _skip_schedules=True)


# range analysis

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler bindings are required", allow_module_level=True)

_range_analysis_compiler = pytest.importorskip("triton.backends.hcu.compiler")


@pytest.mark.parametrize("lower,other,upper,step,limit", [
    (0, 8, 16, 2, 4),
    (1, 10, 19, 3, 3),
    (-8, 0, 8, 2, 4),
])
def test_dynamic_lower_bound_keeps_variable_trip_count(tmp_path, lower, other, upper, step, limit):
    # One input runs more iterations than limit; the other does not. Using
    # the maximum lower bound to bound tripcount incorrectly proves true.
    assert (upper - lower + step - 1) // step > limit
    assert (upper - other + step - 1) // step <= limit
    source = tmp_path / "dynamic_lower.mlir"
    source.write_text(f'''module attributes {{"ttg.num-warps" = 4 : i32}} {{
  tt.func @dynamic_lower(%choose_lower: i1) -> i1 {{
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %low = arith.constant {lower} : i32
    %other = arith.constant {other} : i32
    %upper = arith.constant {upper} : i32
    %step = arith.constant {step} : i32
    %limit = arith.constant {limit} : i32
    %begin = arith.select %choose_lower, %low, %other : i32
    %count = scf.for %iv = %begin to %upper step %step iter_args(%acc = %zero) -> (i32) : i32 {{
      %next = arith.addi %acc, %one : i32
      scf.yield %next : i32
    }}
    %result = arith.cmpi sle, %count, %limit : i32
    tt.return %result : i1
  }}
}}
''')
    ir, passes, hcu = libtriton.ir, libtriton.passes, libtriton.hcu
    context = ir.context()
    ir.load_dialects(context)
    _range_analysis_compiler.HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    module = ir.parse_mlir_module(str(source), context)
    pm = ir.pass_manager(context)
    hcu.passes.ttgpuir.add_fold_true_cmpi(pm)
    passes.common.add_canonicalizer(pm)
    pm.run(module, "test_dynamic_lower_range")
    result = str(module)
    assert "scf.for" in result
    assert "arith.cmpi sle" in result


# structured register vectors

from triton.backends.hcu.compiler import HIPBackend


def _structured_register_vectors_source(dtype, layout="permuted", alignment=16, column=0):
    registers = "[[0, 32], [0, 1], [0, 2], [0, 4], [64, 0]]"
    lanes = "[[1, 0], [2, 0], [4, 0], [8, 0], [0, 8], [0, 16]]"
    if layout == "ordered":
        registers = "[[0, 1], [0, 2], [0, 4], [0, 32], [64, 0]]"
    elif layout == "lane_split":
        registers = "[[0, 32], [0, 8], [0, 2], [0, 4], [64, 0]]"
        lanes = "[[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 16]]"
    return f'''#layout = #ttg.linear<{{register = {registers}, lane = {lanes}, warp = [[16, 0], [32, 0]], block = []}}>
module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"}} {{
  tt.func public @vectors(%p: !tt.ptr<{dtype}> {{tt.divisibility = {alignment} : i32}},
                           %q: !tt.ptr<{dtype}>, %M: i32, %row: i32) {{
    %zero = arith.constant 0 : i32
    %column = arith.constant {column} : i32
    %one = arith.constant 1 : i32
    %K = arith.constant 64 : i32
    %stride = arith.constant 64 : i64
    %unit = arith.constant 1 : i64
    %tile = tt.matrix_load %p, shape = [%M, %K], strides = [%stride, %unit],
      tensorShape = [128, 64], indices = [%row, %column, %one]
      {{hcu.structured_register, hcu.structured_register.internal_width = 8 : i32,
        hcu.structured_register.internal_order = array<i32: 999>}}
      : <{dtype}> -> tensor<128x64x{dtype}, #layout>
    %m = tt.make_range {{start = 0 : i32, end = 128 : i32}}
      : tensor<128xi32, #ttg.slice<{{dim = 1, parent = #layout}}>>
    %n = tt.make_range {{start = 0 : i32, end = 64 : i32}}
      : tensor<64xi32, #ttg.slice<{{dim = 0, parent = #layout}}>>
    %ms = tt.expand_dims %m {{axis = 1 : i32}}
      : tensor<128xi32, #ttg.slice<{{dim = 1, parent = #layout}}>> -> tensor<128x1xi32, #layout>
    %ns = tt.expand_dims %n {{axis = 0 : i32}}
      : tensor<64xi32, #ttg.slice<{{dim = 0, parent = #layout}}>> -> tensor<1x64xi32, #layout>
    %mb = tt.broadcast %ms : tensor<128x1xi32, #layout> -> tensor<128x64xi32, #layout>
    %nb = tt.broadcast %ns : tensor<1x64xi32, #layout> -> tensor<128x64xi32, #layout>
    %stride_tile = arith.constant dense<64> : tensor<128x64xi32, #layout>
    %row_offset = arith.muli %mb, %stride_tile : tensor<128x64xi32, #layout>
    %offset = arith.addi %row_offset, %nb : tensor<128x64xi32, #layout>
    %base = tt.splat %q : !tt.ptr<{dtype}> -> tensor<128x64x!tt.ptr<{dtype}>, #layout>
    %ptrs = tt.addptr %base, %offset : tensor<128x64x!tt.ptr<{dtype}>, #layout>, tensor<128x64xi32, #layout>
    tt.store %ptrs, %tile : tensor<128x64x!tt.ptr<{dtype}>, #layout>
    tt.return
  }}
}}'''


def _structured_register_vectors_lower(tmp_path, dtype, layout, alignment, column, private=False):
    path = tmp_path / "vectors.ttgir"
    text = _structured_register_vectors_source(dtype, layout, alignment, column)
    if private:
        text = text.replace("tt.func public @vectors", "tt.func private @vectors")
    path.write_text(text)
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    module = libtriton.ir.parse_mlir_module(str(path), context)
    manager = libtriton.ir.pass_manager(context)
    libtriton.hcu.passes.ttgpuir.add_to_llvmir(manager, "gfx936", False)
    manager.run(module, "structured_register_vectors")
    result = str(module)
    (tmp_path / "vectors.llvm.mlir").write_text(result)
    return result


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize("layout,alignment,column,width,calls", [
    ("ordered", 16, 0, "v4i32", 4),
    ("permuted", 16, 0, "v4i32", 4),
    ("permuted", 8, 0, "v2i32", 8),
    ("permuted", 2, 0, "i16", 32),
    ("permuted", 16, 1, "i16", 32),
    ("lane_split", 16, 0, "i16", 32),
])
def test_address_grouping_respects_thread_alignment_and_column(tmp_path, dtype, layout, alignment, column, width,
                                                               calls):
    result = _structured_register_vectors_lower(tmp_path, dtype, layout, alignment, column)
    name = "llvm.amdgcn.struct.ptr.buffer.load." + width
    assert len(re.findall(r"llvm.call @" + re.escape(name) + r"\(", result)) == calls
    assert "bpermute" not in result and "llvm.alloca" not in result


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_private_pointer_argument_requires_uniform_base_proof(tmp_path, capfd, dtype):
    # A private function's scalar-typed pointer may vary by lane. Its alignment
    # attribute cannot establish CTA uniformity for a scalar buffer descriptor.
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _structured_register_vectors_lower(tmp_path, dtype, "ordered", 16, 0, private=True)
    assert "structured-register: requires uniform AS1 base" in capfd.readouterr().err


@pytest.mark.parametrize("masked", [False, True])
def test_combined_memory_rules_preserve_sequential_store_result(masked):
    source = _factor_buffer_store_offsets_fixture(masked=masked)
    sequential, count = memory_rules.factor_offsets(source, "gfx936")
    sequential, _ = memory_rules.pack_rounds(sequential)
    sequential, _, _ = memory_rules.combine_shifts(sequential, "gfx936")
    combined, stats = llvm.optimize_hcu_memory_ir(source, "gfx936", True, True, True)
    assert combined == sequential
    assert stats["factored_stores"] == count == 2
    assert llvm.optimize_hcu_memory_ir(source, "gfx936", False, False, False)[0] == source

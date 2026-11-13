"""Exercise the real lowering through marked and unmarked entry parameters."""
import pytest
from triton._C.libtriton import ir, passes
try:
    from triton._C.libtriton import hcu
except ImportError:
    pytest.skip("Requires the HCU backend", allow_module_level=True)

BASE = r"""#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [8, 1], order = [1, 0]}>
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

def budget_fixture():
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
    return BASE.replace("    %m1 = tt.expand_dims %m {",
                        "\n".join(lines) + "\n    %m1 = tt.expand_dims " + previous + " {")

def fixture(kind):
    text = BASE
    if kind == 'budget': return budget_fixture()
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
              '%U: i32 {tt.divisibility = 8 : i32}, %stride: i32 {tt.divisibility = 16 : i32}'
              + extra + ') attributes {sym_visibility = "' + ('public' if public else 'private') + '"' + marker + '} {')
    line = next(l for l in text.splitlines() if 'tt.func public' in l)
    casts = '''
    %A = builtin.unrealized_conversion_cast %Araw : !llvm.ptr<1> to !tt.ptr<bf16>
    %O = builtin.unrealized_conversion_cast %Oraw : !llvm.ptr<1> to !tt.ptr<bf16>'''
    text = text.replace(line, '  ' + header + casts).replace('    tt.return', '    llvm.return')
    if kind in ['marked_i64', 'unsupported_i16']:
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

def lower(tmp_path, text):
    src = tmp_path / 'fixture.mlir'; src.write_text(text)
    ctx = ir.context(); ir.load_dialects(ctx); hcu.load_dialects(ctx)
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
    out = str(mod); (tmp_path / 'lowered.mlir').write_text(out)
    # Count actual instructions, not declarations of the intrinsic.
    count = sum('bpermute' in line and ('llvm.call' in line or 'rocdl.ds' in line)
                for line in out.splitlines())
    assert 'raw.ptr.buffer.load.lds' in out or 'raw_ptr_buffer_load_lds' in out
    return count

@pytest.mark.parametrize('kind,expected', [
    ('original_kernel', 0), ('marked', 0), ('unmarked', 1),
    ('marker_zero', 1), ('private_marked', 1), ('nonentry_block', 1),
    ('scalar_load', 1), ('scalar_call', 1), ('marked_i64', 0),
    ('unsupported_i16', 1), ('budget', 1),
])
def test_kernel_entry_guard(tmp_path, kind, expected):
    assert lower(tmp_path, fixture(kind)) == expected

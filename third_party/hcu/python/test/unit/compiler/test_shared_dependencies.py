# physical shared regions
"""SharedLinear byte ranges must remove only proved-disjoint membar edges."""
import pytest

from triton._C import libtriton
from triton.backends.compiler import GPUTarget

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler bindings are required", allow_module_level=True)
_physical_shared_regions_compiler = pytest.importorskip("triton.backends.hcu.compiler")


def _physical_shared_regions_module_source(second_offset=64, index="bounded", select_load=False):
    index_op = {
        "bounded": "%index = arith.andi %idx, %one : i32", "unknown": "%index = arith.addi %idx, %zero : i32",
        "constant": "%index = arith.constant 0 : i32"
    }[index]
    types = "tensor<8x8xbf16, #blocked>"
    md = "!ttg.memdesc<8x8xbf16, #xor, #smem, mutable>"
    # A legal smaller tile partially overlaps A at [32,64); a 64-element
    # tile at offset32 is rejected by the shared-view verifier itself.
    second_size = 32 if second_offset == 32 else 64
    types_b = "tensor<4x8xbf16, #blocked>" if second_size == 32 else types
    md_b = "!ttg.memdesc<4x8xbf16, #xor_b, #smem, mutable>" if second_size == 32 else md
    if select_load:
        operations = f'''ttg.local_store %ones, %a : {types} -> {md}
    ttg.local_barrier
    %choice = arith.select %cond, %a, %b : {md}
    %value = ttg.local_load %choice : {md} -> {types}
    ttg.local_store %ones_b, %b : {types_b} -> {md_b}'''
    else:
        operations = f'''ttg.local_store %ones, %a : {types} -> {md}
    ttg.local_store %ones_b, %b : {types_b} -> {md_b}
    %value = ttg.local_load %a : {md} -> {types}'''
    return f'''#flat = #ttg.swizzled_shared<{{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}}>
#xor = #ttg.shared_linear<{{offset = [[0, 1], [0, 2], [0, 4], [1, 1], [2, 2], [4, 4]]}}, alignment = 16>
#xor_b = #ttg.shared_linear<{{offset = [[0, 1], [0, 2], [0, 4], [1, 1], [2, 2]]}}, alignment = 16>
#blocked = #ttg.blocked<{{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}}>
#smem = #ttg.shared_memory
module attributes {{"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"}} {{
  tt.func public @regions(%idx: i32, %cond: i1) {{
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    {index_op}
    %ones = arith.constant dense<1.0> : {types}
    %ones_b = arith.constant dense<2.0> : {types_b}
    %root = ttg.local_alloc {{alignment = 16 : i32}} : () -> !ttg.memdesc<2x128xbf16, #flat, #smem, mutable>
    %frame = ttg.memdesc_index %root[%index] : !ttg.memdesc<2x128xbf16, #flat, #smem, mutable> -> !ttg.memdesc<128xbf16, #flat, #smem, mutable>
    %sa = ttg.memdesc_subslice %frame[0] : !ttg.memdesc<128xbf16, #flat, #smem, mutable> -> !ttg.memdesc<64xbf16, #flat, #smem, mutable, 128>
    %sb = ttg.memdesc_subslice %frame[{second_offset}] : !ttg.memdesc<128xbf16, #flat, #smem, mutable> -> !ttg.memdesc<{second_size}xbf16, #flat, #smem, mutable, 128>
    %a = ttg.memdesc_reinterpret %sa : !ttg.memdesc<64xbf16, #flat, #smem, mutable, 128> -> {md}
    %b = ttg.memdesc_reinterpret %sb : !ttg.memdesc<{second_size}xbf16, #flat, #smem, mutable, 128> -> {md_b}
    {operations}
    ttg.local_dealloc %root : !ttg.memdesc<2x128xbf16, #flat, #smem, mutable>
    tt.return
  }}
}}
'''


@pytest.mark.parametrize("second_offset,index,select_load,barriers", [
    (64, "bounded", False, 1),
    (64, "constant", False, 1),
    (32, "bounded", False, 2),
    (0, "bounded", False, 2),
    (64, "unknown", False, 2),
    # A read that can select B must retain its WAR barrier before writing B.
    (64, "bounded", True, 2),
])
def test_shared_linear_disjointness(tmp_path, second_offset, index, select_load, barriers):
    result = _physical_shared_regions_lower_source(
        tmp_path, _physical_shared_regions_module_source(second_offset, index, select_load))
    assert result.count("rocdl.s.barrier") == barriers, result


def _physical_shared_regions_lower_source(tmp_path, source, update_async_wait=False):
    path = tmp_path / "regions.mlir"
    path.write_text(source)
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    _physical_shared_regions_compiler.HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    module = libtriton.ir.parse_mlir_module(str(path), context)
    pm = libtriton.ir.pass_manager(context)
    if update_async_wait:
        libtriton.hcu.passes.ttgpuir.add_update_async_wait_count(pm, "gfx936")
    libtriton.hcu.passes.ttgpuir.add_allocate_shared_memory(pm)
    libtriton.hcu.passes.ttgpuir.add_to_llvmir(pm, "gfx936", False)
    pm.run(module, "physical_shared_regions_contract")
    return str(module)


def test_same_block_conditional_forwarding(tmp_path):
    text = _physical_shared_regions_module_source(select_load=True)
    md = "!ttg.memdesc<8x8xbf16, #xor, #smem, mutable>"
    text = text.replace(f"%choice = arith.select %cond, %a, %b : {md}",
                        f"cf.cond_br %cond, ^merge(%a : {md}), ^merge(%b : {md})\n  ^merge(%choice: {md}):")
    # Both edges from the same predecessor must be included; B aliases the
    # second incoming view and still needs the read-before-write barrier.
    assert _physical_shared_regions_lower_source(tmp_path, text).count("rocdl.s.barrier") == 2


def test_unmodeled_multibuffer_reinterpret_falls_back(tmp_path):
    text = _physical_shared_regions_module_source(index="constant")
    old = "%frame = ttg.memdesc_index %root[%index] : !ttg.memdesc<2x128xbf16, #flat, #smem, mutable> -> !ttg.memdesc<128xbf16, #flat, #smem, mutable>"
    new = "%frame = ttg.memdesc_reinterpret %root : !ttg.memdesc<2x128xbf16, #flat, #smem, mutable> -> !ttg.memdesc<256xbf16, #flat, #smem, mutable>"
    assert old in text
    text = text.replace(old, new).replace("!ttg.memdesc<128xbf16, #flat",
                                          "!ttg.memdesc<256xbf16, #flat").replace("mutable, 128>", "mutable, 256>")
    # SharedLinear conversion helpers assert if handed a whole buffer stack
    # with a lower-rank encoding. This legal reinterpret must compile safely
    # and keep generic barriers until its physical chart is modeled.
    assert _physical_shared_regions_lower_source(tmp_path, text).count("rocdl.s.barrier") == 2


# deferred warp wait
import re

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler required", allow_module_level=True)

from triton.backends.hcu.compiler import HIPBackend


def _deferred_warp_wait_source(dtype="bf16", slots=2, trips=3, start=0, step=1, consumers=1, producer_default=False,
                               carried=False, negative=None, producer_wide=False, tail_store=False):
    root = f"!ttg.memdesc<{slots}x8x8x{dtype}, #shared, #smem, mutable>"
    plane = f"!ttg.memdesc<8x8x{dtype}, #plane, #smem, mutable>"
    half = f"!ttg.memdesc<4x8x{dtype}, #plane, #smem, mutable, 8x8>"
    tensor = f"tensor<8x8x{dtype}, #blocked>"
    fragment = f"tensor<4x8x{dtype}, #blocked>"
    pointers = f"tensor<8x8x!tt.ptr<{dtype}>, #blocked>"
    bar = "ttg.local_barrier {hcu.cta_scope}"

    def role(producer, ptr, owner, dynamic, terminator):
        origin = -3 if producer and producer_wide else start
        begin = f'''%zero = arith.constant 0 : i32
        %one = arith.constant 1 : i32
        %slots = arith.constant {slots} : i32
        %start = arith.constant {origin} : i32
        %stop = arith.constant {origin + trips * step} : i32
        %step = arith.constant {step} : i32
        %pointers = tt.splat {ptr} : !tt.ptr<{dtype}> -> {pointers}
        %zeros = arith.constant dense<0.0> : {tensor}
        %half_zeros = arith.constant dense<0.0> : {fragment}
        '''
        if not producer:
            begin += bar + "\n" + ("" if negative == "phase_count" else bar + "\n")
        limit = dynamic if negative == "dynamic_loop" else "%stop"
        header = f"scf.for %iv = %start to {limit} step %step : i32 {{"
        if carried:
            header = f"%final_slot = scf.for %iv = %start to {limit} step %step iter_args(%carry = %zero) -> (i32) : i32 {{"
        begin += header + '''
          %relative = arith.subi %iv, %start : i32
          %iteration = arith.divsi %relative, %step : i32
          %current = arith.remsi %iteration, %slots : i32
          %future_index = arith.addi %current, %one : i32
          %future = arith.remsi %future_index, %slots : i32
        '''
        slot = "%carry" if carried else "%current"
        if producer and negative == "unknown_slot":
            slot = dynamic
        if not producer and negative == "same_slot":
            slot = "%future"
        if producer and producer_wide:
            begin += f"%slot32 = arith.trunci {slot} : i128 to i32\n"
            slot = "%slot32"
        begin += f"%window = ttg.memdesc_index {owner}[{slot}] : {root} -> {plane}\n"
        if producer:
            begin += f'''%copy = ttg.async_copy_global_to_local %pointers, %window : {pointers} -> {plane}
            %commit = ttg.async_commit_group tokens %copy
            %done = hcug.async_wait %commit {{num_inst = {1 if negative == 'partial_wait' else 0} : i32}}
            {bar}
            {bar}
            '''
            if negative == "token_used":
                begin += f"%used = ttg.local_load %window token %done : {plane} -> {tensor}\n"
            if negative == "global_write":
                begin += f"tt.store %pointers, %zeros : {pointers}\n"
        else:
            begin += f'''%low = ttg.memdesc_subslice %window[0, 0] : {plane} -> {half}
            %read0 = ttg.local_load %low : {half} -> {fragment}
            {bar}
            '''
            if negative == "cross_global_write":
                begin += f"tt.store %pointers, %zeros : {pointers}\n"
            if negative in ("raw_shared_read", "raw_generic_read"):
                space = 3 if negative == "raw_shared_read" else 0
                begin += f"""%address = arith.constant 0 : i64
                %raw = tt.int_to_ptr %address : i64 -> !tt.ptr<{dtype}, {space}>
                %untracked = tt.load %raw : !tt.ptr<{dtype}, {space}>
                """
            read_window = "%window"
            if negative in ("overlap", "shared_write"):
                begin += f"%next_window = ttg.memdesc_index {owner}[%future] : {root} -> {plane}\n"
                read_window = "%next_window"
            begin += f"%high = ttg.memdesc_subslice {read_window}[4, 0] : {plane} -> {half}\n"
            if negative == "unknown_control":
                begin += f'''%test = arith.cmpi eq, {dynamic}, %zero : i32
                scf.if %test {{
                  %read1 = ttg.local_load %high : {half} -> {fragment}
                }}
                '''
            elif negative == "shared_write":
                begin += f"ttg.local_store %half_zeros, %high : {fragment} -> {half}\n"
            else:
                begin += f"%read1 = ttg.local_load %high : {half} -> {fragment}\n"
            begin += bar + "\n"
        if carried:
            begin += '''%increment = arith.addi %carry, %one : i32
            %next_carry = arith.remsi %increment, %slots : i32
            scf.yield %next_carry : i32
            '''
        begin += "}\n"
        if producer:
            if tail_store:
                begin += f"tt.store %pointers, %zeros : {pointers}\n"
            begin += bar + "\n" + bar + "\n"
        if producer and producer_wide:
            begin = re.sub(r"(?<![A-Za-z0-9])i32(?![A-Za-z0-9])", "i128", begin)
            begin = begin.replace("i128 to i128", "i128 to i32")
            begin = begin.replace("num_inst = 0 : i128", "num_inst = 0 : i32")
            begin = begin.replace("num_inst = 1 : i128", "num_inst = 1 : i32")
        return begin + terminator

    prologue = f"%owner = ttg.local_alloc : () -> {root}"
    arguments = f"%p: !tt.ptr<{dtype}>, %n: i32"
    if negative == "external_root":
        prologue = ""
        arguments += f", %owner: {root}"
    if negative == "alias":
        prologue += f"\n%alias = ttg.memdesc_reinterpret %owner : {root} -> {root}"
    kinds = [True] + [False] * consumers if producer_default else [False] * consumers + [True]
    text = f'''#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}}>
#shared = #ttg.swizzled_shared<{{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}}>
#plane = #ttg.swizzled_shared<{{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}}>
module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
"ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"}} {{
 tt.func public @publication({arguments}) {{
  {prologue}
  ttg.warp_specialize(%p, %owner, %n) attributes {{requestedRegisters = array<i32: {', '.join(['32'] * consumers)}>}}
  default {{
    {role(kinds[0], '%p', '%owner', '%n', 'ttg.warp_yield')}
  }}
'''
    for i, producer in enumerate(kinds[1:]):
        text += f'''partition{i}(%p{i}: !tt.ptr<{dtype}>, %owner{i}: {root}, %n{i}: i32) num_warps(4) {{
        {role(producer, f'%p{i}', f'%owner{i}', f'%n{i}', 'ttg.warp_return')}
        }}
'''
    return text + f" : (!tt.ptr<{dtype}>, {root}, i32) -> ()\n tt.return\n }}\n}}"


def _deferred_warp_wait_lower(tmp_path, **kwargs):
    path = tmp_path / "publication.mlir"
    path.write_text(_deferred_warp_wait_source(**kwargs))
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    module = libtriton.ir.parse_mlir_module(str(path), context)
    before = str(module)
    pm = libtriton.ir.pass_manager(context)
    libtriton.hcu.passes.ttgpuir.add_warp_specialize_to_llvm(pm, "gfx936", 4, 4, False, 0, 0, 0)
    pm.run(module, "publication_window_proof")
    after = str(module)
    assert before.count("hcug.async_wait") == after.count("hcug.async_wait") == 1
    for name in ("ttg.local_barrier", "ttg.local_load", "ttg.local_store", "ttg.async_copy_global_to_local"):
        assert before.count(name) == after.count(name)
    return before, after


def _deferred_warp_wait_deferred(text):
    lines = text.splitlines()
    index = next(i for i, line in enumerate(lines) if "hcug.async_wait" in line)
    return "ttg.local_barrier {hcu.cta_scope}" in lines[index - 1]


@pytest.mark.parametrize("dtype", ["bf16", "f16"])
@pytest.mark.parametrize("slots,trips,start,step,consumers,producer_default,carried", [
    (2, 1, 0, 1, 1, False, False),
    (3, 3, 3, 1, 2, True, False),
    (4, 5, 0, 2, 1, False, True),
    (2, 6, 3, 2, 2, True, True),
])
def test_defer_only_across_unobserved_window(tmp_path, dtype, slots, trips, start, step, consumers, producer_default,
                                             carried):
    before, after = _deferred_warp_wait_lower(tmp_path, dtype=dtype, slots=slots, trips=trips, start=start, step=step,
                                              consumers=consumers, producer_default=producer_default, carried=carried)
    assert not _deferred_warp_wait_deferred(before)
    assert _deferred_warp_wait_deferred(after)


@pytest.mark.parametrize("negative", [
    "same_slot", "overlap", "shared_write", "alias", "external_root", "token_used", "partial_wait", "unknown_slot",
    "dynamic_loop", "phase_count", "unknown_control", "global_write"
])
def test_unknown_or_observed_window_keeps_wait(tmp_path, negative):
    before, after = _deferred_warp_wait_lower(tmp_path, negative=negative)
    assert not _deferred_warp_wait_deferred(before)
    assert not _deferred_warp_wait_deferred(after)


def test_negative_wide_producer_matches_normalized_consumer(tmp_path):
    before, after = _deferred_warp_wait_lower(tmp_path, slots=3, producer_wide=True)
    assert not _deferred_warp_wait_deferred(before)
    assert _deferred_warp_wait_deferred(after)


def test_negative_wide_loop_cannot_hide_early_read_alias(tmp_path):
    before, after = _deferred_warp_wait_lower(tmp_path, slots=3, producer_wide=True, negative="overlap")
    assert not _deferred_warp_wait_deferred(before)
    assert not _deferred_warp_wait_deferred(after)


def test_global_write_after_all_waits_does_not_disable_proof(tmp_path):
    before, after = _deferred_warp_wait_lower(tmp_path, tail_store=True)
    assert not _deferred_warp_wait_deferred(before)
    assert _deferred_warp_wait_deferred(after)


def test_other_role_must_not_overwrite_pending_global_source(tmp_path):
    before, after = _deferred_warp_wait_lower(tmp_path, negative="cross_global_write")
    assert not _deferred_warp_wait_deferred(before)
    assert not _deferred_warp_wait_deferred(after)


@pytest.mark.parametrize("negative", ["raw_shared_read", "raw_generic_read"])
def test_raw_pointer_cannot_hide_shared_alias(tmp_path, negative):
    before, after = _deferred_warp_wait_lower(tmp_path, negative=negative)
    assert not _deferred_warp_wait_deferred(before)
    assert not _deferred_warp_wait_deferred(after)


# wave private recycled inputs

import triton
import triton.language as tl
from triton.compiler import ASTSource


@triton.jit
def _wave_private_recycled_inputs_recycled_inputs(A, B, C, BM: tl.constexpr, BN: tl.constexpr):
    part = tl.arange(0, 4)
    mi = tl.arange(0, BM)
    ni = tl.arange(0, BN)
    ki = part[:, None] * 64 + tl.arange(0, 64)[None, :]
    acc = tl.zeros((4, BM, BN), tl.float32)
    for start in range(0, 8):
        kk = start * 256 + ki
        a = tl.load(A + mi[None, :, None] * 2048 + kk[:, None, :])
        b = tl.load(B + ni[None, None, :] * 2048 + kk[:, :, None])
        acc = tl.dot(a, b, acc, allow_tf32=False)
    pairs = tl.reshape(tl.permute(acc, (1, 2, 0)), (BM * BN, 2, 2))
    even, odd = tl.split(pairs)
    lo, hi = tl.split(even + odd)
    value = tl.reshape(lo + hi, (BM, BN))
    tl.store(C + mi[:, None] * BN + ni[None, :], value.to(C.dtype.element_ty))


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("bm,bn", [(16, 16), (32, 32), (16, 32), (32, 16)])
@pytest.mark.parametrize("alignment", [2, 16])
def test_recycled_wave_partition_sync(dtype, bm, bn, alignment):
    signature = {
        name: ('constexpr' if name in ('BM', 'BN') else '*' + dtype)
        for name in _wave_private_recycled_inputs_recycled_inputs.arg_names
    }
    source = ASTSource(_wave_private_recycled_inputs_recycled_inputs, signature, dict(BM=bm, BN=bn),
                       attrs={(i, ): [('tt.divisibility', alignment), ('tt.pointer_range', 32)]
                              for i in range(3)})
    compiled = triton.compile(
        source, target=GPUTarget('hip', 'gfx936', 64),
        options=dict(num_warps=4, num_stages=1, kpack=2, mmac_transposed=False, wave_private_sync=True,
                     disable_llvm_loop_unroll=True, loop_unroll_stage='post-pipeline'))
    asm = compiled.asm['amdgcn']
    loop = re.search(r'(?P<label>\.LBB\d+_\d+):[^\n]*Inner Loop.*?s_cbranch_\w+\s+(?P=label)', asm, re.S)
    assert loop, 'test must exercise a real steady loop'
    barriers = loop.group().count('s_barrier')
    # A and B are separate IDs recycled at the same address. Equal plane
    # pitches preserve wave ownership; unequal pitches must retain CTA edges.
    assert (barriers == 0) == (bm == bn)
    # Cross-wave output reduction remains synchronized in both cases.
    assert asm.count('s_barrier') > barriers


# wave private sync
from triton.experimental.gluon._runtime import GluonASTSource

pytest.importorskip("triton.experimental.gluon.language.hcu")

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDMFMALayout
from triton.experimental.gluon.language.amd.cdna3 import buffer_load, buffer_store, mfma
from triton.experimental.gluon.language.hcu import warp_group_id
from triton.language import range as pipeline_range
from triton.runtime.jit import constexpr_function


@constexpr_function
def _wave_private_sync_producer_layout(rows, cooperative=False):
    assert rows >= 8 and rows & (rows - 1) == 0
    registers = [[0, 0, 1], [0, 0, 2], [0, 0, 4]]
    bit = 8
    while bit < rows:
        registers.append([0, bit, 0])
        bit *= 2
    if cooperative:
        # Same logical domain; exchange one row lane-bit with one batch wave-bit.
        # FullCTA remains numerically valid, private-wave proof MUST decline.
        return gl.DistributedLinearLayout(registers,
                                          [[0, 0, 8], [0, 0, 16], [0, 0, 32], [1, 0, 0], [0, 2, 0], [0, 4, 0]],
                                          [[0, 1, 0], [2, 0, 0]], [], [4, rows, 64])
    # lane K8/K16/K32 then row1/2/4; wave bits select the K64 partition.
    return gl.DistributedLinearLayout(registers, [[0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0], [0, 2, 0], [0, 4, 0]],
                                      [[1, 0, 0], [2, 0, 0]], [], [4, rows, 64])


@constexpr_function
def _wave_private_sync_partial_layout(bm, bn):
    # Actual rank3 HCU LEGACY, transposed=False, native four-accumulator group.
    # Physical offset bits: r0,r1,lane0..5,remaining r,split0,split1.
    bases = [[0, 0, 4], [0, 0, 8], [0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 0, 1], [0, 0, 2]]
    bit = 16
    while bit < bn:
        bases.append([0, 0, bit])
        bit *= 2
    bit = 16
    while bit < bm:
        bases.append([0, bit, 0])
        bit *= 2
    bases += [[1, 0, 0], [2, 0, 0]]
    # This interval is the whole allocation: no interior C padding.
    return gl.PaddedSharedLayout([[4 * bm * bn, 1]], bases, [], [4, bm, bn])


@g.jit
def _wave_private_sync_input_coords(D: gl.constexpr, L: gl.constexpr):
    sp = gl.arange(0, 4, layout=gl.SliceLayout(1, gl.SliceLayout(2, L)))
    row = gl.arange(0, D, layout=gl.SliceLayout(0, gl.SliceLayout(2, L)))
    k = gl.arange(0, 64, layout=gl.SliceLayout(0, gl.SliceLayout(1, L)))
    return row[None, :, None], sp[:, None, None] * 64 + k[None, None, :]


@g.jit
def _wave_private_sync_wave_private_control(A, B, C, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                                            BM: gl.constexpr = 32, BN: gl.constexpr = 32, PROBE: gl.constexpr = 0):
    gl.static_assert(M > 0 and N > 0 and K >= 256 and K % 256 == 0)
    gl.static_assert(BM >= 16 and BN >= 16 and BM & (BM - 1) == 0 and BN & (BN - 1) == 0)
    gl.static_assert(BM + BN <= 112 and BM * BN <= 4096)
    gl.static_assert(2 * M * K < 2147483648 and 2 * N * K < 2147483648 and 2 * M * N < 2147483648)
    gl.static_assert(A.dtype.element_ty == gl.bfloat16 or A.dtype.element_ty == gl.float16)
    gl.static_assert(B.dtype.element_ty == A.dtype.element_ty and C.dtype.element_ty == A.dtype.element_ty)
    mma: gl.constexpr = AMDMFMALayout(3, [16, 16, 16], False, [4, 1, 1], tiles_per_warp=[1, 1, 1], mmac_layout=1)
    da: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    db: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    pa: gl.constexpr = _wave_private_sync_producer_layout(BM, PROBE == 1)
    pb: gl.constexpr = _wave_private_sync_producer_layout(BN, PROBE == 1)
    sa: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[64, 8]], [4, BM, 64],
                                                               [0, 2, 1] if PROBE == 2 else [2, 1, 0])
    sb: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[64, 8]], [4, BN, 64],
                                                               [0, 2, 1] if PROBE == 2 else [2, 1, 0])
    rm, ka = _wave_private_sync_input_coords(BM, pa)
    rn, kb = _wave_private_sync_input_coords(BN, pb)
    row_m = gl.program_id(0) * BM
    col_n = gl.program_id(1) * BN
    if PROBE == 3:
        # A deliberately opaque sideeffect asm must disable ALL private roots.
        gl.inline_asm_elementwise("", constraints="=s,0", args=[row_m], dtype=gl.int32, is_pure=False, pack=1)
    abase = A + row_m.to(gl.int64) * K
    bbase = B + col_n.to(gl.int64) * K
    ao = (rm * K + ka).to(gl.int32)
    bo = (rn * K + kb).to(gl.int32)
    if M % BM == 0:
        am = gl.full([4, BM, 64], True, gl.int1, pa)
    else:
        am = row_m + rm < M
    if N % BN == 0:
        bm = gl.full([4, BN, 64], True, gl.int1, pb)
    else:
        bm = col_n + rn < N
    acc = gl.full([4, BM, BN], 0, gl.float32, mma)
    aa = gl.allocate_shared_memory(A.dtype.element_ty, [4, BM, 64], sa)
    bb = gl.allocate_shared_memory(A.dtype.element_ty, [4, BN, 64], sb)
    av = buffer_load(abase, ao, am, other=0)
    bv = buffer_load(bbase, bo, bm, other=0)
    for q in pipeline_range(0, K // 256, loop_unroll_factor=1):
        aa.store(av)
        bb.store(bv)
        # No explicit A/B CTA: compiler must prove wave-private ownership
        # and supply per-wave LDS waits, or conservatively retain full CTA.
        ar = aa.load(da)
        br = bb.permute([0, 2, 1]).load(db)
        have_next = q + 1 < K // 256
        next_a = buffer_load(abase + (q + 1) * 256, ao, am & have_next, other=0)
        next_b = buffer_load(bbase + (q + 1) * 256, bo, bm & have_next, other=0)
        acc = mfma(ar, br, acc)
        av = next_a
        bv = next_b
    aa._keep_alive()
    bb._keep_alive()

    pc: gl.constexpr = _wave_private_sync_partial_layout(BM, BN)
    cs = gl.allocate_shared_memory(gl.float32, [4, BM, BN], pc)
    cs.store(acc)
    gl.thread_barrier()
    wave = warp_group_id(1)
    if wave == 0:
        # batch=1 makes the MFMA warp bits redundant. All C coordinates live
        # in wave0, with exactly the same native register/lane mapping.
        c0 = cs.slice(0, 1, 0).load(mma)
        c1 = cs.slice(1, 1, 0).load(mma)
        c2 = cs.slice(2, 1, 0).load(mma)
        c3 = cs.slice(3, 1, 0).load(mma)
        value = ((c0 + c1) + (c2 + c3)).reshape([BM, BN])
        cl: gl.constexpr = gl.SliceLayout(0, mma)
        value = gl.convert_layout(value, cl)
        mi = gl.arange(0, BM, layout=gl.SliceLayout(1, cl))
        ni = gl.arange(0, BN, layout=gl.SliceLayout(0, cl))
        co = (mi[:, None] * N + ni[None, :]).to(gl.int32)
        cbase = C + row_m.to(gl.int64) * N + col_n.to(gl.int64)
        buffer_store(value.to(C.dtype.element_ty), cbase, co, (row_m + mi[:, None] < M) & (col_n + ni[None, :] < N))


@pytest.mark.parametrize("probe,dtype,private", [(0, "bf16", True), (0, "fp16", True),
                                                 (1, "bf16", False),  # Producer lanes cross wave ownership.
                                                 (2, "bf16", False),  # Physical planes fail the current certificate.
                                                 (3, "bf16", False),  # Opaque effects retain full synchronization.
                                                 ])
def test_gluon_wave_private_padded_sync(probe, dtype, private):
    # Explicit target compilation never queries or initializes a GPU driver.
    # The kernel keeps an unrelated FP32 C allocation with sliced aliases:
    # declining those aliases must preserve an independent A/B certificate.
    fn = _wave_private_sync_wave_private_control
    constants = dict(M=32, N=32, K=256, BM=32, BN=32, PROBE=probe)
    signature = {name: ("*" + dtype if name in ("A", "B", "C") else "constexpr") for name in fn.arg_names}
    source = GluonASTSource(fn, signature, constants)
    compiled = []
    for enabled in (False, True):
        compiled.append(
            triton.compile(
                source,
                target=GPUTarget("hip", "gfx936", 64),
                options=dict(num_warps=4, num_stages=1, waves_per_eu=1, wave_private_sync=enabled,
                             loop_unroll_stage="post-pipeline", disable_llvm_loop_unroll=True, kpack=2,
                             mmac_transposed=False),
            ))
    before, after = compiled

    def waves(kernel):
        return sum("call" in line and "@llvm.amdgcn.wave.barrier" in line for line in kernel.asm["llir"].splitlines())

    assert waves(before) == 0
    if private:
        # Gluon bypasses make_ttgir: this fails if option forwarding is lost.
        assert waves(after) > 0
        assert before.asm["hsaco"] != after.asm["hsaco"]
    else:
        assert waves(after) == 0
        assert before.asm["hsaco"] == after.asm["hsaco"]


# standard async tokens

from triton.experimental.gluon.language.amd.cdna4.async_copy import (
    buffer_load_to_shared,
    commit_group,
    wait_group,
)

from triton.experimental.gluon.language.hcu import sched_barrier


@g.jit
def _standard_async_tokens_copy(A, O, WITH_TOKEN: gl.constexpr, BAD_TOKEN: gl.constexpr = False):
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [4], [0])
    i = gl.arange(0, 2048, layout=layout)
    mem = gl.allocate_shared_memory(A.dtype.element_ty, [2048], gl.SwizzledSharedLayout(1, 1, 1, [0]))
    buffer_load_to_shared(mem, A, i, i < 1536, other=0)
    commit_group()
    ready = wait_group(0, return_token=WITH_TOKEN)
    if BAD_TOKEN:
        result = mem.load(layout, token=42)
    elif WITH_TOKEN:
        result = mem.load(layout, token=ready)
    else:
        result = mem.load(layout)
    gl.store(O + i, result)


def _standard_async_tokens_compile_copy(dtype, with_token, bad_token=False):
    source = GluonASTSource(_standard_async_tokens_copy,
                            {"A": "*" + dtype, "O": "*" + dtype, "WITH_TOKEN": "constexpr", "BAD_TOKEN": "constexpr"},
                            {"WITH_TOKEN": with_token, "BAD_TOKEN": bad_token}, attrs={(0, ): [("tt.divisibility", 16)],
                                                                                       (1, ):
                                                                                       [("tt.divisibility", 16)]})
    return triton.compile(source, target=GPUTarget("hip", "gfx936", 64), options=dict(num_warps=4, num_stages=1))


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("with_token", [False, True])
def test_standard_wait_result_reaches_local_load(dtype, with_token):
    # Explicit target compilation does not query or initialize a GPU driver.
    kernel = _standard_async_tokens_compile_copy(dtype, with_token)
    loads = re.findall(r"ttg.local_load[^\n]+", kernel.asm["ttgir"])
    assert loads and any(" token " in line for line in loads) == with_token
    assert "ttg.async_wait" in kernel.asm["ttgir"]
    assert "mls_shared" not in kernel.asm["ttgir"]
    assert "hcu.packet" not in kernel.asm["ttgir"]


def test_shared_load_rejects_non_token():
    with pytest.raises(triton.CompilationError, match="expected an async wait token"):
        _standard_async_tokens_compile_copy("bf16", True, bad_token=True)


def test_wait_token_preserves_read_before_overwrite(tmp_path):
    text = _physical_shared_regions_module_source(select_load=True)
    text = text.replace("    ttg.local_barrier", "    %ready = ttg.async_wait {num = 0 : i32}\n    ttg.local_barrier")
    text = text.replace("ttg.local_load %choice :", "ttg.local_load %choice token %ready :")
    # Completion of a previous write does not complete a later shared read.
    # Writing B must still wait for the read which could select B.
    assert _physical_shared_regions_lower_source(tmp_path, text, update_async_wait=True).count("rocdl.s.barrier") == 2


@g.jit
def _standard_async_tokens_schedule_boundary(MASK: gl.constexpr):
    sched_barrier(MASK)


@pytest.mark.parametrize("mask", [0, 1, -1, 1 << 31])
def test_target_schedule_boundary_has_no_packet_or_thread_sync(mask):
    source = GluonASTSource(_standard_async_tokens_schedule_boundary, {"MASK": "constexpr"}, {"MASK": mask})
    if mask < 0 or mask > 0x7fffffff:
        with pytest.raises(triton.CompilationError):
            triton.compile(source, target=GPUTarget("hip", "gfx936", 64), options=dict(num_warps=4))
        return
    kernel = triton.compile(source, target=GPUTarget("hip", "gfx936", 64), options=dict(num_warps=4))
    assert f"@llvm.amdgcn.sched.barrier(i32 {mask})" in kernel.asm["llir"]
    assert "rocdl.s.barrier" not in kernel.asm["ttgir"]
    assert "whole_packet" not in kernel.asm["ttgir"]

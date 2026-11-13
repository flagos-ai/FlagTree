"""CTA publication waits require a closed, cross-role shared-window proof."""
import re

import pytest
from triton._C import libtriton
from triton.backends.compiler import GPUTarget

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler required", allow_module_level=True)

from triton.backends.hcu.compiler import HIPBackend


def source(dtype="bf16", slots=2, trips=3, start=0, step=1,
           consumers=1, producer_default=False, carried=False, negative=None, producer_wide=False, tail_store=False):
    root = f"!ttg.memdesc<{slots}x8x8x{dtype}, #shared, #smem, mutable>"
    plane = f"!ttg.memdesc<8x8x{dtype}, #plane, #smem, mutable>"
    half = f"!ttg.memdesc<4x8x{dtype}, #plane, #smem, mutable, 8x8>"
    tensor = f"tensor<8x8x{dtype}, #blocked>"
    fragment = f"tensor<4x8x{dtype}, #blocked>"
    pointers = f"tensor<8x8x!tt.ptr<{dtype}>, #blocked>"
    bar = "ttg.local_barrier {hcu.cta_scope}"

    def role(producer, ptr, owner, dynamic, terminator):
        kind = "i128" if producer and producer_wide else "i32"
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


def lower(tmp_path, **kwargs):
    path = tmp_path / "publication.mlir"
    path.write_text(source(**kwargs))
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    module = libtriton.ir.parse_mlir_module(str(path), context)
    before = str(module)
    pm = libtriton.ir.pass_manager(context)
    libtriton.hcu.passes.ttgpuir.add_warp_specialize_to_llvm(
        pm, "gfx936", 4, 4, False, 0, 0, 0)
    pm.run(module, "publication_window_proof")
    after = str(module)
    assert before.count("hcug.async_wait") == after.count("hcug.async_wait") == 1
    for name in ("ttg.local_barrier", "ttg.local_load", "ttg.local_store",
                 "ttg.async_copy_global_to_local"):
        assert before.count(name) == after.count(name)
    return before, after


def deferred(text):
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
def test_defer_only_across_unobserved_window(
        tmp_path, dtype, slots, trips, start, step, consumers, producer_default, carried):
    before, after = lower(tmp_path, dtype=dtype, slots=slots, trips=trips,
                          start=start, step=step, consumers=consumers,
                          producer_default=producer_default, carried=carried)
    assert not deferred(before)
    assert deferred(after)


@pytest.mark.parametrize("negative", ["same_slot", "overlap", "shared_write", "alias",
                                    "external_root", "token_used", "partial_wait",
                                    "unknown_slot", "dynamic_loop", "phase_count",
                                    "unknown_control", "global_write"])
def test_unknown_or_observed_window_keeps_wait(tmp_path, negative):
    before, after = lower(tmp_path, negative=negative)
    assert not deferred(before)
    assert not deferred(after)


def test_negative_wide_producer_matches_normalized_consumer(tmp_path):
    before, after = lower(tmp_path, slots=3, producer_wide=True)
    assert not deferred(before)
    assert deferred(after)


def test_negative_wide_loop_cannot_hide_early_read_alias(tmp_path):
    before, after = lower(tmp_path, slots=3, producer_wide=True, negative="overlap")
    assert not deferred(before)
    assert not deferred(after)


def test_global_write_after_all_waits_does_not_disable_proof(tmp_path):
    before, after = lower(tmp_path, tail_store=True)
    assert not deferred(before)
    assert deferred(after)


def test_other_role_must_not_overwrite_pending_global_source(tmp_path):
    before, after = lower(tmp_path, negative="cross_global_write")
    assert not deferred(before)
    assert not deferred(after)


@pytest.mark.parametrize("negative", ["raw_shared_read", "raw_generic_read"])
def test_raw_pointer_cannot_hide_shared_alias(tmp_path, negative):
    before, after = lower(tmp_path, negative=negative)
    assert not deferred(before)
    assert not deferred(after)

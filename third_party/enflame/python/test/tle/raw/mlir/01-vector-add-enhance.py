"""
TLE-Raw GCU MLIR EDSL: Vector Addition (DTE+L1+TAR Pipeline)
=============================================================

Three-stage pipeline matching native Triton IR:
  Stage 1: DTE bulk transfer Global -> L1 local memory (gcu.slice_pad_async)
  Stage 2: TAR chained vector load from L1 (gcu.tar_load)
  Stage 3: Compute + maskedstore write back to global

Persistent kernel pattern:
  - Grid capped at NUM_SM=24 (GCU400: 4 SIC x 6 SIP)
  - Each block loops over tiles with stride=NUM_SM
  - BLOCK_SIZE=16384 per tile, 4 lanes per block (thread_id % 4)
  - Each lane handles 4096 elements via DTE+TAR

Architecture: GCU400
Lowering: gcu-compiler-opt (use_gcu_opt=True, default)
"""
from typing_extensions import Literal as L

from mlir import ir
from mlir.dialects import arith, gpu, memref as memref_d, scf
import torch
import triton
import triton.language as tl
from triton.experimental.tle.raw import dialect, Input
from triton.experimental.tle.raw.tops.gcu_dialect import gcu
import triton.experimental.tle.language.raw as tle_raw

DEVICE = triton.runtime.driver.active.get_active_torch_device()
NUM_SM = 24


def _ci32(i32, v):
    return arith.constant(i32, ir.IntegerAttr.get(i32, v))


def _ci64(i64, v):
    return arith.constant(i64, ir.IntegerAttr.get(i64, v))


def _cidx(idx, v):
    return arith.constant(idx, ir.IntegerAttr.get(idx, v))


def _extsi(val, ty):
    return ir.Operation.create("arith.extsi", operands=[val], results=[ty]).result


def _unroll_tar_add_interleaved(cx, cy, co, stride_tar, vec128xf32, count):
    """Interleaved-load: x0,y0,x1,y1,... then compute+store. Better for small sizes."""
    xvs = []
    yvs = []
    for _ in range(count):
        xv, cx = gcu.tar_load(cx, stride_tar, vec128xf32)
        yv, cy = gcu.tar_load(cy, stride_tar, vec128xf32)
        xvs.append(xv)
        yvs.append(yv)
    for i in range(count):
        res = arith.addf(xvs[i], yvs[i])
        co = gcu.tar_store(res, co, stride_tar)
    return cx, cy, co


def _unroll_tar_add_batched(cx, cy, co, stride_tar, vec128xf32, count):
    """Batched: all x loads, all y loads, all compute+store. Better for large sizes."""
    xvs = []
    for _ in range(count):
        xv, cx = gcu.tar_load(cx, stride_tar, vec128xf32)
        xvs.append(xv)
    yvs = []
    for _ in range(count):
        yv, cy = gcu.tar_load(cy, stride_tar, vec128xf32)
        yvs.append(yv)
    for i in range(count):
        res = arith.addf(xvs[i], yvs[i])
        co = gcu.tar_store(res, co, stride_tar)
    return cx, cy, co


def _trunci(val, ty):
    return ir.Operation.create("arith.trunci", operands=[val], results=[ty]).result


def _cidx_val(v):
    """Create index constant without needing the type from outer scope."""
    idx = ir.IndexType.get()
    return arith.constant(idx, ir.IntegerAttr.get(idx, v))


def _make_edsl(block_size: int, num_sm: int = NUM_SM, use_batched: bool = False, nw: int = 4, uf: int = None):
    """Factory: DTE+L1+TAR persistent kernel.

    Args:
        use_batched: If True, use batched TAR pattern (better for large sizes).
                     If False, use interleaved-load (better for small sizes).
        nw: num_warps (threads per block).
        uf: unroll_factor override (None = auto).
    """
    _unroll_fn = _unroll_tar_add_batched if use_batched else _unroll_tar_add_interleaved
    num_warps = nw
    block_size_m1 = block_size - 1
    elems_per_lane = block_size // num_warps
    buf_bytes = elems_per_lane * 4
    per_thread_l1 = buf_bytes * 2  # x, y buffers
    unroll_factor = uf if uf is not None else min(8, elems_per_lane // 128)
    inner_step = 128 * unroll_factor

    @dialect(name="tops_mlir", use_gcu_opt=True)
    def edsl(
        output: Input[L["!gcu.ptr<f32>"]],
        x: Input[L["!gcu.ptr<f32>"]],
        y: Input[L["!gcu.ptr<f32>"]],
        n_elements: Input[L["i32"]],
    ):
        i32 = ir.IntegerType.get_signless(32)
        i64 = ir.IntegerType.get_signless(64)
        idx = ir.IndexType.get()
        f32 = ir.F32Type.get()
        vec128xf32 = ir.VectorType.get([128], f32)
        vec128xi32 = ir.VectorType.get([128], i32)
        dyn = ir.ShapedType.get_dynamic_size()
        memref_dyn_f32 = ir.MemRefType.get([dyn], f32)
        memref_epl_f32 = ir.MemRefType.get([elems_per_lane], f32)

        gcu_ptr_f32 = ir.Type.parse("!gcu.ptr<f32>")

        c0_i32 = _ci32(i32, 0)
        c0_idx = _cidx(idx, 0)
        c4_idx = _cidx(idx, 4)
        c128_idx = _cidx(idx, 128)
        c_epl_idx = _cidx(idx, elems_per_lane)
        c_bs = _ci32(i32, block_size)
        c_bs_m1 = _ci32(i32, block_size_m1)
        c4_i64 = _ci64(i64, 4)
        c0_i64 = _ci64(i64, 0)
        c_epl_i64 = _ci64(i64, elems_per_lane)
        c_stride = _ci64(i64, 512)
        zero_f32 = arith.constant(f32, ir.FloatAttr.get(f32, 0.0))
        c_num_sm_idx = _cidx(idx, num_sm)
        c_epl_i32 = _ci32(i32, elems_per_lane)

        dte = gcu.alloc_dte("private")

        pid = gpu.block_id(gpu.Dimension.x)
        num_tiles = arith.divsi(arith.addi(n_elements, c_bs_m1), c_bs)
        num_tiles_idx = arith.index_cast(idx, num_tiles)
        num_sm_idx = c_num_sm_idx

        x_base = gcu.ptr2int(x)
        y_base = gcu.ptr2int(y)
        n_i64 = _extsi(n_elements, i64)

        out_mr = gcu.ptr2memref(output, memref_dyn_f32)
        bcast_n = gcu.vector_broadcast(n_elements, vec128xi32)
        out_base_i64 = gcu.ptr2int(output)

        raw_l1 = gcu.alloc_shared_raw()

        tid = gpu.thread_id(gpu.Dimension.x)
        lane = arith.remsi(tid, c4_idx)
        lane_off = arith.muli(lane, c_epl_idx)
        lane_off_i32 = arith.index_cast(i32, lane_off)
        thread_l1_base = arith.muli(lane, _cidx(idx, per_thread_l1))
        lx_a, gx_a = gcu.view_local(raw_l1, f32, elems_per_lane, thread_l1_base)
        ly_a, gy_a = gcu.view_local(raw_l1, f32, elems_per_lane, arith.addi(thread_l1_base, _cidx(idx, buf_bytes)))

        stride_tar = gcu.tar_init(c_stride)

        outer = scf.ForOp(pid, num_tiles_idx, num_sm_idx)
        with ir.InsertionPoint(outer.body):
            tile_i32 = arith.index_cast(i32, outer.induction_variable)
            tile_base = arith.muli(tile_i32, c_bs)
            goff = arith.addi(tile_base, lane_off_i32)
            goff_i64 = _extsi(goff, i64)
            byte_off = arith.muli(goff_i64, c4_i64)

            rem = arith.subi(n_i64, goff_i64)
            rem = arith.maxsi(rem, c0_i64)
            rem = arith.minsi(rem, c_epl_i64)
            rem_i32 = _trunci(rem, i32)

            rem_gt0 = arith.cmpi(arith.CmpIPredicate.sgt, rem_i32, c0_i32)
            if_op = scf.IfOp(rem_gt0)
            with ir.InsertionPoint(if_op.then_block):
                x_ptr = gcu.int2ptr(arith.addi(x_base, byte_off), gcu_ptr_f32)
                x_src_dyn = gcu.ptr2memref(x_ptr, memref_dyn_f32)
                x_src = memref_d.reinterpret_cast(memref_epl_f32, x_src_dyn, offsets=[], sizes=[], strides=[],
                                                  static_offsets=[0], static_sizes=[elems_per_lane], static_strides=[1])
                y_ptr = gcu.int2ptr(arith.addi(y_base, byte_off), gcu_ptr_f32)
                y_src_dyn = gcu.ptr2memref(y_ptr, memref_dyn_f32)
                y_src = memref_d.reinterpret_cast(memref_epl_f32, y_src_dyn, offsets=[], sizes=[], strides=[],
                                                  static_offsets=[0], static_sizes=[elems_per_lane], static_strides=[1])

                gcu.slice_pad_async(dte, gx_a, x_src, [c0_i32], [rem_i32], zero_f32)
                gcu.wait_dte(dte)
                gcu.slice_pad_async(dte, gy_a, y_src, [c0_i32], [rem_i32], zero_f32)
                gcu.wait_dte(dte)

                x_l1 = gcu.ptr2int(gcu.memref2ptr(lx_a, gcu_ptr_f32))
                y_l1 = gcu.ptr2int(gcu.memref2ptr(ly_a, gcu_ptr_f32))
                x_tar = gcu.tar_init(x_l1)
                y_tar = gcu.tar_init(y_l1)

                is_full = arith.cmpi(arith.CmpIPredicate.sge, rem_i32, c_epl_i32)
                if_full = scf.IfOp(is_full, hasElse=True)
                with ir.InsertionPoint(if_full.then_block):
                    out_off = arith.muli(goff_i64, c4_i64)
                    out_tar = gcu.tar_init(arith.addi(out_base_i64, out_off))

                    c_step_idx = _cidx(idx, inner_step)
                    inner_f = scf.ForOp(c0_idx, c_epl_idx, c_step_idx, [x_tar, y_tar, out_tar])
                    with ir.InsertionPoint(inner_f.body):
                        cx = inner_f.inner_iter_args[0]
                        cy = inner_f.inner_iter_args[1]
                        co = inner_f.inner_iter_args[2]

                        cx, cy, co = _unroll_fn(cx, cy, co, stride_tar, vec128xf32, unroll_factor)

                        scf.yield_([cx, cy, co])

                    scf.yield_([])

                with ir.InsertionPoint(if_full.else_block):
                    inner_p = scf.ForOp(c0_idx, c_epl_idx, c128_idx, [x_tar, y_tar])
                    with ir.InsertionPoint(inner_p.body):
                        cx = inner_p.inner_iter_args[0]
                        cy = inner_p.inner_iter_args[1]

                        xv, nx = gcu.tar_load(cx, stride_tar, vec128xf32)
                        yv, ny = gcu.tar_load(cy, stride_tar, vec128xf32)
                        res = arith.addf(xv, yv)

                        chunk = arith.index_cast(i32, inner_p.induction_variable)
                        woff = arith.addi(goff, chunk)
                        sv = gcu.vector_step(woff, vec128xi32)
                        mask = arith.cmpi(arith.CmpIPredicate.slt, sv, bcast_n)
                        widx = arith.index_cast(idx, woff)
                        gcu.maskedstore(out_mr, widx, mask, res)

                        scf.yield_([nx, ny])

                    scf.yield_([])

                scf.yield_([])

            scf.yield_([])

    return edsl


edsl_4096 = _make_edsl(4096, use_batched=True)
edsl_16384 = _make_edsl(16384, use_batched=True)
edsl_32768 = _make_edsl(32768, use_batched=True)
edsl_49152 = _make_edsl(49152, use_batched=True)


@triton.jit
def add_kernel_4096(x_ptr, y_ptr, output_ptr, n_elements):
    tle_raw.call(edsl_4096, [output_ptr, x_ptr, y_ptr, n_elements])


@triton.jit
def add_kernel_16384(x_ptr, y_ptr, output_ptr, n_elements):
    tle_raw.call(edsl_16384, [output_ptr, x_ptr, y_ptr, n_elements])


@triton.jit
def add_kernel_32768(x_ptr, y_ptr, output_ptr, n_elements):
    tle_raw.call(edsl_32768, [output_ptr, x_ptr, y_ptr, n_elements])


@triton.jit
def add_kernel_49152(x_ptr, y_ptr, output_ptr, n_elements):
    tle_raw.call(edsl_49152, [output_ptr, x_ptr, y_ptr, n_elements])


def add(x: torch.Tensor, y: torch.Tensor):
    """Vector add with heuristic BLOCK_SIZE selection (avoids autotune noise)."""
    output = torch.empty_like(x)
    n_elements = output.numel()
    if n_elements <= 65536:
        bs = 4096
        kernel = add_kernel_4096
    elif n_elements <= 524288:
        bs = 16384
        kernel = add_kernel_16384
    elif n_elements <= 8388608:
        bs = 32768
        kernel = add_kernel_32768
    else:
        bs = 49152
        kernel = add_kernel_49152
    grid = (min(NUM_SM, triton.cdiv(n_elements, bs)), )
    kernel[grid](x, y, output, n_elements)
    return output


@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": bs}) for bs in [2**i for i in range(14, 16)]],
    key=["n_elements"],
)
@triton.jit
def _native_add_persistent(x_ptr, y_ptr, output_ptr, n_elements, NUM_SM: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_tile = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(pid, num_tile, NUM_SM):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, x + y, mask=mask)


def native_add_persistent(x: torch.Tensor, y: torch.Tensor):
    """Golden Triton native: persistent with inner loop."""
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (min(NUM_SM, triton.cdiv(n_elements, meta['BLOCK_SIZE'])), )
    _native_add_persistent[grid](x, y, output, n_elements, NUM_SM)
    return output


if __name__ == "__main__":
    torch.manual_seed(0)

    print("=== Correctness check ===")
    for size in [128, 2048, 98432, 1048576]:
        x = torch.randn(size, device=DEVICE)
        y = torch.randn(size, device=DEVICE)
        z = add(x, y)
        ref = x + y
        assert torch.allclose(ref, z), \
            f"size={size} MISMATCH!\n  expected: {ref[:8]}\n  got:      {z[:8]}"
        print(f"  size={size:>10}: PASS")
    print("Correctness PASSED\n")

    print("\n[Benchmark] EDSL vs Triton Native vs Torch  (GB/s)\n")

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['size'],
            x_vals=[2**i for i in range(10, 29)],
            line_arg='provider',
            line_vals=['edsl', 'triton', 'torch'],
            line_names=['EDSL (GB/s)', 'Triton (GB/s)', 'Torch (GB/s)'],
            ylabel='GB/s',
            plot_name='gcu400-edsl-vector-add-performance',
            args={},
        ))
    def benchmark(size, provider):
        x = torch.rand(size, device=DEVICE, dtype=torch.float32)
        y = torch.rand(size, device=DEVICE, dtype=torch.float32)
        if provider == 'torch':
            ms = triton.testing.do_bench(lambda: x + y)
        elif provider == 'triton':
            ms = triton.testing.do_bench(lambda: native_add_persistent(x, y))
        else:
            ms = triton.testing.do_bench(lambda: add(x, y))
        gbps = lambda ms: 3 * size * 4 * 1e-6 / ms
        return gbps(ms)

    benchmark.run(print_data=True)

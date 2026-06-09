"""GCU Dialect Python helpers for TLE-Raw EDSL.

Provides lightweight Python wrappers around GCU MLIR operations
using ir.Operation.create(), avoiding the need for full Python Bindings.

This module covers ALL operations defined in GCUOps.td, organized into:
  - Thread indexing (get_global_thread_id)
  - DTE (Data Transformation Engine) lifecycle & data-flow ops
  - Barrier synchronization
  - Pipeline producer/consumer
  - ESL (Elastic Shared Link) communication
  - Memory conversion (ptr2memref, memref2ptr, int2ptr, ptr2int)
  - Memory helpers (load, store – wraps llvm.getelementptr + llvm.load/store)
  - Gather / Scatter
  - TAR (Thread Address Register) vector load/store
  - MatMul
  - Reduce / ReduceArg
  - Vector ops (vector_convert, vector_movsft, vector_step)
  - Atomic ops (atomic_rmw, atomic_cas)
  - Misc (mfence, begin_clock, end_clock, dynamic_shared_memory,
          assert, border, mem_map/unmap, get_memref_offset,
          materialize_in_destination, extern_elementwise_op,
          builtin_elementwise_op)
  - Warp specialization (warp_yield, warp_return)

Usage:
    from triton.experimental.tle.raw.tops.gcu_dialect import gcu
    # Then in an EDSL function:
    dte = gcu.alloc_dte("private")
"""
from __future__ import annotations

from mlir import ir
from mlir.dialects import arith, llvm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _register_gcu_dialect(ctx: ir.Context) -> None:
    ctx.allow_unregistered_dialects = True


def _i1(ctx: ir.Context) -> ir.IntegerType:
    return ir.IntegerType.get_signless(1, context=ctx)


def _i32(ctx: ir.Context) -> ir.IntegerType:
    return ir.IntegerType.get_signless(32, context=ctx)


def _i64(ctx: ir.Context) -> ir.IntegerType:
    return ir.IntegerType.get_signless(64, context=ctx)


def _index(ctx: ir.Context) -> ir.IndexType:
    return ir.IndexType.get(context=ctx)


def _parse(type_str: str) -> ir.Type:
    return ir.Type.parse(type_str)


# ---------------------------------------------------------------------------
# GCUOps – mirrors every op in GCUOps.td
# ---------------------------------------------------------------------------


class GCUOps:
    """Stateless helper that builds GCU dialect MLIR operations.

    Requires an active MLIR InsertionPoint.

    API Verification Status (E2E = tested on GCU400 hardware):
      [E2E ✓] alloc_dte, wait_dte, slice_pad_async, alloc_shared_raw,
              view_local, ptr2memref(memref_type), memref2ptr, ptr2int,
              dynamic_shared_memory, tar_init, tar_load, maskedstore
      [BUG]   memcpy_async — 0-byte transfer when src offset=0, use slice_pad_async
      [IR ✓]  All other ops generate valid MLIR IR but not E2E tested
    """

    # -----------------------------------------------------------------------
    # Thread indexing
    # -----------------------------------------------------------------------

    @staticmethod
    def get_global_thread_id(res=None) -> ir.Value:
        """gcu.get_global_thread_id -> index"""
        ctx = ir.Context.current
        result_type = res if res is not None else _index(ctx)
        return ir.Operation.create("gcu.get_global_thread_id", results=[result_type]).result

    # -----------------------------------------------------------------------
    # DTE lifecycle
    # -----------------------------------------------------------------------

    @staticmethod
    def alloc_shared_raw() -> ir.Value:
        """gcu.dynamic_shared_memory() -> memref<?xi8, local>.

        Returns the raw byte-level local memory handle.
        Multiple views can be carved from the same handle at different offsets.
        """
        raw_ty = ir.MemRefType.get([ir.ShapedType.get_dynamic_size()], ir.IntegerType.get_signless(8),
                                   memory_space=ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 9))
        return gcu.dynamic_shared_memory(raw_ty)

    @staticmethod
    def view_local(raw: ir.Value, elem_type: ir.Type, num_elems: int,
                   byte_offset: ir.Value) -> tuple[ir.Value, ir.Value]:
        """Create a typed view into raw local memory at given byte offset.

        Mirrors native Triton IR pattern:
          %view = "memref.view"(%raw, %off) -> memref<Nxf32, local>

        Returns (typed_local_memref, generic_memref) pair.
        The generic memref (no address space) is suitable for DTE/ptr ops.
        """
        local_ty = ir.MemRefType.get([num_elems], elem_type,
                                     memory_space=ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 9))
        gen_ty = ir.MemRefType.get([num_elems], elem_type)
        typed = ir.Operation.create("memref.view", operands=[raw, byte_offset], results=[local_ty]).result
        generic = ir.Operation.create("memref.memory_space_cast", operands=[typed], results=[gen_ty]).result
        return typed, generic

    @staticmethod
    def alloc_dte(address_space: str = "private") -> ir.Value:
        """gcu.alloc_dte -> !gcu.dte<address_space>"""
        dte_type = _parse(f'!gcu.dte<{address_space}>')
        return ir.Operation.create("gcu.alloc_dte", results=[dte_type]).result

    @staticmethod
    def init_dte(dte: ir.Value) -> None:
        ir.Operation.create("gcu.init_dte", operands=[dte])

    @staticmethod
    def destroy_dte(dte: ir.Value) -> None:
        ir.Operation.create("gcu.destroy_dte", operands=[dte])

    @staticmethod
    def dealloc_dte(dte: ir.Value) -> None:
        ir.Operation.create("gcu.dealloc_dte", operands=[dte])

    @staticmethod
    def trigger_dte(dte: ir.Value) -> None:
        ir.Operation.create("gcu.trigger_dte", operands=[dte])

    @staticmethod
    def wait_dte(dte: ir.Value) -> None:
        ir.Operation.create("gcu.wait_dte", operands=[dte])

    @staticmethod
    def connect_dte(from_dte: ir.Value, to_dte: ir.Value) -> None:
        ir.Operation.create("gcu.connect_dte", operands=[from_dte, to_dte])

    @staticmethod
    def set_dst_addr(dte: ir.Value, addr: ir.Value) -> None:
        ir.Operation.create("gcu.set_dst_addr", operands=[dte, addr])

    @staticmethod
    def set_src_offset(dte: ir.Value, dim: ir.Value, offset: ir.Value) -> None:
        ir.Operation.create("gcu.set_src_offset", operands=[dte, dim, offset])

    @staticmethod
    def set_dst_offset(dte: ir.Value, dim: ir.Value, offset: ir.Value) -> None:
        ir.Operation.create("gcu.set_dst_offset", operands=[dte, dim, offset])

    # -----------------------------------------------------------------------
    # DTE data-flow (async): memcpy / memset / slice / deslice / transpose
    #   broadcast / mirror / composite operations
    # -----------------------------------------------------------------------

    @staticmethod
    def memcpy_async(dte: ir.Value, dst: ir.Value, src: ir.Value) -> None:
        """WARNING: gcu-compiler-opt has a known bug where memcpy_async with
        byte_offset=0 source memref produces a 0-byte transfer size.
        Use slice_pad_async with explicit shape parameters instead."""
        ir.Operation.create("gcu.memcpy_async", operands=[dte, dst, src])

    @staticmethod
    def memset_async(dte: ir.Value, dst: ir.Value, value: ir.Value) -> None:
        ir.Operation.create("gcu.memset_async", operands=[dte, dst, value])

    @staticmethod
    def slice_async(dte: ir.Value, dst: ir.Value, src: ir.Value, offsets: list[ir.Value],
                    default_value: ir.Value) -> None:
        ir.Operation.create("gcu.slice_async", operands=[dte, dst, src, *offsets, default_value])

    @staticmethod
    def slice_pad_async(dte: ir.Value, dst: ir.Value, src: ir.Value, offsets: list[ir.Value],
                        slice_shape: list[ir.Value], pad_value: ir.Value) -> None:
        seg = [1, 1, 1, len(offsets), len(slice_shape), 1]
        seg_attr = ir.DenseI32ArrayAttr.get(seg)
        ir.Operation.create("gcu.slice_pad_async", operands=[dte, dst, src, *offsets, *slice_shape, pad_value],
                            attributes={"operandSegmentSizes": seg_attr})

    @staticmethod
    def deslice_async(dte: ir.Value, dst: ir.Value, src: ir.Value, offsets: list[ir.Value]) -> None:
        ir.Operation.create("gcu.deslice_async", operands=[dte, dst, src, *offsets])

    @staticmethod
    def slice_deslice_async(dte: ir.Value, dst: ir.Value, src: ir.Value, offsets: list[ir.Value],
                            slice_shape: list[ir.Value], dst_offsets: list[ir.Value]) -> None:
        ir.Operation.create("gcu.slice_deslice_async", operands=[dte, dst, src, *offsets, *slice_shape, *dst_offsets])

    @staticmethod
    def transpose_async(dte: ir.Value, dst: ir.Value, src: ir.Value, layout: list[ir.Value]) -> None:
        ir.Operation.create("gcu.transpose_async", operands=[dte, dst, src, *layout])

    @staticmethod
    def broadcast_async(dte: ir.Value, dst: ir.Value, src: ir.Value) -> None:
        ir.Operation.create("gcu.broadcast_async", operands=[dte, dst, src])

    @staticmethod
    def slice_broadcast_async(dte: ir.Value, dst: ir.Value, src: ir.Value, offsets: list[ir.Value],
                              slice_shape: list[ir.Value]) -> None:
        ir.Operation.create("gcu.slice_broadcast_async", operands=[dte, dst, src, *offsets, *slice_shape])

    @staticmethod
    def slice_transpose_async(dte: ir.Value, dst: ir.Value, src: ir.Value, offsets: list[ir.Value],
                              layout: list[ir.Value], pad_value: ir.Value) -> None:
        ir.Operation.create("gcu.slice_transpose_async", operands=[dte, dst, src, *offsets, *layout, pad_value])

    @staticmethod
    def transpose_deslice_async(dte: ir.Value, dst: ir.Value, src: ir.Value, layout: list[ir.Value],
                                offsets: list[ir.Value]) -> None:
        ir.Operation.create("gcu.transpose_deslice_async", operands=[dte, dst, src, *layout, *offsets])

    @staticmethod
    def memset_deslice_async(dte: ir.Value, dst: ir.Value, src: ir.Value, offsets: list[ir.Value],
                             value: ir.Value) -> None:
        ir.Operation.create("gcu.memset_deslice_async", operands=[dte, dst, src, *offsets, value])

    # mirror ops

    @staticmethod
    def mirror_tb_async(dte: ir.Value, dst: ir.Value, src: ir.Value) -> None:
        ir.Operation.create("gcu.mirror_tb_async", operands=[dte, dst, src])

    @staticmethod
    def mirror_lr_async(dte: ir.Value, dst: ir.Value, src: ir.Value) -> None:
        ir.Operation.create("gcu.mirror_lr_async", operands=[dte, dst, src])

    @staticmethod
    def mirror_tb_pad_async(dte: ir.Value, dst: ir.Value, src: ir.Value, pad_low: list[ir.Value],
                            pad_high: list[ir.Value], pad_mid: list[ir.Value], pad_value: ir.Value) -> None:
        ir.Operation.create("gcu.mirror_tb_pad_async",
                            operands=[dte, dst, src, *pad_low, *pad_high, *pad_mid, pad_value])

    @staticmethod
    def mirror_lr_pad_async(dte: ir.Value, dst: ir.Value, src: ir.Value, pad_low: list[ir.Value],
                            pad_high: list[ir.Value], pad_mid: list[ir.Value], pad_value: ir.Value) -> None:
        ir.Operation.create("gcu.mirror_lr_pad_async",
                            operands=[dte, dst, src, *pad_low, *pad_high, *pad_mid, pad_value])

    @staticmethod
    def mirror_tb_deslice_async(dte: ir.Value, dst: ir.Value, src: ir.Value, dst_offsets: list[ir.Value]) -> None:
        ir.Operation.create("gcu.mirror_tb_deslice_async", operands=[dte, dst, src, *dst_offsets])

    @staticmethod
    def mirror_lr_deslice_async(dte: ir.Value, dst: ir.Value, src: ir.Value, dst_offsets: list[ir.Value]) -> None:
        ir.Operation.create("gcu.mirror_lr_deslice_async", operands=[dte, dst, src, *dst_offsets])

    # -----------------------------------------------------------------------
    # Barrier synchronization
    # -----------------------------------------------------------------------

    @staticmethod
    def alloc_barrier(address_space: str = "shared") -> ir.Value:
        barrier_type = _parse(f'!gcu.barrier<{address_space}>')
        return ir.Operation.create("gcu.alloc_barrier", results=[barrier_type]).result

    @staticmethod
    def init_barrier(barrier: ir.Value, count: ir.Value) -> None:
        ir.Operation.create("gcu.init_barrier", operands=[barrier, count])

    @staticmethod
    def arrive_and_wait_barrier(barrier: ir.Value) -> None:
        ir.Operation.create("gcu.arrive_and_wait_barrier", operands=[barrier])

    @staticmethod
    def arrive_barrier(barrier: ir.Value) -> None:
        ir.Operation.create("gcu.arrive_barrier", operands=[barrier])

    @staticmethod
    def wait_barrier(barrier: ir.Value) -> None:
        ir.Operation.create("gcu.wait_barrier", operands=[barrier])

    @staticmethod
    def destroy_barrier(barrier: ir.Value) -> None:
        ir.Operation.create("gcu.destroy_barrier", operands=[barrier])

    @staticmethod
    def dealloc_barrier(barrier: ir.Value) -> None:
        ir.Operation.create("gcu.dealloc_barrier", operands=[barrier])

    # -----------------------------------------------------------------------
    # Pipeline (gcu dialect – distinct from gcuws dialect)
    # -----------------------------------------------------------------------

    @staticmethod
    def alloc_pipeline(pipeline_type_str: str) -> ir.Value:
        """gcu.alloc_pipeline -> !gcu.pipeline<...>

        Args:
            pipeline_type_str: e.g. "2, 1, 1, true" for
                !gcu.pipeline<2, 1, 1, true>
        """
        pt = _parse(f'!gcu.pipeline<{pipeline_type_str}>')
        return ir.Operation.create("gcu.alloc_pipeline", results=[pt]).result

    @staticmethod
    def init_pipeline(pipeline: ir.Value) -> None:
        ir.Operation.create("gcu.init_pipeline", operands=[pipeline])

    @staticmethod
    def producer_acquire(pipeline: ir.Value) -> None:
        ir.Operation.create("gcu.producer_acquire", operands=[pipeline])

    @staticmethod
    def producer_commit(pipeline: ir.Value) -> None:
        ir.Operation.create("gcu.producer_commit", operands=[pipeline])

    @staticmethod
    def consumer_wait(pipeline: ir.Value) -> None:
        ir.Operation.create("gcu.consumer_wait", operands=[pipeline])

    @staticmethod
    def consumer_release(pipeline: ir.Value) -> None:
        ir.Operation.create("gcu.consumer_release", operands=[pipeline])

    # -----------------------------------------------------------------------
    # ESL (Elastic Shared Link) communication
    # -----------------------------------------------------------------------

    @staticmethod
    def alloc_esl_engine(ibgda_info_ptr: ir.Value, qp_shared_states_ptr: ir.Value, rank: ir.Value,
                         num_ranks: ir.Value) -> ir.Value:
        engine_type = _parse('!gcu.esl_engine')
        return ir.Operation.create("gcu.alloc_esl_engine",
                                   operands=[ibgda_info_ptr, qp_shared_states_ptr, rank,
                                             num_ranks], results=[engine_type]).result

    @staticmethod
    def alloc_esl_endpoint() -> ir.Value:
        ep_type = _parse('!gcu.esl_endpoint')
        return ir.Operation.create("gcu.alloc_esl_endpoint", results=[ep_type]).result

    @staticmethod
    def esl_get_qp(engine: ir.Value, dst_rank: ir.Value, qp_idx: ir.Value, endpoint: ir.Value) -> None:
        ir.Operation.create("gcu.esl_get_qp", operands=[engine, dst_rank, qp_idx, endpoint])

    @staticmethod
    def esl_get_num_rc_per_pe(engine: ir.Value) -> ir.Value:
        ctx = ir.Context.current
        return ir.Operation.create("gcu.esl_get_num_rc_per_pe", operands=[engine], results=[_i32(ctx)]).result

    @staticmethod
    def esl_get_wqe_slots(engine: ir.Value, endpoint: ir.Value, num_wqes: ir.Value) -> ir.Value:
        ctx = ir.Context.current
        return ir.Operation.create("gcu.esl_get_wqe_slots", operands=[engine, endpoint, num_wqes],
                                   results=[_i64(ctx)]).result

    @staticmethod
    def esl_ring_db(engine: ir.Value, endpoint: ir.Value, wqe_idx: ir.Value, num_wqes: ir.Value) -> ir.Value:
        ctx = ir.Context.current
        return ir.Operation.create("gcu.esl_ring_db", operands=[engine, endpoint, wqe_idx, num_wqes],
                                   results=[_i64(ctx)]).result

    @staticmethod
    def esl_wait_done(engine: ir.Value, endpoint: ir.Value, wait_id: ir.Value) -> None:
        ir.Operation.create("gcu.esl_wait_done", operands=[engine, endpoint, wait_id])

    @staticmethod
    def esl_send(engine: ir.Value, dst: ir.Value, src: ir.Value, endpoint: ir.Value, wqe_idx: ir.Value, nelem: ir.Value,
                 fence: ir.Value) -> None:
        ir.Operation.create("gcu.esl_send", operands=[engine, dst, src, endpoint, wqe_idx, nelem, fence])

    @staticmethod
    def esl_get_rdma_peer_base(prims: ir.Value, rank: ir.Value) -> ir.Value:
        ctx = ir.Context.current
        return ir.Operation.create("gcu.esl_get_rdma_peer_base", operands=[prims, rank], results=[_i64(ctx)]).result

    @staticmethod
    def esl_get_ibgda_info_ptr(prims: ir.Value) -> ir.Value:
        ctx = ir.Context.current
        return ir.Operation.create("gcu.esl_get_ibgda_info_ptr", operands=[prims], results=[_i64(ctx)]).result

    @staticmethod
    def esl_get_qp_shared_states_ptr(prims: ir.Value) -> ir.Value:
        ctx = ir.Context.current
        return ir.Operation.create("gcu.esl_get_qp_shared_states_ptr", operands=[prims], results=[_i64(ctx)]).result

    # -----------------------------------------------------------------------
    # Memory conversion
    # -----------------------------------------------------------------------

    @staticmethod
    def ptr2memref(ptr: ir.Value, result_type_or_elem: ir.Type) -> "ir.Value | GCUMemRef":
        """Convert a GCU pointer to memref or GCUMemRef wrapper.

        When result_type_or_elem is a memref type (e.g. memref<?xf32>):
            Returns ir.Value from actual gcu.ptr2memref MLIR op.
        When result_type_or_elem is a scalar type (e.g. f32):
            Returns GCUMemRef Python wrapper for use in executable EDSL.
        """
        type_str = str(result_type_or_elem)
        if type_str.startswith("memref"):
            return ir.Operation.create("gcu.ptr2memref", results=[result_type_or_elem], operands=[ptr]).result
        else:
            return GCUMemRef(ptr, result_type_or_elem)

    @staticmethod
    def memref2ptr(memref: ir.Value, ptr_type: ir.Type) -> ir.Value:
        return ir.Operation.create("gcu.memref2ptr", results=[ptr_type], operands=[memref]).result

    @staticmethod
    def ptr2int(ptr: ir.Value) -> ir.Value:
        ctx = ir.Context.current
        return ir.Operation.create("gcu.ptr2int", results=[_i64(ctx)], operands=[ptr]).result

    @staticmethod
    def int2ptr(value: ir.Value, ptr_type: ir.Type) -> ir.Value:
        return ir.Operation.create("gcu.int2ptr", results=[ptr_type], operands=[value]).result

    @staticmethod
    def get_memref_offset(memref: ir.Value, offsets: list[ir.Value] | None = None) -> ir.Value:
        ctx = ir.Context.current
        ops = [memref] + (offsets or [])
        return ir.Operation.create("gcu.get_memref_offset", results=[_index(ctx)], operands=ops).result

    @staticmethod
    def materialize_in_destination(source: ir.Value, dest: ir.Value) -> None:
        ir.Operation.create("gcu.materialize_in_destination", operands=[source, dest])

    @staticmethod
    def dynamic_shared_memory(memref_type: ir.Type) -> ir.Value:
        return ir.Operation.create("gcu.dynamic_shared_memory", results=[memref_type]).result

    # -----------------------------------------------------------------------
    # Memory helpers (load / store over LLVM pointers)
    # -----------------------------------------------------------------------

    @staticmethod
    def load(ptr: ir.Value, indices: list, elem_type: ir.Type = None) -> ir.Value:
        """Load element from a GCU pointer (mapped to !llvm.ptr<1>).

        Emits llvm.getelementptr + llvm.load.
        """
        if elem_type is None:
            elem_type = ir.Type.parse("f32")
        ptrty = ir.Type.parse("!llvm.ptr<1>")
        idx = indices[0]
        if str(idx.type) == "index":
            idx = arith.index_cast(ir.IntegerType.get_signless(32), idx)
        gep = llvm.getelementptr(ptrty, ptr, [idx], [-2147483648], elem_type, 0)
        return llvm.load(elem_type, gep)

    @staticmethod
    def store(value: ir.Value, ptr: ir.Value, indices: list, elem_type: ir.Type = None) -> None:
        """Store element to a GCU pointer (mapped to !llvm.ptr<1>).

        Emits llvm.getelementptr + llvm.store.
        """
        if elem_type is None:
            elem_type = ir.Type.parse("f32")
        ptrty = ir.Type.parse("!llvm.ptr<1>")
        idx = indices[0]
        if str(idx.type) == "index":
            idx = arith.index_cast(ir.IntegerType.get_signless(32), idx)
        gep = llvm.getelementptr(ptrty, ptr, [idx], [-2147483648], elem_type, 0)
        llvm.store(value, gep)

    # -----------------------------------------------------------------------
    # ptr2memref – returns a Python-level GCUMemRef wrapper
    #
    # Mirrors gcu_compiler.ir pattern:
    #   %mr = "gcu.ptr2memref"(%ptr) : (!gcu.ptr<f32>) -> memref<?xf32>
    #   %val = "vector.maskedload"(%mr, %idx, ...) OR memref.load %mr[%idx]
    #
    # Under the hood, load/store emit llvm.getelementptr + llvm.load/store
    # which the EDSL pipeline can lower to llvm.func.
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Gather / Scatter
    # -----------------------------------------------------------------------

    @staticmethod
    def gather_load(dst: ir.Value, src: ir.Value, offsets: ir.Value, masks: ir.Value, others: ir.Value,
                    size: ir.Value) -> None:
        ir.Operation.create("gcu.gather_load", operands=[dst, src, offsets, masks, others, size])

    @staticmethod
    def scatter_store(dst: ir.Value, src: ir.Value, offsets: ir.Value, masks: ir.Value, size: ir.Value) -> None:
        ir.Operation.create("gcu.scatter_store", operands=[dst, src, offsets, masks, size])

    # -----------------------------------------------------------------------
    # TAR (Thread Address Register) – GCU400 only
    # -----------------------------------------------------------------------

    @staticmethod
    def _tar_type():
        return ir.VectorType.get([1], ir.IntegerType.get_signless(64))

    @staticmethod
    def tar_init(addr_i64: ir.Value) -> ir.Value:
        return ir.Operation.create("gcu.tar_init", operands=[addr_i64], results=[gcu._tar_type()]).result

    @staticmethod
    def tar_load(src_addr: ir.Value, stride: ir.Value, vec_type: ir.Type) -> tuple[ir.Value, ir.Value]:
        op = ir.Operation.create("gcu.tar_load", operands=[src_addr, stride], results=[vec_type, gcu._tar_type()])
        return op.results[0], op.results[1]

    @staticmethod
    def tar_store(v: ir.Value, src_addr: ir.Value, stride: ir.Value) -> ir.Value:
        return ir.Operation.create("gcu.tar_store", operands=[v, src_addr, stride], results=[gcu._tar_type()]).result

    @staticmethod
    def tar_gather(src_addr: ir.Value, num: ir.Value, other: ir.Value, vec_type: ir.Type,
                   mask: ir.Value = None) -> tuple[ir.Value, ir.Value]:
        tar_type = _parse('!gcu.tar')
        ops = [src_addr, num, other]
        if mask is not None:
            ops.append(mask)
        op = ir.Operation.create("gcu.tar_gather", operands=ops, results=[vec_type, tar_type])
        return op.results[0], op.results[1]

    @staticmethod
    def tar_scatter(src_addr: ir.Value, v: ir.Value, num: ir.Value, mask: ir.Value = None) -> ir.Value:
        tar_type = _parse('!gcu.tar')
        ops = [src_addr, v, num]
        if mask is not None:
            ops.append(mask)
        return ir.Operation.create("gcu.tar_scatter", operands=ops, results=[tar_type]).result

    # -----------------------------------------------------------------------
    # MatMul
    # -----------------------------------------------------------------------

    @staticmethod
    def matmul(out: ir.Value, lhs: ir.Value, rhs: ir.Value, bias: ir.Value = None) -> None:
        operands = [out, lhs, rhs]
        if bias is not None:
            operands.append(bias)
        ir.Operation.create("gcu.matmul", operands=operands)

    # -----------------------------------------------------------------------
    # Reduce / ReduceArg
    # -----------------------------------------------------------------------

    @staticmethod
    def reduce(op_attr: str, out: ir.Value, inp: ir.Value, axis: int, workspace: ir.Value = None) -> None:
        """gcu.reduce with named reduction op (e.g. "add", "max")."""
        operands = [out, inp]
        if workspace is not None:
            operands.append(workspace)
        ir.Operation.create(
            "gcu.reduce", operands=operands, attributes={
                "op": ir.Attribute.parse(f'#gcu<reduce_op {op_attr}>'), "axis":
                ir.IntegerAttr.get(_i32(ir.Context.current), axis)
            })

    @staticmethod
    def reduce_arg(op_attr: str, out: ir.Value, out_index: ir.Value, inp: ir.Value, in_index: ir.Value, axis: int,
                   workspace: ir.Value = None) -> None:
        operands = [out, out_index, inp, in_index]
        if workspace is not None:
            operands.append(workspace)
        ir.Operation.create(
            "gcu.reduce_arg", operands=operands, attributes={
                "op": ir.Attribute.parse(f'#gcu<reduce_op {op_attr}>'), "axis":
                ir.IntegerAttr.get(_i32(ir.Context.current), axis)
            })

    # -----------------------------------------------------------------------
    # Vector ops
    # -----------------------------------------------------------------------

    @staticmethod
    def vector_convert(inputs: list[ir.Value], output_types: list[ir.Type]) -> list[ir.Value]:
        op = ir.Operation.create("gcu.vector_convert", operands=inputs, results=output_types)
        return list(op.results)

    @staticmethod
    def vector_movsft(vin: ir.Value, unit: int, mode: str = "left") -> ir.Value:
        """gcu.vector_movsft with mode attr and unit attr."""
        return ir.Operation.create(
            "gcu.vector_movsft", operands=[vin], results=[vin.type], attributes={
                "mode": ir.Attribute.parse(f'#gcu<vector_movsft_mode {mode}>'),
                "unit": ir.IntegerAttr.get(_i32(ir.Context.current), unit),
            }).result

    @staticmethod
    def vector_step(start: ir.Value, result_type: ir.Type) -> ir.Value:
        return ir.Operation.create("gcu.vector_step", operands=[start], results=[result_type]).result

    # -----------------------------------------------------------------------
    # Atomic ops
    # -----------------------------------------------------------------------

    @staticmethod
    def atomic_rmw(rmw_op: str, ptr: ir.Value, val: ir.Value, result_type: ir.Type, sem: str = "relaxed",
                   scope: str = "gpu") -> ir.Value:
        return ir.Operation.create(
            "gcu.atomic_rmw", operands=[ptr, val], results=[result_type], attributes={
                "atomic_rmw_op": ir.Attribute.parse(f'#gcu<atomic_rmw_op {rmw_op}>'),
                "sem": ir.Attribute.parse(f'#gcu<mem_semantic {sem}>'),
                "scope": ir.Attribute.parse(f'#gcu<mem_sync_scope {scope}>'),
            }).result

    @staticmethod
    def atomic_cas(ptr: ir.Value, cmp: ir.Value, val: ir.Value, result_type: ir.Type, sem: str = "relaxed",
                   scope: str = "gpu") -> ir.Value:
        return ir.Operation.create(
            "gcu.atomic_cas", operands=[ptr, cmp, val], results=[result_type], attributes={
                "sem": ir.Attribute.parse(f'#gcu<mem_semantic {sem}>'),
                "scope": ir.Attribute.parse(f'#gcu<mem_sync_scope {scope}>'),
            }).result

    # -----------------------------------------------------------------------
    # Border (region markers)
    # -----------------------------------------------------------------------

    @staticmethod
    def border(enter: str = None, leave: str = None) -> None:
        """Emit gcu.border op for marking fusion region boundaries.

        Usage:
            gcu.border(enter="tt.get_program_id")
            gcu.border(leave="triton_gcu.elementwise_fusion_region")
        """
        attrs = {}
        ctx = ir.Context.current
        if enter is not None:
            attrs["enter"] = ir.StringAttr.get(enter, context=ctx)
        if leave is not None:
            attrs["leave"] = ir.StringAttr.get(leave, context=ctx)
        ir.Operation.create("gcu.border", attributes=attrs)

    # -----------------------------------------------------------------------
    # Vector helpers (wrappers for vector dialect ops)
    # -----------------------------------------------------------------------

    @staticmethod
    def vector_broadcast(scalar: ir.Value, vector_type: ir.Type) -> ir.Value:
        """vector.broadcast: scalar → vector.
        Named vector_broadcast to avoid confusion with DTE broadcast_async."""
        return ir.Operation.create("vector.broadcast", operands=[scalar], results=[vector_type]).result

    @staticmethod
    def constant_mask(mask_dims: list[int], mask_type: ir.Type) -> ir.Value:
        """vector.constant_mask with given active lanes."""
        return ir.Operation.create("vector.constant_mask", results=[mask_type],
                                   attributes={"mask_dim_sizes": ir.DenseI64ArrayAttr.get(mask_dims)}).result

    @staticmethod
    def iota_vector(vec_type: ir.Type) -> ir.Value:
        """Create iota vector constant [0, 1, 2, ..., N-1] of given vector type."""
        shape = ir.VectorType(vec_type).shape
        n = shape[0]
        elem_type = ir.VectorType(vec_type).element_type
        attrs = [ir.IntegerAttr.get(elem_type, i) for i in range(n)]
        dense = ir.DenseElementsAttr.get(attrs, type=vec_type)
        return ir.Operation.create("arith.constant", results=[vec_type], attributes={"value": dense}).result

    @staticmethod
    def ptr_to_memref(ptr: ir.Value, memref_type: ir.Type) -> ir.Value:
        """Cast !llvm.ptr<1> to memref<?xT> via unrealized_conversion_cast.

        Use this in executable EDSL kernels where function args are !llvm.ptr<1>
        but vector.maskedload/maskedstore require memref operands.
        NOTE: This may not survive LLVM translation; prefer vec_masked_load/store.
        """
        return ir.Operation.create("builtin.unrealized_conversion_cast", operands=[ptr], results=[memref_type]).result

    @staticmethod
    def vec_masked_load(base_ptr: ir.Value, offset: ir.Value, mask: ir.Value, passthru: ir.Value, result_type: ir.Type,
                        elem_type: ir.Type = None, alignment: int = 4) -> ir.Value:
        """Vectorized masked load via llvm.getelementptr + llvm.intr.masked.load.

        Computes effective address = base_ptr + offset * sizeof(elem),
        then performs masked load returning vector of results.
        """
        if elem_type is None:
            elem_type = ir.F32Type.get()
        ptr_type = ir.Type.parse("!llvm.ptr<1>")
        # Cast offset to i64 for getelementptr
        i64 = ir.IntegerType.get_signless(64)
        offset_i64 = ir.Operation.create("llvm.sext", operands=[offset], results=[i64]).result
        # getelementptr
        eff_ptr = ir.Operation.create(
            "llvm.getelementptr", operands=[base_ptr, offset_i64], results=[ptr_type], attributes={
                "elem_type": ir.TypeAttr.get(elem_type),
                "rawConstantIndices": ir.DenseI32ArrayAttr.get([-2147483648]),
            }).result
        # llvm.intr.masked.load
        return ir.Operation.create(
            "llvm.intr.masked.load", operands=[eff_ptr, mask, passthru], results=[result_type],
            attributes={"alignment": ir.IntegerAttr.get(ir.IntegerType.get_signless(32), alignment)}).result

    @staticmethod
    def vec_masked_store(base_ptr: ir.Value, offset: ir.Value, mask: ir.Value, value: ir.Value,
                         elem_type: ir.Type = None, alignment: int = 4) -> None:
        """Vectorized masked store via llvm.getelementptr + llvm.intr.masked.store.

        Computes effective address = base_ptr + offset * sizeof(elem),
        then performs masked store of value vector.
        """
        if elem_type is None:
            elem_type = ir.F32Type.get()
        ptr_type = ir.Type.parse("!llvm.ptr<1>")
        i64 = ir.IntegerType.get_signless(64)
        offset_i64 = ir.Operation.create("llvm.sext", operands=[offset], results=[i64]).result
        eff_ptr = ir.Operation.create(
            "llvm.getelementptr", operands=[base_ptr, offset_i64], results=[ptr_type], attributes={
                "elem_type": ir.TypeAttr.get(elem_type),
                "rawConstantIndices": ir.DenseI32ArrayAttr.get([-2147483648]),
            }).result
        ir.Operation.create("llvm.intr.masked.store", operands=[value, eff_ptr, mask],
                            attributes={"alignment": ir.IntegerAttr.get(ir.IntegerType.get_signless(32), alignment)})

    @staticmethod
    def maskedload(memref_val: ir.Value, index: ir.Value, mask: ir.Value, passthru: ir.Value,
                   result_type: ir.Type) -> ir.Value:
        """vector.maskedload: vectorized memory read with mask."""
        return ir.Operation.create("vector.maskedload", operands=[memref_val, index, mask, passthru],
                                   results=[result_type]).result

    @staticmethod
    def maskedstore(memref_val: ir.Value, index: ir.Value, mask: ir.Value, value: ir.Value) -> None:
        """vector.maskedstore: vectorized memory write with mask."""
        ir.Operation.create("vector.maskedstore", operands=[memref_val, index, mask, value])

    # -----------------------------------------------------------------------
    # Misc
    # -----------------------------------------------------------------------

    @staticmethod
    def mfence() -> None:
        ir.Operation.create("gcu.mfence")

    @staticmethod
    def begin_clock() -> ir.Value:
        return ir.Operation.create("gcu.begin_clock", results=[_i64(ir.Context.current)]).result

    @staticmethod
    def end_clock() -> ir.Value:
        return ir.Operation.create("gcu.end_clock", results=[_i64(ir.Context.current)]).result

    @staticmethod
    def device_assert(condition: ir.Value, message: str = "", file: str = "", func: str = "", line: int = 0) -> None:
        ctx = ir.Context.current
        ir.Operation.create(
            "gcu.assert", operands=[condition], attributes={
                "message": ir.StringAttr.get(message, context=ctx),
                "file": ir.StringAttr.get(file, context=ctx),
                "func": ir.StringAttr.get(func, context=ctx),
                "line": ir.IntegerAttr.get(_i32(ctx), line),
            })

    @staticmethod
    def mem_map(ptr: ir.Value, num: ir.Value) -> ir.Value:
        """GCU300 only: mmu map memory."""
        return ir.Operation.create("gcu.mem_map", operands=[ptr, num], results=[_i32(ir.Context.current)]).result

    @staticmethod
    def mem_unmap(addr: ir.Value, num: ir.Value) -> None:
        """GCU300 only: mmu unmap memory."""
        ir.Operation.create("gcu.mem_unmap", operands=[addr, num])

    @staticmethod
    def extern_elementwise_op(srcs: list[ir.Value], result_type: ir.Type, symbol: str) -> ir.Value:
        ctx = ir.Context.current
        return ir.Operation.create("gcu.extern_elementwise_op", operands=srcs, results=[result_type],
                                   attributes={"symbol": ir.StringAttr.get(symbol, context=ctx)}).result

    @staticmethod
    def builtin_elementwise_op(output: ir.Value, inputs: list[ir.Value], symbol: str,
                               params: list[ir.Value] | None = None) -> None:
        ctx = ir.Context.current
        ops = [output, *inputs]
        if params:
            ops.extend(params)
        ir.Operation.create("gcu.builtin_elementwise_op", operands=ops,
                            attributes={"symbol": ir.StringAttr.get(symbol, context=ctx)})


gcu = GCUOps()

# ---------------------------------------------------------------------------
# GCUMemRef – Python-level memref wrapper over !llvm.ptr<1>
# ---------------------------------------------------------------------------


class GCUMemRef:
    """Memref-like Python wrapper over a GCU pointer (!llvm.ptr<1>).

    WARNING: Does NOT work with use_gcu_opt=True pipeline because kernel
    arguments are !gcu.ptr<f32> (not !llvm.ptr<1>), and llvm.getelementptr
    rejects non-LLVM pointer types. Only use with non-gcu-opt pipelines.

    Created by gcu.ptr2memref(ptr, elem_type).  Mirrors the IR pattern:
        %mr = "gcu.ptr2memref"(%ptr) : (!gcu.ptr<f32>) -> memref<?xf32>
        %v  = memref.load %mr[%i] : memref<?xf32>

    Under the hood, .load() and .store() emit llvm.getelementptr + llvm.load/store
    so the generated IR passes through convert-func-to-llvm cleanly.
    """

    def __init__(self, ptr: ir.Value, elem_type: ir.Type):
        self._ptr = ptr
        self._elem_type = elem_type

    @property
    def ptr(self) -> ir.Value:
        return self._ptr

    @property
    def elem_type(self) -> ir.Type:
        return self._elem_type

    def load(self, *indices) -> ir.Value:
        """Load a scalar element at the given index.

        Usage: val = mr.load(i)
        Mirrors: memref.load %mr[%i] : memref<?xf32>
        """
        idx = indices[0] if len(indices) == 1 else indices
        if isinstance(idx, (list, tuple)):
            idx = idx[0]
        ptrty = ir.Type.parse("!llvm.ptr<1>")
        i = idx
        if str(i.type) == "index":
            i = arith.index_cast(ir.IntegerType.get_signless(32), i)
        gep = llvm.getelementptr(ptrty, self._ptr, [i], [-2147483648], self._elem_type, 0)
        return llvm.load(self._elem_type, gep)

    def store(self, value: ir.Value, *indices) -> None:
        """Store a scalar element at the given index.

        Usage: mr.store(val, i)
        Mirrors: memref.store %val, %mr[%i] : memref<?xf32>
        """
        idx = indices[0] if len(indices) == 1 else indices
        if isinstance(idx, (list, tuple)):
            idx = idx[0]
        ptrty = ir.Type.parse("!llvm.ptr<1>")
        i = idx
        if str(i.type) == "index":
            i = arith.index_cast(ir.IntegerType.get_signless(32), i)
        gep = llvm.getelementptr(ptrty, self._ptr, [i], [-2147483648], self._elem_type, 0)
        llvm.store(value, gep)


# ---------------------------------------------------------------------------
# GCUWS (Warp Specialization) dialect helpers
# ---------------------------------------------------------------------------


class GCUWSOps:
    """Python helpers for the GCUWS (GCU Warp Specialization) dialect.

    Dialect name: "gcuws"
    """

    @staticmethod
    def init_pipeline(
        stage_count: int = 2,
        producer_count: int = 1,
        consumer_count: int = 1,
        inner_barrier: bool = True,
    ) -> ir.Value:
        ib = "true" if inner_barrier else "false"
        pipeline_type = _parse(f'!gcuws.pipeline<{stage_count}, {producer_count}, '
                               f'{consumer_count}, {ib}>')
        return ir.Operation.create("gcuws.init_pipeline", results=[pipeline_type]).result

    @staticmethod
    def producer_acquire(pipeline: ir.Value) -> None:
        ir.Operation.create("gcuws.producer_acquire", operands=[pipeline])

    @staticmethod
    def producer_commit(pipeline: ir.Value) -> None:
        ir.Operation.create("gcuws.producer_commit", operands=[pipeline])

    @staticmethod
    def consumer_wait(pipeline: ir.Value) -> None:
        ir.Operation.create("gcuws.consumer_wait", operands=[pipeline])

    @staticmethod
    def consumer_release(pipeline: ir.Value) -> None:
        ir.Operation.create("gcuws.consumer_release", operands=[pipeline])


gcuws = GCUWSOps()

# ---------------------------------------------------------------------------
# GCU Warp Specialize ops (in the 'gcu' dialect)
# ---------------------------------------------------------------------------


class GCUWarpOps:
    """Helpers for gcu.warp_yield / gcu.warp_return."""

    @staticmethod
    def warp_yield(values: list[ir.Value] | None = None) -> None:
        ir.Operation.create("gcu.warp_yield", operands=values or [])

    @staticmethod
    def warp_return() -> None:
        ir.Operation.create("gcu.warp_return")


gcu_warp = GCUWarpOps()

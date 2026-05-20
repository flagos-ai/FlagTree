"""
GCU Dialect EDSL: Unit Tests for MLIR IR Generation
====================================================

Tests that gcu_dialect.py correctly generates GCU dialect MLIR operations
using ir.Operation.create() with unregistered dialects.

These tests are cardless-safe — they only verify IR construction, not
runtime execution.

Usage:
    python test_gcu_dialect_ir.py
"""
import importlib.util
import sys
from pathlib import Path

from mlir import ir
from mlir.dialects import arith, func, memref, scf

_this_dir = Path(__file__).resolve().parent
_gcu_mod_path = _this_dir.parent.parent.parent.parent / "triton" / "experimental" / "tle" / "raw" / "tops" / "gcu_dialect.py"
if not _gcu_mod_path.exists():
    _gcu_mod_path = Path(
        sys.prefix
    ) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "triton" / "experimental" / "tle" / "raw" / "tops" / "gcu_dialect.py"

spec = importlib.util.spec_from_file_location("gcu_dialect", str(_gcu_mod_path))
_gcu_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_gcu_mod)
gcu = _gcu_mod.gcu
gcuws = _gcu_mod.gcuws
GCUWarpOps = _gcu_mod.GCUWarpOps
_register_gcu_dialect = _gcu_mod._register_gcu_dialect


def _make_ctx() -> ir.Context:
    ctx = ir.Context()
    _register_gcu_dialect(ctx)
    return ctx


def test_get_global_thread_id():
    """gcu.get_global_thread_id should produce an index-typed result."""
    ctx = _make_ctx()
    with ctx, ir.Location.unknown():
        mod = ir.Module.create()
        with ir.InsertionPoint(mod.body):
            fnty = ir.FunctionType.get([], [])
            fn = func.FuncOp("kernel", fnty, visibility="public")
            block = fn.add_entry_block()
            with ir.InsertionPoint(block):
                tid = gcu.get_global_thread_id()
                assert str(tid.type) == "index", f"expected index, got {tid.type}"
                func.return_([])
        ir_text = str(mod)
        assert "gcu.get_global_thread_id" in ir_text
    print("[PASS] test_get_global_thread_id")


def test_dte_lifecycle():
    """Full DTE lifecycle: alloc → init → trigger → wait → destroy → dealloc."""
    ctx = _make_ctx()
    with ctx, ir.Location.unknown():
        mod = ir.Module.create()
        with ir.InsertionPoint(mod.body):
            fnty = ir.FunctionType.get([], [])
            fn = func.FuncOp("dte_lifecycle", fnty, visibility="public")
            block = fn.add_entry_block()
            with ir.InsertionPoint(block):
                dte = gcu.alloc_dte("private")
                gcu.init_dte(dte)
                gcu.trigger_dte(dte)
                gcu.wait_dte(dte)
                gcu.destroy_dte(dte)
                gcu.dealloc_dte(dte)
                func.return_([])
        ir_text = str(mod)
        for op_name in [
                "gcu.alloc_dte",
                "gcu.init_dte",
                "gcu.trigger_dte",
                "gcu.wait_dte",
                "gcu.destroy_dte",
                "gcu.dealloc_dte",
        ]:
            assert op_name in ir_text, f"missing {op_name}"
        assert "!gcu.dte<private>" in ir_text
    print("[PASS] test_dte_lifecycle")


def test_dte_connect():
    """Two DTEs connected for chained data movement."""
    ctx = _make_ctx()
    with ctx, ir.Location.unknown():
        mod = ir.Module.create()
        with ir.InsertionPoint(mod.body):
            fnty = ir.FunctionType.get([], [])
            fn = func.FuncOp("dte_connect", fnty, visibility="public")
            block = fn.add_entry_block()
            with ir.InsertionPoint(block):
                dte1 = gcu.alloc_dte("private")
                dte2 = gcu.alloc_dte("private")
                gcu.init_dte(dte1)
                gcu.init_dte(dte2)
                gcu.connect_dte(dte1, dte2)
                gcu.trigger_dte(dte1)
                gcu.trigger_dte(dte2)
                gcu.wait_dte(dte2)
                gcu.destroy_dte(dte1)
                gcu.destroy_dte(dte2)
                gcu.dealloc_dte(dte1)
                gcu.dealloc_dte(dte2)
                func.return_([])
        ir_text = str(mod)
        assert "gcu.connect_dte" in ir_text
    print("[PASS] test_dte_connect")


def test_barrier_lifecycle():
    """Full barrier lifecycle: alloc → init → arrive_and_wait → destroy → dealloc."""
    ctx = _make_ctx()
    with ctx, ir.Location.unknown():
        mod = ir.Module.create()
        with ir.InsertionPoint(mod.body):
            i32 = ir.IntegerType.get_signless(32)
            fnty = ir.FunctionType.get([], [])
            fn = func.FuncOp("barrier_test", fnty, visibility="public")
            block = fn.add_entry_block()
            with ir.InsertionPoint(block):
                barrier = gcu.alloc_barrier("shared")
                count = arith.constant(i32, 6)
                gcu.init_barrier(barrier, count)
                gcu.arrive_and_wait_barrier(barrier)
                gcu.destroy_barrier(barrier)
                gcu.dealloc_barrier(barrier)
                func.return_([])
        ir_text = str(mod)
        for op_name in [
                "gcu.alloc_barrier",
                "gcu.init_barrier",
                "gcu.arrive_and_wait_barrier",
                "gcu.destroy_barrier",
                "gcu.dealloc_barrier",
        ]:
            assert op_name in ir_text, f"missing {op_name}"
        assert "!gcu.barrier<shared>" in ir_text
    print("[PASS] test_barrier_lifecycle")


def test_memcpy_async():
    """gcu.memcpy_async with DTE and memref operands."""
    ctx = _make_ctx()
    with ctx, ir.Location.unknown():
        mod = ir.Module.create()
        with ir.InsertionPoint(mod.body):
            f32 = ir.F32Type.get()
            memref_ty = ir.MemRefType.get([1024], f32)
            fnty = ir.FunctionType.get([memref_ty, memref_ty], [])
            fn = func.FuncOp("memcpy_test", fnty, visibility="public")
            block = fn.add_entry_block()
            with ir.InsertionPoint(block):
                dst = block.arguments[0]
                src = block.arguments[1]
                dte = gcu.alloc_dte("private")
                gcu.init_dte(dte)
                gcu.memcpy_async(dte, dst, src)
                gcu.wait_dte(dte)
                gcu.destroy_dte(dte)
                gcu.dealloc_dte(dte)
                func.return_([])
        ir_text = str(mod)
        assert "gcu.memcpy_async" in ir_text
    print("[PASS] test_memcpy_async")


def test_ptr_conversions():
    """ptr2memref, memref2ptr, ptr2int, int2ptr roundtrip."""
    ctx = _make_ctx()
    with ctx, ir.Location.unknown():
        mod = ir.Module.create()
        with ir.InsertionPoint(mod.body):
            ptr_ty = ir.Type.parse("!llvm.ptr<1>")
            f32 = ir.F32Type.get()
            memref_ty = ir.MemRefType.get([1024], f32)
            fnty = ir.FunctionType.get([ptr_ty], [])
            fn = func.FuncOp("ptr_conv", fnty, visibility="public")
            block = fn.add_entry_block()
            with ir.InsertionPoint(block):
                ptr = block.arguments[0]
                mr = gcu.ptr2memref(ptr, memref_ty)
                ptr2 = gcu.memref2ptr(mr, ptr_ty)
                ival = gcu.ptr2int(ptr2)
                _ptr3 = gcu.int2ptr(ival, ptr_ty)
                func.return_([])
        ir_text = str(mod)
        assert "gcu.ptr2memref" in ir_text
        assert "gcu.memref2ptr" in ir_text
        assert "gcu.ptr2int" in ir_text
        assert "gcu.int2ptr" in ir_text
    print("[PASS] test_ptr_conversions")


def test_mfence():
    """gcu.mfence produces a zero-result operation."""
    ctx = _make_ctx()
    with ctx, ir.Location.unknown():
        mod = ir.Module.create()
        with ir.InsertionPoint(mod.body):
            fnty = ir.FunctionType.get([], [])
            fn = func.FuncOp("fence_test", fnty, visibility="public")
            block = fn.add_entry_block()
            with ir.InsertionPoint(block):
                gcu.mfence()
                func.return_([])
        ir_text = str(mod)
        assert "gcu.mfence" in ir_text
    print("[PASS] test_mfence")


def test_clock():
    """begin_clock / end_clock should return i64 values."""
    ctx = _make_ctx()
    with ctx, ir.Location.unknown():
        mod = ir.Module.create()
        with ir.InsertionPoint(mod.body):
            fnty = ir.FunctionType.get([], [])
            fn = func.FuncOp("clock_test", fnty, visibility="public")
            block = fn.add_entry_block()
            with ir.InsertionPoint(block):
                t0 = gcu.begin_clock()
                t1 = gcu.end_clock()
                assert str(t0.type) == "i64"
                assert str(t1.type) == "i64"
                func.return_([])
        ir_text = str(mod)
        assert "gcu.begin_clock" in ir_text
        assert "gcu.end_clock" in ir_text
    print("[PASS] test_clock")


def test_dynamic_shared_memory():
    """dynamic_shared_memory should return the requested memref type."""
    ctx = _make_ctx()
    with ctx, ir.Location.unknown():
        mod = ir.Module.create()
        with ir.InsertionPoint(mod.body):
            f32 = ir.F32Type.get()
            smem_ty = ir.MemRefType.get([4096], f32)
            fnty = ir.FunctionType.get([], [])
            fn = func.FuncOp("smem_test", fnty, visibility="public")
            block = fn.add_entry_block()
            with ir.InsertionPoint(block):
                smem = gcu.dynamic_shared_memory(smem_ty)
                assert str(smem.type) == "memref<4096xf32>"
                func.return_([])
        ir_text = str(mod)
        assert "gcu.dynamic_shared_memory" in ir_text
    print("[PASS] test_dynamic_shared_memory")


def test_combined_dte_barrier_kernel():
    """Realistic kernel pattern: DTE memcpy + barrier sync + compute."""
    ctx = _make_ctx()
    with ctx, ir.Location.unknown():
        mod = ir.Module.create()
        with ir.InsertionPoint(mod.body):
            f32 = ir.F32Type.get()
            i32 = ir.IntegerType.get_signless(32)
            memref_ty = ir.MemRefType.get([1024], f32)
            fnty = ir.FunctionType.get([memref_ty, memref_ty, memref_ty], [])
            fn = func.FuncOp("dte_barrier_kernel", fnty, visibility="public")
            block = fn.add_entry_block()
            with ir.InsertionPoint(block):
                dst = block.arguments[0]
                src_a = block.arguments[1]
                src_b = block.arguments[2]

                dte = gcu.alloc_dte("private")
                gcu.init_dte(dte)

                barrier = gcu.alloc_barrier("shared")
                count = arith.constant(i32, 6)
                gcu.init_barrier(barrier, count)

                gcu.memcpy_async(dte, dst, src_a)
                gcu.wait_dte(dte)

                gcu.arrive_and_wait_barrier(barrier)

                gcu.memcpy_async(dte, dst, src_b)
                gcu.wait_dte(dte)

                gcu.mfence()

                gcu.destroy_barrier(barrier)
                gcu.dealloc_barrier(barrier)
                gcu.destroy_dte(dte)
                gcu.dealloc_dte(dte)
                func.return_([])

        ir_text = str(mod)
        expected_ops = [
            "gcu.alloc_dte",
            "gcu.init_dte",
            "gcu.alloc_barrier",
            "gcu.init_barrier",
            "gcu.memcpy_async",
            "gcu.wait_dte",
            "gcu.arrive_and_wait_barrier",
            "gcu.mfence",
            "gcu.destroy_barrier",
            "gcu.destroy_dte",
        ]
        for op_name in expected_ops:
            assert op_name in ir_text, f"missing {op_name}"
    print("[PASS] test_combined_dte_barrier_kernel")


def test_print_sample_ir():
    """Print a sample IR module for visual inspection."""
    ctx = _make_ctx()
    with ctx, ir.Location.unknown():
        mod = ir.Module.create()
        with ir.InsertionPoint(mod.body):
            f32 = ir.F32Type.get()
            i32 = ir.IntegerType.get_signless(32)
            ptr_ty = ir.Type.parse("!llvm.ptr<1>")
            memref_ty = ir.MemRefType.get([256], f32)
            fnty = ir.FunctionType.get([ptr_ty, ptr_ty], [])
            fn = func.FuncOp("vector_add_gcu", fnty, visibility="public")
            block = fn.add_entry_block()
            with ir.InsertionPoint(block):
                tid = gcu.get_global_thread_id()

                src_mr = gcu.ptr2memref(block.arguments[0], memref_ty)
                dst_mr = gcu.ptr2memref(block.arguments[1], memref_ty)

                dte = gcu.alloc_dte("private")
                gcu.init_dte(dte)
                gcu.memcpy_async(dte, dst_mr, src_mr)
                gcu.wait_dte(dte)
                gcu.destroy_dte(dte)
                gcu.dealloc_dte(dte)

                gcu.mfence()
                func.return_([])

    print("\n=== Sample GCU Dialect MLIR IR ===")
    print(mod)
    print("=== End ===\n")
    print("[PASS] test_print_sample_ir")


def test_gcuws_pipeline_lifecycle():
    """GCUWS pipeline: init → producer_acquire → commit → consumer_wait → release."""
    ctx = _make_ctx()
    with ctx, ir.Location.unknown():
        mod = ir.Module.create()
        with ir.InsertionPoint(mod.body):
            fnty = ir.FunctionType.get([], [])
            fn = func.FuncOp("ws_pipeline_test", fnty, visibility="public")
            block = fn.add_entry_block()
            with ir.InsertionPoint(block):
                pipeline = gcuws.init_pipeline(stage_count=3, producer_count=2, consumer_count=4)
                gcuws.producer_acquire(pipeline)
                gcuws.producer_commit(pipeline)
                gcuws.consumer_wait(pipeline)
                gcuws.consumer_release(pipeline)
                func.return_([])
        ir_text = str(mod)
        for op_name in [
                "gcuws.init_pipeline",
                "gcuws.producer_acquire",
                "gcuws.producer_commit",
                "gcuws.consumer_wait",
                "gcuws.consumer_release",
        ]:
            assert op_name in ir_text, f"missing {op_name}"
        assert "!gcuws.pipeline<3, 2, 4, true>" in ir_text
    print("[PASS] test_gcuws_pipeline_lifecycle")


def test_gcuws_pipeline_with_dte():
    """Combined GCUWS pipeline + DTE data movement pattern."""
    ctx = _make_ctx()
    with ctx, ir.Location.unknown():
        mod = ir.Module.create()
        with ir.InsertionPoint(mod.body):
            f32 = ir.F32Type.get()
            mr_ty = ir.MemRefType.get([1024], f32)
            fnty = ir.FunctionType.get([mr_ty, mr_ty], [])
            fn = func.FuncOp("ws_dte_combined", fnty, visibility="public")
            block = fn.add_entry_block()
            with ir.InsertionPoint(block):
                src = block.arguments[0]
                dst = block.arguments[1]

                pipeline = gcuws.init_pipeline(stage_count=2, producer_count=1, consumer_count=1)
                dte = gcu.alloc_dte("private")
                gcu.init_dte(dte)

                gcuws.producer_acquire(pipeline)
                gcu.memcpy_async(dte, dst, src)
                gcu.wait_dte(dte)
                gcuws.producer_commit(pipeline)

                gcuws.consumer_wait(pipeline)
                gcuws.consumer_release(pipeline)

                gcu.mfence()
                gcu.destroy_dte(dte)
                gcu.dealloc_dte(dte)
                func.return_([])
        ir_text = str(mod)
        assert "gcuws.init_pipeline" in ir_text
        assert "gcuws.producer_acquire" in ir_text
        assert "gcu.memcpy_async" in ir_text
        assert "gcuws.consumer_wait" in ir_text
    print("[PASS] test_gcuws_pipeline_with_dte")


def test_warp_yield_and_return():
    """gcu.warp_yield and gcu.warp_return ops."""
    ctx = _make_ctx()
    with ctx, ir.Location.unknown():
        mod = ir.Module.create()
        with ir.InsertionPoint(mod.body):
            i32 = ir.IntegerType.get_signless(32)
            fnty = ir.FunctionType.get([i32], [])
            fn = func.FuncOp("warp_ops_test", fnty, visibility="public")
            block = fn.add_entry_block()
            with ir.InsertionPoint(block):
                val = block.arguments[0]
                GCUWarpOps.warp_yield([val])
                GCUWarpOps.warp_return()
                func.return_([])
        ir_text = str(mod)
        assert "gcu.warp_yield" in ir_text
        assert "gcu.warp_return" in ir_text
    print("[PASS] test_warp_yield_and_return")


def test_print_gcuws_ir():
    """Print a sample GCUWS pipeline IR for visual inspection."""
    ctx = _make_ctx()
    with ctx, ir.Location.unknown():
        mod = ir.Module.create()
        with ir.InsertionPoint(mod.body):
            f32 = ir.F32Type.get()
            i32 = ir.IntegerType.get_signless(32)
            mr_ty = ir.MemRefType.get([512], f32)
            fnty = ir.FunctionType.get([mr_ty, mr_ty], [])
            fn = func.FuncOp("gemm_ws_pipeline", fnty, visibility="public")
            block = fn.add_entry_block()
            with ir.InsertionPoint(block):
                src = block.arguments[0]
                dst = block.arguments[1]

                pipeline = gcuws.init_pipeline(
                    stage_count=3,
                    producer_count=2,
                    consumer_count=4,
                    inner_barrier=True,
                )

                dte = gcu.alloc_dte("private")
                gcu.init_dte(dte)

                barrier = gcu.alloc_barrier("shared")
                count = arith.constant(i32, 6)
                gcu.init_barrier(barrier, count)

                gcuws.producer_acquire(pipeline)
                gcu.memcpy_async(dte, dst, src)
                gcu.wait_dte(dte)
                gcuws.producer_commit(pipeline)

                gcu.arrive_and_wait_barrier(barrier)

                gcuws.consumer_wait(pipeline)
                gcuws.consumer_release(pipeline)

                gcu.mfence()
                gcu.destroy_barrier(barrier)
                gcu.dealloc_barrier(barrier)
                gcu.destroy_dte(dte)
                gcu.dealloc_dte(dte)
                func.return_([])

    print("\n=== Sample GCUWS Pipeline MLIR IR ===")
    print(mod)
    print("=== End ===\n")
    print("[PASS] test_print_gcuws_ir")


def test_alloc_shared_raw_and_view_local():
    """alloc_shared_raw + view_local produces correct memref types."""
    ctx = _make_ctx()
    with ctx, ir.Location.unknown():
        mod = ir.Module.create()
        with ir.InsertionPoint(mod.body):
            f32 = ir.F32Type.get()
            idx = ir.IndexType.get()
            fnty = ir.FunctionType.get([], [])
            fn = func.FuncOp("l1_alloc_test", fnty, visibility="public")
            block = fn.add_entry_block()
            with ir.InsertionPoint(block):
                raw = gcu.alloc_shared_raw()
                c0 = arith.constant(idx, 0)
                lx, gx = gcu.view_local(raw, f32, 128, c0)
                assert "memref<128xf32, 9>" in str(lx.type) or "local" in str(lx.type)
                assert "memref<128xf32>" in str(gx.type)
                func.return_([])
        ir_text = str(mod)
        assert "gcu.dynamic_shared_memory" in ir_text
        assert "memref.view" in ir_text
        assert "memref.memory_space_cast" in ir_text
    print("[PASS] test_alloc_shared_raw_and_view_local")


def test_slice_pad_async():
    """gcu.slice_pad_async with operandSegmentSizes attribute."""
    ctx = _make_ctx()
    with ctx, ir.Location.unknown():
        mod = ir.Module.create()
        with ir.InsertionPoint(mod.body):
            f32 = ir.F32Type.get()
            i32 = ir.IntegerType.get_signless(32)
            memref_ty = ir.MemRefType.get([1024], f32)
            dte_ty = ir.Type.parse("!gcu.dte<private>")
            fnty = ir.FunctionType.get([memref_ty, memref_ty], [])
            fn = func.FuncOp("slice_pad_test", fnty, visibility="public")
            block = fn.add_entry_block()
            with ir.InsertionPoint(block):
                dst = block.arguments[0]
                src = block.arguments[1]
                dte = gcu.alloc_dte("private")
                offset = arith.constant(i32, 0)
                shape = arith.constant(i32, 256)
                pad_val = arith.constant(f32, 0.0)
                gcu.slice_pad_async(dte, dst, src, [offset], [shape], pad_val)
                gcu.wait_dte(dte)
                func.return_([])
        ir_text = str(mod)
        assert "gcu.slice_pad_async" in ir_text
        assert "operandSegmentSizes" in ir_text
    print("[PASS] test_slice_pad_async")


def test_tar_init_load_store():
    """TAR init, load, and store produce correct types."""
    ctx = _make_ctx()
    with ctx, ir.Location.unknown():
        mod = ir.Module.create()
        with ir.InsertionPoint(mod.body):
            f32 = ir.F32Type.get()
            i32 = ir.IntegerType.get_signless(32)
            i64 = ir.IntegerType.get_signless(64)
            vec_f32 = ir.VectorType.get([128], f32)
            fnty = ir.FunctionType.get([i64], [])
            fn = func.FuncOp("tar_test", fnty, visibility="public")
            block = fn.add_entry_block()
            with ir.InsertionPoint(block):
                addr = block.arguments[0]
                stride = arith.constant(i32, 512)
                tar = gcu.tar_init(addr)
                assert "vector<1xi64>" in str(tar.type)
                vec_val, tar2 = gcu.tar_load(tar, stride, vec_f32)
                assert str(vec_val.type) == "vector<128xf32>"
                tar3 = gcu.tar_store(vec_val, tar2, stride)
                assert "vector<1xi64>" in str(tar3.type)
                func.return_([])
        ir_text = str(mod)
        assert "gcu.tar_init" in ir_text
        assert "gcu.tar_load" in ir_text
        assert "gcu.tar_store" in ir_text
    print("[PASS] test_tar_init_load_store")


def test_vector_broadcast_renamed():
    """vector_broadcast (renamed from broadcast) creates vector.broadcast op."""
    ctx = _make_ctx()
    with ctx, ir.Location.unknown():
        mod = ir.Module.create()
        with ir.InsertionPoint(mod.body):
            f32 = ir.F32Type.get()
            vec_f32 = ir.VectorType.get([128], f32)
            fnty = ir.FunctionType.get([], [])
            fn = func.FuncOp("bcast_test", fnty, visibility="public")
            block = fn.add_entry_block()
            with ir.InsertionPoint(block):
                scalar = arith.constant(f32, 1.0)
                vec = gcu.vector_broadcast(scalar, vec_f32)
                assert str(vec.type) == "vector<128xf32>"
                func.return_([])
        ir_text = str(mod)
        assert "vector.broadcast" in ir_text
    print("[PASS] test_vector_broadcast_renamed")


if __name__ == "__main__":
    tests = [
        test_get_global_thread_id,
        test_dte_lifecycle,
        test_dte_connect,
        test_barrier_lifecycle,
        test_memcpy_async,
        test_ptr_conversions,
        test_mfence,
        test_clock,
        test_dynamic_shared_memory,
        test_combined_dte_barrier_kernel,
        test_alloc_shared_raw_and_view_local,
        test_slice_pad_async,
        test_tar_init_load_store,
        test_vector_broadcast_renamed,
        test_print_sample_ir,
        test_gcuws_pipeline_lifecycle,
        test_gcuws_pipeline_with_dte,
        test_warp_yield_and_return,
        test_print_gcuws_ir,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    if failed:
        sys.exit(1)
    print("All tests passed!")

"""
Test: use_gcu_opt=True with gcu.ptr2memref + vector.maskedload/maskedstore.

This test verifies the "native GCU IR" path where:
  1. Function signature keeps !gcu.ptr<f32> (map_gcu_ptr=False automatically)
  2. gcu.ptr2memref converts pointer to memref<?xf32, 1>
  3. vector.maskedload / vector.maskedstore operate on the memref
  4. gcu-compiler-opt handles all lowering (including GCU-specific ops)

This path mirrors gcu_compiler.ir lines 21-23:
  %mr = "gcu.ptr2memref"(%ptr) : (!gcu.ptr<f32>) -> memref<?xf32, 1>
"""
from typing_extensions import Literal as L

from mlir import ir
from mlir.dialects import arith, gpu
from triton.experimental.tle.raw import dialect, Input
from triton.experimental.tle.raw.tops.gcu_dialect import gcu


@dialect(name="tops_mlir", use_gcu_opt=True)
def edsl_ptr2memref(
    output: Input[L["!gcu.ptr<f32>"]],
    x: Input[L["!gcu.ptr<f32>"]],
    y: Input[L["!gcu.ptr<f32>"]],
    n_elements: Input[L["i32"]],
):
    i32 = ir.IntegerType.get_signless(32)
    idx_ty = ir.IndexType.get()
    f32 = ir.F32Type.get()
    vec128xf32 = ir.VectorType.get([128], f32)
    vec128xi1 = ir.VectorType.get([128], ir.IntegerType.get_signless(1))
    dyn = ir.ShapedType.get_dynamic_size()
    memref_type = ir.MemRefType.get([dyn], f32)

    zeros = arith.constant(
        vec128xf32,
        ir.DenseElementsAttr.get_splat(vec128xf32, ir.FloatAttr.get(f32, 0.0)))
    c128 = arith.constant(i32, ir.IntegerAttr.get(i32, 128))

    block_id = gpu.block_id(gpu.Dimension.x)
    block_id_i32 = arith.index_cast(i32, block_id)
    global_offset = arith.muli(block_id_i32, c128)

    # Convert !gcu.ptr<f32> → memref<?xf32, 1>
    x_memref = gcu.ptr2memref(x, memref_type)
    y_memref = gcu.ptr2memref(y, memref_type)
    out_memref = gcu.ptr2memref(output, memref_type)

    # Compute index for maskedload/store
    offset_idx = arith.index_cast(idx_ty, global_offset)

    # Mask: all 128 lanes active
    mask_full = gcu.constant_mask([128], vec128xi1)

    # vector.maskedload/maskedstore (standard MLIR vector ops on memref)
    x_vec = gcu.maskedload(x_memref, offset_idx, mask_full, zeros, vec128xf32)
    y_vec = gcu.maskedload(y_memref, offset_idx, mask_full, zeros, vec128xf32)
    out_vec = arith.addf(x_vec, y_vec)
    gcu.maskedstore(out_memref, offset_idx, mask_full, out_vec)


if __name__ == "__main__":
    print("=== Testing use_gcu_opt=True with gcu.ptr2memref ===")
    print()

    print("--- Step 1: EDSL MLIR module (before lowering) ---")
    print(edsl_ptr2memref.mlir_module)
    print()

    print("--- Step 2: Calling gcu-compiler-opt for LLVM lowering ---")
    try:
        llvm_ir = edsl_ptr2memref.make_llvm()
        print("SUCCESS! LLVM IR output (first 80 lines):")
        lines = llvm_ir.splitlines()
        for line in lines[:80]:
            print(line)
        if len(lines) > 80:
            print(f"... ({len(lines) - 80} more lines)")
        print()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

// RUN: triton-opt %s --triton-to-unstructure --verify-each | FileCheck %s --check-prefix=T2U
// RUN: triton-opt %s --triton-to-unstructure --bubble-up-operation --triton-to-linalg --verify-each | FileCheck %s --check-prefix=LOWER
// RUN: triton-opt %s --triton-to-unstructure='compile-on-910-95=true compile-mode=simd_simt_template' --bubble-up-operation --triton-to-linalg='compile-on-910-95=true compile-mode=simd_simt_template' --verify-each | FileCheck %s --check-prefix=LOWER
// RUN: sed 's/%mask, %other/%mask/' %s | triton-opt --triton-to-unstructure --bubble-up-operation --triton-to-linalg --verify-each | FileCheck %s --check-prefix=LOWER
// RUN: sed 's/cmpi slt, %x/cmpi eq, %x/' %s | triton-opt --triton-to-unstructure --verify-each | FileCheck %s --check-prefix=KEEP
// RUN: sed 's/cmpi slt, %x/cmpi sge, %x/' %s | triton-opt --triton-to-unstructure --verify-each | FileCheck %s --check-prefix=KEEP

// The row addresses are nonstructured, but columns have a fixed stride. Both
// mask bounds must guard memory access, even when the result is selected later.
// Runtime lower/upper bounds also cover nonzero starts and empty intersections.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // T2U-LABEL: tt.func public @mixed_axis_load_mask(
  // T2U: %[[COLMASK:.*]] = tt.broadcast %{{.*}} : tensor<1x32xi1> -> tensor<3x32xi1>
  // T2U: %[[FULLMASK:.*]] = arith.andi
  // T2U: scf.for
  // T2U: %[[MASK:.*]] = tensor.extract_slice %[[COLMASK]][%{{.*}}, 0] [1, 32] [1, 1] {DiscreteMemAccess} : tensor<3x32xi1> to tensor<1x32xi1>
  // T2U: tt.load %{{[^,]+}}, %[[MASK]] {DiscreteMemAccess} : tensor<1x32x!tt.ptr<f32>>
  // T2U: arith.select %[[FULLMASK]],
  // KEEP-LABEL: tt.func public @mixed_axis_load_mask(
  // KEEP: %[[FULLMASK:.*]] = arith.andi %{{.*}}, %{{.*}} : tensor<3x32xi1>
  // KEEP: scf.for
  // KEEP: %[[MASK:.*]] = tensor.extract_slice %[[FULLMASK]][%{{.*}}, 0] [1, 32] [1, 1] {DiscreteMemAccess} : tensor<3x32xi1> to tensor<1x32xi1>
  // KEEP: tt.load %{{[^,]+}}, %[[MASK]] {DiscreteMemAccess} : tensor<1x32x!tt.ptr<f32>>
  // LOWER-LABEL: func.func @mixed_axis_load_mask(
  // LOWER: scf.for
  // LOWER: %[[BASE:.*]] = memref.reinterpret_cast
  // LOWER: %[[SOURCE:.*]] = memref.subview %[[BASE]]
  // LOWER: memref.copy %[[SOURCE]], %{{.*}} : memref<1x?xf32,
  // LOWER: return
  tt.func public @mixed_axis_load_mask(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>, %rows: i32, %lower: i32, %upper: i32) {
    %x = tt.make_range {start = 0 : i32, end = 3 : i32} : tensor<3xi32>
    %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %two = arith.constant dense<2> : tensor<3xi32>
    %stride = arith.constant dense<16> : tensor<32xi32>
    %row = arith.remsi %x, %two : tensor<3xi32>
    %col = arith.muli %r, %stride : tensor<32xi32>
    %row2d = tt.expand_dims %row {axis = 1 : i32} : tensor<3xi32> -> tensor<3x1xi32>
    %col2d = tt.expand_dims %col {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %rowb = tt.broadcast %row2d : tensor<3x1xi32> -> tensor<3x32xi32>
    %colb = tt.broadcast %col2d : tensor<1x32xi32> -> tensor<3x32xi32>
    %offset = arith.addi %rowb, %colb : tensor<3x32xi32>
    %base = tt.splat %input : !tt.ptr<f32> -> tensor<3x32x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %offset : tensor<3x32x!tt.ptr<f32>>, tensor<3x32xi32>
    %rowsb = tt.splat %rows : i32 -> tensor<3xi32>
    %xm = arith.cmpi slt, %x, %rowsb : tensor<3xi32>
    %lo = tt.splat %lower : i32 -> tensor<32xi32>
    %hi = tt.splat %upper : i32 -> tensor<32xi32>
    %rmlo = arith.cmpi sge, %r, %lo : tensor<32xi32>
    %rmhi = arith.cmpi slt, %r, %hi : tensor<32xi32>
    %rm = arith.andi %rmlo, %rmhi : tensor<32xi1>
    %xm2d = tt.expand_dims %xm {axis = 1 : i32} : tensor<3xi1> -> tensor<3x1xi1>
    %rm2d = tt.expand_dims %rm {axis = 0 : i32} : tensor<32xi1> -> tensor<1x32xi1>
    %xmb = tt.broadcast %xm2d : tensor<3x1xi1> -> tensor<3x32xi1>
    %rmb = tt.broadcast %rm2d : tensor<1x32xi1> -> tensor<3x32xi1>
    %mask = arith.andi %xmb, %rmb : tensor<3x32xi1>
    %other = arith.constant dense<-7.0> : tensor<3x32xf32>
    %value = tt.load %ptr, %mask, %other : tensor<3x32x!tt.ptr<f32>>
    %outstride = arith.constant dense<32> : tensor<3xi32>
    %outrow = arith.muli %x, %outstride : tensor<3xi32>
    %outrow2d = tt.expand_dims %outrow {axis = 1 : i32} : tensor<3xi32> -> tensor<3x1xi32>
    %r2d = tt.expand_dims %r {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %outrowb = tt.broadcast %outrow2d : tensor<3x1xi32> -> tensor<3x32xi32>
    %rb = tt.broadcast %r2d : tensor<1x32xi32> -> tensor<3x32xi32>
    %outoffset = arith.addi %outrowb, %rb : tensor<3x32xi32>
    %outbase = tt.splat %output : !tt.ptr<f32> -> tensor<3x32x!tt.ptr<f32>>
    %outptr = tt.addptr %outbase, %outoffset : tensor<3x32x!tt.ptr<f32>>, tensor<3x32xi32>
    tt.store %outptr, %value : tensor<3x32x!tt.ptr<f32>>
    tt.return
  }
}

// RUN: triton-opt %s --triton-to-unstructure='compile-on-910-95=true compile-mode=simd_simt_template' \
// RUN:                --triton-to-linalg='compile-on-910-95=true compile-mode=simd_simt_template' --split-input-file \
// RUN: | FileCheck %s
// RUN: triton-opt %s --mlir-print-ir-after-all \
// RUN:                --triton-to-unstructure='compile-on-910-95=true compile-mode=simd_simt_template' \
// RUN:                --triton-to-linalg='compile-on-910-95=true compile-mode=simd_simt_template' --split-input-file 2>&1 \
// RUN: | FileCheck %s --check-prefix=SLS-INSPECT

// The static stride-4 AddPtr below is deliberately inspected but cannot be
// converted to an indirect load.  The trace locks down the original greedy
// protocol: mark it after the Chunk/SLS stage, retain the mark through T2L's
// in-place CSE/Canonicalizer cleanup, then lower it through memref.copy.
// SLS-INSPECT: IR Dump After {{.*}}LayoutMemoryCompatibilityPass{{.*}}
// SLS-INSPECT: tt.func public @addptr_stride4_1d
// SLS-INSPECT: tt.load {{.*}} {InspectedByStridedLoadStoreRewrite}
// SLS-INSPECT: IR Dump After CSE
// SLS-INSPECT: tt.func public @addptr_stride4_1d
// SLS-INSPECT: tt.load {{.*}} {InspectedByStridedLoadStoreRewrite}
// SLS-INSPECT: IR Dump After Canonicalizer
// SLS-INSPECT: tt.func public @addptr_stride4_1d
// SLS-INSPECT: tt.load {{.*}} {InspectedByStridedLoadStoreRewrite}
// SLS-INSPECT: IR Dump After TritonToLinalg
// SLS-INSPECT: func.func @addptr_stride4_1d
// SLS-INSPECT-NOT: call @triton_indirect_load
// SLS-INSPECT: memref.copy

// The masked stride-4 variant follows the indirect branch.  Keep this trace
// separate from the inspect/no-rewrite case above so both greedy outcomes are
// covered at the Graph compatibility boundary.
// SLS-INSPECT: IR Dump After {{.*}}LayoutMemoryCompatibilityPass{{.*}}
// SLS-INSPECT: tt.func public @addptr_stride4_masked_single_tile
// SLS-INSPECT: ascend.indirect_load {{.*}} {RewrittenByStridedLoadStoreRewrite, isVolatile = false}
// SLS-INSPECT: IR Dump After TritonToLinalg
// SLS-INSPECT: func.func @addptr_stride4_masked_single_tile
// SLS-INSPECT: call @triton_indirect_load

// -----
// V1 rank-1 miss (AddPtr, static power-of-two stride 4):
// tt.addptr(splat ptr, arange*4) -> should stay on the reinterpret_cast/memref.copy path.
// CHECK-LABEL: func.func @addptr_stride4_1d
// CHECK-NOT: call @triton_indirect_load
// CHECK: memref.copy
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @addptr_stride4_1d(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                    %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c4 = arith.constant dense<4> : tensor<256xi32>
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %1 = arith.muli %0, %c4 : tensor<256xi32>
    %2 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %4 = tt.load %3 : tensor<256x!tt.ptr<f32>>
    %5 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %6 = tt.addptr %5, %0 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %6, %4 : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// V1 rank-1 hit (AddPtr, static power-of-two stride 4, masked single tile):
// The structured strided-copy route would create a dynamic boundary size that
// can become zero for over-launched programs. Route to indirect_load instead.
// CHECK-LABEL: func.func @addptr_stride4_masked_single_tile
// CHECK-SAME: parallel_mode = "mix_simd_simt"
// CHECK: call @triton_indirect_load(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {isVolatile = false} : (memref<?xf16>, tensor<1024xi64>, tensor<1024xi1>, tensor<1024xf16>) -> tensor<1024xf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @addptr_stride4_masked_single_tile(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                                    %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %zero = arith.constant dense<0.000000e+00> : tensor<1024xf16>
    %one = arith.constant dense<1.000000e+00> : tensor<1024xf16>
    %c4 = arith.constant dense<4> : tensor<1024xi32>
    %bound = arith.constant dense<1024> : tensor<1024xi32>
    %c1024_i32 = arith.constant 1024 : i32
    %pid = tt.get_program_id x : i32
    %tile_base = arith.muli %pid, %c1024_i32 : i32
    %range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %base_splat = tt.splat %tile_base : i32 -> tensor<1024xi32>
    %idx = arith.addi %base_splat, %range : tensor<1024xi32>
    %mask = arith.cmpi slt, %idx, %bound : tensor<1024xi32>
    %src_offsets = arith.muli %idx, %c4 : tensor<1024xi32>
    %src_base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
    %src_ptr = tt.addptr %src_base, %src_offsets : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    %val = tt.load %src_ptr, %mask, %zero : tensor<1024x!tt.ptr<f16>>
    %out = arith.addf %val, %one : tensor<1024xf16>
    %dst_base = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
    %dst_ptr = tt.addptr %dst_base, %idx : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    tt.store %dst_ptr, %out, %mask : tensor<1024x!tt.ptr<f16>>
    tt.return
  }
}

// -----
// V1 guard (AddPtr, static power-of-two stride 4, masked single tile):
// This otherwise indirect-eligible load was already handled by
// ImplicitPermute.  SLS must leave it on the normal memref.copy lowering path.
// CHECK-LABEL: func.func @addptr_stride4_masked_implicit_permute_handled
// CHECK-NOT: call @triton_indirect_load
// CHECK: memref.copy
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @addptr_stride4_masked_implicit_permute_handled(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                                                 %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %zero = arith.constant dense<0.000000e+00> : tensor<1024xf16>
    %one = arith.constant dense<1.000000e+00> : tensor<1024xf16>
    %c4 = arith.constant dense<4> : tensor<1024xi32>
    %bound = arith.constant dense<1024> : tensor<1024xi32>
    %c1024_i32 = arith.constant 1024 : i32
    %pid = tt.get_program_id x : i32
    %tile_base = arith.muli %pid, %c1024_i32 : i32
    %range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %base_splat = tt.splat %tile_base : i32 -> tensor<1024xi32>
    %idx = arith.addi %base_splat, %range : tensor<1024xi32>
    %mask = arith.cmpi slt, %idx, %bound : tensor<1024xi32>
    %src_offsets = arith.muli %idx, %c4 : tensor<1024xi32>
    %src_base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
    %src_ptr = tt.addptr %src_base, %src_offsets : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    %val = tt.load %src_ptr, %mask, %zero {ImplicitPermuteHandled} : tensor<1024x!tt.ptr<f16>>
    %out = arith.addf %val, %one : tensor<1024xf16>
    %dst_base = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
    %dst_ptr = tt.addptr %dst_base, %idx : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    tt.store %dst_ptr, %out, %mask : tensor<1024x!tt.ptr<f16>>
    tt.return
  }
}

// -----
// V1 rank-1 hit (AddPtr, static non-power-of-two stride 3):
// tt.addptr(splat ptr, arange*3) -> should become triton_stride_load call.
// CHECK-LABEL: func.func @addptr_stride3_1d
// CHECK-NOT: call @triton_indirect_load
// CHECK: call @triton_stride_load
// CHECK-SAME: : (memref<?xf32>, i64, f32, i64, i64) -> tensor<256xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @addptr_stride3_1d(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                    %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c3 = arith.constant dense<3> : tensor<256xi32>
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %1 = arith.muli %0, %c3 : tensor<256xi32>
    %2 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %4 = tt.load %3 : tensor<256x!tt.ptr<f32>>
    %5 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %6 = tt.addptr %5, %0 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %6, %4 : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// V1 rank-1 miss (AddPtr, dynamic stride):
// Runtime stride may be 1 or power-of-two, so keep the structured SIMD path.
// CHECK-LABEL: func.func @addptr_dynamic_stride_1d
// CHECK-NOT: call @triton_indirect_load
// CHECK-NOT: call @triton_stride_load
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @addptr_dynamic_stride_1d(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                           %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                           %stride: i32) {
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %stride_splat = tt.splat %stride : i32 -> tensor<256xi32>
    %1 = arith.muli %0, %stride_splat : tensor<256xi32>
    %2 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %4 = tt.load %3 : tensor<256x!tt.ptr<f32>>
    %5 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %6 = tt.addptr %5, %0 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %6, %4 : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// V2 rank-1 miss (AddPtr store, scalar base AddPtr + dynamic stride):
// Runtime stride may be 1 or power-of-two, so keep the structured SIMD path.
// CHECK-LABEL: func.func @addptr_dynamic_stride_store_scalar_base
// CHECK-NOT: call @triton_indirect_store
// CHECK-NOT: call @triton_stride_store
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @addptr_dynamic_stride_store_scalar_base(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                          %base_offset: i64,
                                                          %stride: i64) {
    %cst = arith.constant dense<1.000000e+00> : tensor<32xf32>
    %range_i32 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %range_i64 = arith.extsi %range_i32 : tensor<32xi32> to tensor<32xi64>
    %stride_splat = tt.splat %stride : i64 -> tensor<32xi64>
    %offsets = arith.muli %range_i64, %stride_splat : tensor<32xi64>
    %scalar_base = tt.addptr %arg0, %base_offset : !tt.ptr<f32>, i64
    %base = tt.splat %scalar_base : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %offsets : tensor<32x!tt.ptr<f32>>, tensor<32xi64>
    tt.store %ptr, %cst : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// V1 rank-2 hit (AddPtr, static non-power-of-two stride 3):
// CHECK-LABEL: func.func @addptr_stride3_2d
// CHECK-NOT: call @triton_indirect_load
// CHECK: call @triton_stride_load
// CHECK-SAME: : (memref<?xf32>, i64, f32, i64, i64, i64, i64) -> tensor<4x8xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @addptr_stride3_2d(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                    %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %m = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %m_exp = tt.expand_dims %m {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %m_b = tt.broadcast %m_exp : tensor<4x1xi32> -> tensor<4x8xi32>
    %n = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %n_exp = tt.expand_dims %n {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %n_b = tt.broadcast %n_exp : tensor<1x8xi32> -> tensor<4x8xi32>
    %c96 = arith.constant dense<96> : tensor<4x8xi32>
    %c3 = arith.constant dense<3> : tensor<4x8xi32>
    %row = arith.muli %m_b, %c96 : tensor<4x8xi32>
    %col = arith.muli %n_b, %c3 : tensor<4x8xi32>
    %offset = arith.addi %row, %col : tensor<4x8xi32>
    %src = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x8x!tt.ptr<f32>>
    %ptr = tt.addptr %src, %offset : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
    %val = tt.load %ptr : tensor<4x8x!tt.ptr<f32>>
    %c8 = arith.constant dense<8> : tensor<4x8xi32>
    %out_row = arith.muli %m_b, %c8 : tensor<4x8xi32>
    %out_offset = arith.addi %out_row, %n_b : tensor<4x8xi32>
    %dst = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x8x!tt.ptr<f32>>
    %out_ptr = tt.addptr %dst, %out_offset : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
    tt.store %out_ptr, %val : tensor<4x8x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// V1 miss (AddPtr, 1D, stride == 1):
// arange + splat (no muli > 1) -> NOT rewritten, normal strided memref.copy
// CHECK-LABEL: func.func @addptr_stride1_1d
// CHECK-NOT: call @triton_indirect_load
// CHECK: memref.copy
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @addptr_stride1_1d(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                    %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %3 = tt.load %2 : tensor<256x!tt.ptr<f32>>
    %4 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %5 = tt.addptr %4, %0 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %5, %3 : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// V3 hit (make_tensor_ptr, rank-2): non-power-of-two low-dim stride uses
// triton_stride_load.
// CHECK-LABEL: func.func @mtpt_low_stride5
// CHECK-NOT: call @triton_indirect_load
// CHECK: call @triton_stride_load
// CHECK-SAME: : (memref<?xf32>, i64, f32, i64, i64, i64, i64) -> tensor<4x8xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @mtpt_low_stride5(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                   %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c0 = arith.constant 0 : i32
    %size_m = arith.constant 64 : i64
    %size_n = arith.constant 256 : i64
    %stride_m = arith.constant 256 : i64
    %stride_n = arith.constant 5 : i64
    %0 = tt.make_tensor_ptr %arg0, [%size_m, %size_n], [%stride_m, %stride_n], [%c0, %c0]
         {order = array<i32: 1, 0>} : <tensor<4x8xf32>>
    %1 = tt.load %0 : !tt.ptr<tensor<4x8xf32>>
    %sz1 = arith.constant 1 : i64
    %sz_out_m = arith.constant 4 : i64
    %sz_out_n = arith.constant 8 : i64
    %2 = tt.make_tensor_ptr %arg1, [%sz_out_m, %sz_out_n], [%sz_out_n, %sz1], [%c0, %c0]
         {order = array<i32: 1, 0>} : <tensor<4x8xf32>>
    tt.store %2, %1 : !tt.ptr<tensor<4x8xf32>>
    tt.return
  }
}

// -----
// V3 hit (make_tensor_ptr, rank-3): non-power-of-two low-dim stride uses
// triton_stride_load with one base offset, 3 strides and 3 numels.
// CHECK-LABEL: func.func @mtpt_3d_low_stride3
// CHECK-NOT: call @triton_indirect_load
// CHECK: call @triton_stride_load
// CHECK-SAME: : (memref<?xf32>, i64, f32, i64, i64, i64, i64, i64, i64) -> tensor<2x4x8xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @mtpt_3d_low_stride3(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                      %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c0 = arith.constant 0 : i32
    %size_b = arith.constant 16 : i64
    %size_m = arith.constant 64 : i64
    %size_n = arith.constant 256 : i64
    %stride_b = arith.constant 16384 : i64
    %stride_m = arith.constant 256 : i64
    %stride_n = arith.constant 3 : i64
    %0 = tt.make_tensor_ptr %arg0, [%size_b, %size_m, %size_n], [%stride_b, %stride_m, %stride_n], [%c0, %c0, %c0]
         {order = array<i32: 2, 1, 0>} : <tensor<2x4x8xf32>>
    %1 = tt.load %0 : !tt.ptr<tensor<2x4x8xf32>>
    %sz_b = arith.constant 2 : i64
    %sz_m = arith.constant 4 : i64
    %sz_n = arith.constant 8 : i64
    %out_stride_b = arith.constant 32 : i64
    %out_stride_m = arith.constant 8 : i64
    %out_stride_n = arith.constant 1 : i64
    %2 = tt.make_tensor_ptr %arg1, [%sz_b, %sz_m, %sz_n], [%out_stride_b, %out_stride_m, %out_stride_n], [%c0, %c0, %c0]
         {order = array<i32: 2, 1, 0>} : <tensor<2x4x8xf32>>
    tt.store %2, %1 : !tt.ptr<tensor<2x4x8xf32>>
    tt.return
  }
}

// -----
// V1 rank-4 fallback (make_tensor_ptr): stride_load only supports 1-3D, so
// rank-4 static non-power-of-two stride still uses triton_indirect_load.
// CHECK-LABEL: func.func @mtpt_4d_low_stride3_indirect
// CHECK-NOT: call @triton_stride_load
// CHECK: call @triton_indirect_load
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @mtpt_4d_low_stride3_indirect(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                               %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c0 = arith.constant 0 : i32
    %size_b = arith.constant 8 : i64
    %size_h = arith.constant 16 : i64
    %size_m = arith.constant 64 : i64
    %size_n = arith.constant 256 : i64
    %stride_b = arith.constant 262144 : i64
    %stride_h = arith.constant 16384 : i64
    %stride_m = arith.constant 256 : i64
    %stride_n = arith.constant 3 : i64
    %0 = tt.make_tensor_ptr %arg0, [%size_b, %size_h, %size_m, %size_n], [%stride_b, %stride_h, %stride_m, %stride_n], [%c0, %c0, %c0, %c0]
         {order = array<i32: 3, 2, 1, 0>} : <tensor<2x3x4x8xf32>>
    %1 = tt.load %0 : !tt.ptr<tensor<2x3x4x8xf32>>
    %sz_b = arith.constant 2 : i64
    %sz_h = arith.constant 3 : i64
    %sz_m = arith.constant 4 : i64
    %sz_n = arith.constant 8 : i64
    %out_stride_b = arith.constant 96 : i64
    %out_stride_h = arith.constant 32 : i64
    %out_stride_m = arith.constant 8 : i64
    %out_stride_n = arith.constant 1 : i64
    %2 = tt.make_tensor_ptr %arg1, [%sz_b, %sz_h, %sz_m, %sz_n], [%out_stride_b, %out_stride_h, %out_stride_m, %out_stride_n], [%c0, %c0, %c0, %c0]
         {order = array<i32: 3, 2, 1, 0>} : <tensor<2x3x4x8xf32>>
    tt.store %2, %1 : !tt.ptr<tensor<2x3x4x8xf32>>
    tt.return
  }
}

// -----
// V1 miss (make_tensor_ptr, non-permuted, low-dim stride 1):
// strides=[8, 1] = contiguous -> last-dim stride 1, V1 not hit
// CHECK-LABEL: func.func @mtpt_low_stride1
// CHECK-NOT: call @triton_indirect_load
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @mtpt_low_stride1(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                   %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c0 = arith.constant 0 : i32
    %size_m = arith.constant 4 : i64
    %size_n = arith.constant 8 : i64
    %stride_m = arith.constant 8 : i64
    %stride_n = arith.constant 1 : i64
    %0 = tt.make_tensor_ptr %arg0, [%size_m, %size_n], [%stride_m, %stride_n], [%c0, %c0]
         {order = array<i32: 1, 0>} : <tensor<4x8xf32>>
    %1 = tt.load %0 : !tt.ptr<tensor<4x8xf32>>
    %2 = tt.make_tensor_ptr %arg1, [%size_m, %size_n], [%stride_m, %stride_n], [%c0, %c0]
         {order = array<i32: 1, 0>} : <tensor<4x8xf32>>
    tt.store %2, %1 : !tt.ptr<tensor<4x8xf32>>
    tt.return
  }
}

// -----
// V1 rank-1 miss (make_tensor_ptr, static power-of-two stride 8):
// CHECK-LABEL: func.func @mtpt_1d_stride8_pow2
// CHECK-NOT: call @triton_indirect_load
// CHECK: memref.copy
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @mtpt_1d_stride8_pow2(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                       %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c0 = arith.constant 0 : i32
    %t = arith.constant 2048 : i64
    %stride = arith.constant 8 : i64
    %0 = tt.make_tensor_ptr %arg0, [%t], [%stride], [%c0]
         {order = array<i32: 0>} : <tensor<128xf32>>
    %1 = tt.load %0 : !tt.ptr<tensor<128xf32>>
    %sz1 = arith.constant 1 : i64
    %2 = tt.make_tensor_ptr %arg1, [%t], [%sz1], [%c0]
         {order = array<i32: 0>} : <tensor<128xf32>>
    tt.store %2, %1 : !tt.ptr<tensor<128xf32>>
    tt.return
  }
}

// -----
// V1 rank-1 hit (make_tensor_ptr, static non-power-of-two stride 7):
// CHECK-LABEL: func.func @mtpt_1d_stride7
// CHECK-NOT: call @triton_indirect_load
// CHECK: call @triton_stride_load
// CHECK-SAME: : (memref<?xf32>, i64, f32, i64, i64) -> tensor<128xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @mtpt_1d_stride7(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                  %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c0 = arith.constant 0 : i32
    %t = arith.constant 2048 : i64
    %stride = arith.constant 7 : i64
    %0 = tt.make_tensor_ptr %arg0, [%t], [%stride], [%c0]
         {order = array<i32: 0>} : <tensor<128xf32>>
    %1 = tt.load %0 : !tt.ptr<tensor<128xf32>>
    %sz1 = arith.constant 1 : i64
    %2 = tt.make_tensor_ptr %arg1, [%t], [%sz1], [%c0]
         {order = array<i32: 0>} : <tensor<128xf32>>
    tt.store %2, %1 : !tt.ptr<tensor<128xf32>>
    tt.return
  }
}

// -----
// V1 rank-1 miss (make_tensor_ptr, dynamic stride):
// Runtime stride may be 1 or power-of-two, so keep the structured SIMD path.
// CHECK-LABEL: func.func @mtpt_1d_dynamic_stride
// CHECK-NOT: call @triton_indirect_load
// CHECK-NOT: call @triton_stride_load
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @mtpt_1d_dynamic_stride(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                         %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                         %stride: i64) {
    %c0 = arith.constant 0 : i32
    %t = arith.constant 2048 : i64
    %0 = tt.make_tensor_ptr %arg0, [%t], [%stride], [%c0]
         {order = array<i32: 0>} : <tensor<128xf32>>
    %1 = tt.load %0 : !tt.ptr<tensor<128xf32>>
    %sz1 = arith.constant 1 : i64
    %2 = tt.make_tensor_ptr %arg1, [%t], [%sz1], [%c0]
         {order = array<i32: 0>} : <tensor<128xf32>>
    tt.store %2, %1 : !tt.ptr<tensor<128xf32>>
    tt.return
  }
}

// -----
// V1 miss (make_tensor_ptr, permuted - ImplicitPermute should handle, V1 must stay out):
// strides=[1, 8], order=[0,1] -> permuted; ImplicitPermute rewrites with tt.trans,
// V1 sees ImplicitPermuteHandledTAG and bails.
// CHECK-LABEL: func.func @mtpt_permuted
// CHECK-NOT: call @triton_indirect_load
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @mtpt_permuted(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c0 = arith.constant 0 : i32
    %size_m = arith.constant 4 : i64
    %size_n = arith.constant 8 : i64
    %stride_m = arith.constant 1 : i64
    %stride_n = arith.constant 8 : i64
    %0 = tt.make_tensor_ptr %arg0, [%size_m, %size_n], [%stride_m, %stride_n], [%c0, %c0]
         {order = array<i32: 0, 1>} : <tensor<4x8xf32>>
    %1 = tt.load %0 : !tt.ptr<tensor<4x8xf32>>
    %out_stride_m = arith.constant 8 : i64
    %out_stride_n = arith.constant 1 : i64
    %2 = tt.make_tensor_ptr %arg1, [%size_m, %size_n], [%out_stride_m, %out_stride_n], [%c0, %c0]
         {order = array<i32: 1, 0>} : <tensor<4x8xf32>>
    tt.store %2, %1 : !tt.ptr<tensor<4x8xf32>>
    tt.return
  }
}

// -----
// V2 hit (AddPtr Store, 1D, static non-power-of-two stride 3):
// tt.store(tt.addptr(splat ptr, arange*3), value) -> tt.stride_store -> call @triton_stride_store
// CHECK-LABEL: func.func @addptr_store_stride3_1d
// CHECK-NOT: call @triton_indirect_store
// CHECK: call @triton_stride_store(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (memref<?xf32>, tensor<256xf32>, i64, i64, i64) -> ()
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @addptr_store_stride3_1d(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                          %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c3 = arith.constant dense<3> : tensor<256xi32>
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %1 = arith.muli %0, %c3 : tensor<256xi32>
    %2 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %3 = tt.addptr %2, %0 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %4 = tt.load %3 : tensor<256x!tt.ptr<f32>>
    %5 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %6 = tt.addptr %5, %1 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %6, %4 : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// V2 miss (AddPtr Store, stride == 1):
// CHECK-LABEL: func.func @addptr_store_stride1_1d
// CHECK-NOT: call @triton_indirect_store
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @addptr_store_stride1_1d(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                          %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %3 = tt.load %2 : tensor<256x!tt.ptr<f32>>
    %4 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %5 = tt.addptr %4, %0 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %5, %3 : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// V2 hit (make_tensor_ptr Store, non-permuted, low-dim non-power-of-two stride 5):
// CHECK-LABEL: func.func @mtpt_store_low_stride5
// CHECK-NOT: call @triton_indirect_store
// CHECK: call @triton_stride_store(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (memref<?xf32>, tensor<4x8xf32>, i64, i64, i64, i64, i64) -> ()
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @mtpt_store_low_stride5(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                         %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c0 = arith.constant 0 : i32
    %size_m = arith.constant 64 : i64
    %size_n = arith.constant 256 : i64
    %stride_m = arith.constant 256 : i64
    %stride_n = arith.constant 5 : i64
    %sz1 = arith.constant 1 : i64
    %sz_in_m = arith.constant 4 : i64
    %sz_in_n = arith.constant 8 : i64
    %0 = tt.make_tensor_ptr %arg0, [%sz_in_m, %sz_in_n], [%sz_in_n, %sz1], [%c0, %c0]
         {order = array<i32: 1, 0>} : <tensor<4x8xf32>>
    %1 = tt.load %0 : !tt.ptr<tensor<4x8xf32>>
    %2 = tt.make_tensor_ptr %arg1, [%size_m, %size_n], [%stride_m, %stride_n], [%c0, %c0]
         {order = array<i32: 1, 0>} : <tensor<4x8xf32>>
    tt.store %2, %1 : !tt.ptr<tensor<4x8xf32>>
    tt.return
  }
}

// -----
// V2 rank-4 fallback (make_tensor_ptr Store): stride_store only supports 1-3D,
// so rank-4 static non-power-of-two stride still uses triton_indirect_store.
// CHECK-LABEL: func.func @mtpt_store_4d_low_stride5_indirect
// CHECK-NOT: call @triton_stride_store
// CHECK: call @triton_indirect_store
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @mtpt_store_4d_low_stride5_indirect(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                     %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c0 = arith.constant 0 : i32
    %size_b = arith.constant 8 : i64
    %size_h = arith.constant 16 : i64
    %size_m = arith.constant 64 : i64
    %size_n = arith.constant 256 : i64
    %stride_b = arith.constant 327680 : i64
    %stride_h = arith.constant 20480 : i64
    %stride_m = arith.constant 256 : i64
    %stride_n = arith.constant 5 : i64
    %sz_b = arith.constant 2 : i64
    %sz_h = arith.constant 3 : i64
    %sz_m = arith.constant 4 : i64
    %sz_n = arith.constant 8 : i64
    %in_stride_b = arith.constant 96 : i64
    %in_stride_h = arith.constant 32 : i64
    %in_stride_m = arith.constant 8 : i64
    %in_stride_n = arith.constant 1 : i64
    %0 = tt.make_tensor_ptr %arg0, [%sz_b, %sz_h, %sz_m, %sz_n], [%in_stride_b, %in_stride_h, %in_stride_m, %in_stride_n], [%c0, %c0, %c0, %c0]
         {order = array<i32: 3, 2, 1, 0>} : <tensor<2x3x4x8xf32>>
    %1 = tt.load %0 : !tt.ptr<tensor<2x3x4x8xf32>>
    %2 = tt.make_tensor_ptr %arg1, [%size_b, %size_h, %size_m, %size_n], [%stride_b, %stride_h, %stride_m, %stride_n], [%c0, %c0, %c0, %c0]
         {order = array<i32: 3, 2, 1, 0>} : <tensor<2x3x4x8xf32>>
    tt.store %2, %1 : !tt.ptr<tensor<2x3x4x8xf32>>
    tt.return
  }
}

// -----
// V2 miss (make_tensor_ptr Store, dynamic stride):
// Runtime stride may be 1 or power-of-two, so keep the structured SIMD path.
// CHECK-LABEL: func.func @mtpt_store_dynamic_stride
// CHECK-NOT: call @triton_indirect_store
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @mtpt_store_dynamic_stride(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                            %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                            %stride: i64) {
    %c0 = arith.constant 0 : i32
    %sz1 = arith.constant 1 : i64
    %t = arith.constant 2048 : i64
    %0 = tt.make_tensor_ptr %arg0, [%t], [%sz1], [%c0]
         {order = array<i32: 0>} : <tensor<128xf32>>
    %1 = tt.load %0 : !tt.ptr<tensor<128xf32>>
    %2 = tt.make_tensor_ptr %arg1, [%t], [%stride], [%c0]
         {order = array<i32: 0>} : <tensor<128xf32>>
    tt.store %2, %1 : !tt.ptr<tensor<128xf32>>
    tt.return
  }
}

// -----
// V2 miss (make_tensor_ptr Store, contiguous):
// CHECK-LABEL: func.func @mtpt_store_low_stride1
// CHECK-NOT: call @triton_indirect_store
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @mtpt_store_low_stride1(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                         %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c0 = arith.constant 0 : i32
    %size_m = arith.constant 4 : i64
    %size_n = arith.constant 8 : i64
    %stride_m = arith.constant 8 : i64
    %stride_n = arith.constant 1 : i64
    %0 = tt.make_tensor_ptr %arg0, [%size_m, %size_n], [%stride_m, %stride_n], [%c0, %c0]
         {order = array<i32: 1, 0>} : <tensor<4x8xf32>>
    %1 = tt.load %0 : !tt.ptr<tensor<4x8xf32>>
    %2 = tt.make_tensor_ptr %arg1, [%size_m, %size_n], [%stride_m, %stride_n], [%c0, %c0]
         {order = array<i32: 1, 0>} : <tensor<4x8xf32>>
    tt.store %2, %1 : !tt.ptr<tensor<4x8xf32>>
    tt.return
  }
}

// -----
// V3 hit (make_tensor_ptr Load + boundary_check + PAD_ZERO, rank-2):
// zero padding is represented by per-dimension numel operands.
// CHECK-LABEL: func.func @mtpt_load_boundary_pad_zero
// CHECK-NOT: call @triton_indirect_load
// CHECK: call @triton_stride_load
// CHECK-SAME: : (memref<?xf32>, i64, f32, i64, i64, i64, i64) -> tensor<8x8xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @mtpt_load_boundary_pad_zero(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                              %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c0 = arith.constant 0 : i32
    %parent_m = arith.constant 7 : i64
    %parent_n = arith.constant 5 : i64
    %stride_m = arith.constant 20 : i64
    %stride_n = arith.constant 5 : i64
    %0 = tt.make_tensor_ptr %arg0, [%parent_m, %parent_n], [%stride_m, %stride_n], [%c0, %c0]
         {order = array<i32: 1, 0>} : <tensor<8x8xf32>>
    %1 = tt.load %0 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32}
         : !tt.ptr<tensor<8x8xf32>>
    %out_stride_m = arith.constant 8 : i64
    %out_stride_n = arith.constant 1 : i64
    %sz8 = arith.constant 8 : i64
    %2 = tt.make_tensor_ptr %arg1, [%sz8, %sz8], [%out_stride_m, %out_stride_n], [%c0, %c0]
         {order = array<i32: 1, 0>} : <tensor<8x8xf32>>
    tt.store %2, %1 : !tt.ptr<tensor<8x8xf32>>
    tt.return
  }
}

// -----
// V1.5 hit (make_tensor_ptr Load + boundary_check + PAD_NAN, low-dim non-power-of-two stride 5):
// CHECK-LABEL: func.func @mtpt_load_boundary_pad_nan
// CHECK-NOT: call @triton_indirect_load
// CHECK: call @triton_stride_load
// CHECK-SAME: : (memref<?xf32>, i64, f32, i64, i64, i64, i64) -> tensor<8x8xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @mtpt_load_boundary_pad_nan(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                             %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c0 = arith.constant 0 : i32
    %parent_m = arith.constant 7 : i64
    %parent_n = arith.constant 5 : i64
    %stride_m = arith.constant 20 : i64
    %stride_n = arith.constant 5 : i64
    %0 = tt.make_tensor_ptr %arg0, [%parent_m, %parent_n], [%stride_m, %stride_n], [%c0, %c0]
         {order = array<i32: 1, 0>} : <tensor<8x8xf32>>
    %1 = tt.load %0 {boundaryCheck = array<i32: 0, 1>, padding = 2 : i32}
         : !tt.ptr<tensor<8x8xf32>>
    %out_stride_m = arith.constant 8 : i64
    %out_stride_n = arith.constant 1 : i64
    %sz8 = arith.constant 8 : i64
    %2 = tt.make_tensor_ptr %arg1, [%sz8, %sz8], [%out_stride_m, %out_stride_n], [%c0, %c0]
         {order = array<i32: 1, 0>} : <tensor<8x8xf32>>
    tt.store %2, %1 : !tt.ptr<tensor<8x8xf32>>
    tt.return
  }
}

// -----
// V1.5 hit (make_tensor_ptr Load + boundary_check + SCALAR-AddPtr base):
// kernel pattern `tl.make_block_ptr(s + bos*H + i_h, ...)` -- the base is a
// chain of scalar AddPtr ops, NOT the raw function-arg ptr. V1.5 must unwrap
// the chain so the lowered call's src is `memref<?xf32>` (full underlying),
// NOT `memref<1xf32, strided<[1], offset: ?>>` (size-1 view that would make
// per-element offsets index out of bounds).
//
// Regression test for the chunk_local_cumsum precision bug fixed 2026-05-28.
// CHECK-LABEL: func.func @mtpt_load_scalar_addptr_base
// CHECK-NOT: call @triton_indirect_load
// CHECK-NOT: memref<1xf32
// CHECK: call @triton_stride_load
// CHECK-SAME: : (memref<?xf32>, i64, f32, i64, i64) -> tensor<128xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @mtpt_load_scalar_addptr_base(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                %i_b: i32, %i_h: i32, %i_t: i32) {
    // base = arg0 + i_b*T*H + i_h, where T=2048, H=7 statically
    %c0 = arith.constant 0 : i32
    %c7 = arith.constant 7 : i32
    %c2048 = arith.constant 2048 : i32
    %bos = arith.muli %i_b, %c2048 : i32
    %bosH = arith.muli %bos, %c7 : i32
    %off1 = arith.addi %bosH, %i_h : i32
    %base = tt.addptr %arg0, %off1 : !tt.ptr<f32>, i32
    %t = arith.constant 2048 : i64
    %h = arith.constant 7 : i64
    %off_t = arith.muli %i_t, %c7 : i32
    %0 = tt.make_tensor_ptr %base, [%t], [%h], [%off_t]
         {order = array<i32: 0>} : <tensor<128xf32>>
    %1 = tt.load %0 {boundaryCheck = array<i32: 0>, padding = 1 : i32}
         : !tt.ptr<tensor<128xf32>>
    %sz1 = arith.constant 1 : i64
    %2 = tt.make_tensor_ptr %arg1, [%t], [%sz1], [%off_t]
         {order = array<i32: 0>} : <tensor<128xf32>>
    tt.store %2, %1 : !tt.ptr<tensor<128xf32>>
    tt.return
  }
}

// -----
// V1.5 hit (make_tensor_ptr Store + boundary_check, low-dim non-power-of-two stride 5):
// Store boundary is represented by per-dimension numel operands.
// CHECK-LABEL: func.func @mtpt_store_boundary
// CHECK-NOT: call @triton_indirect_store
// CHECK: call @triton_stride_store(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (memref<?xf32>, tensor<8x8xf32>, i64, i64, i64, i64, i64) -> ()
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @mtpt_store_boundary(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                      %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c0 = arith.constant 0 : i32
    %sz8 = arith.constant 8 : i64
    %sz1 = arith.constant 1 : i64
    %in_stride_m = arith.constant 8 : i64
    %0 = tt.make_tensor_ptr %arg0, [%sz8, %sz8], [%in_stride_m, %sz1], [%c0, %c0]
         {order = array<i32: 1, 0>} : <tensor<8x8xf32>>
    %1 = tt.load %0 : !tt.ptr<tensor<8x8xf32>>
    %parent_m = arith.constant 7 : i64
    %parent_n = arith.constant 5 : i64
    %stride_m = arith.constant 20 : i64
    %stride_n = arith.constant 5 : i64
    %2 = tt.make_tensor_ptr %arg1, [%parent_m, %parent_n], [%stride_m, %stride_n], [%c0, %c0]
         {order = array<i32: 1, 0>} : <tensor<8x8xf32>>
    tt.store %2, %1 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x8xf32>>
    tt.return
  }
}

// -----
// CHECK-LABEL: func.func @indirect_load_without_prior_same_ptr_store
// CHECK: call @triton_indirect_load(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {isVolatile = false} : (memref<?xi32>, tensor<128xi64>, tensor<128xi1>, tensor<128xi32>) -> tensor<128xi32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @indirect_load_without_prior_same_ptr_store(%arg0: !tt.ptr<i32>,
                                                             %arg1: !tt.ptr<i32>) {
    %zero = arith.constant dense<0> : tensor<128xi32>
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %mask = arith.cmpi sge, %range, %zero : tensor<128xi32>
    %src_splat = tt.splat %arg0 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %src_ptr = tt.addptr %src_splat, %range : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %dst_splat = tt.splat %arg1 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %dst_ptr = tt.addptr %dst_splat, %range : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %value = tt.load %src_ptr, %mask, %zero {MixCompileDiscreteMask} : tensor<128x!tt.ptr<i32>>
    tt.store %dst_ptr, %value, %mask {MixCompileDiscreteMask} : tensor<128x!tt.ptr<i32>>
    tt.return
  }
}

// -----
// CHECK-LABEL: func.func @indirect_load_after_prior_same_ptr_store
// CHECK: call @triton_indirect_load(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {isVolatile = true} : (memref<?xi32>, tensor<128xi64>, tensor<128xi1>, tensor<128xi32>) -> tensor<128xi32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @indirect_load_after_prior_same_ptr_store(%arg0: !tt.ptr<i32>,
                                                           %arg1: !tt.ptr<i32>) {
    %one = arith.constant dense<1> : tensor<128xi32>
    %zero = arith.constant dense<0> : tensor<128xi32>
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %mask = arith.cmpi sge, %range, %zero : tensor<128xi32>
    %src_splat = tt.splat %arg0 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %src_ptr = tt.addptr %src_splat, %range : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %dst_splat = tt.splat %arg1 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %dst_ptr = tt.addptr %dst_splat, %range : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    tt.store %src_ptr, %one, %mask {MixCompileDiscreteMask} : tensor<128x!tt.ptr<i32>>
    %value = tt.load %src_ptr, %mask, %zero {MixCompileDiscreteMask} : tensor<128x!tt.ptr<i32>>
    tt.store %dst_ptr, %value, %mask {MixCompileDiscreteMask} : tensor<128x!tt.ptr<i32>>
    tt.return
  }
}

// -----
// CHECK-LABEL: func.func @indirect_load_unknown_root_int_to_ptr
// CHECK: call @triton_indirect_load(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {isVolatile = true} : (memref<1xi32, strided<[1]>>, tensor<128xi64>, tensor<128xi1>, tensor<128xi32>) -> tensor<128xi32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @indirect_load_unknown_root_int_to_ptr(%arg0: !tt.ptr<i32>) {
    %base_i64 = arith.constant 1024 : i64
    %base = tt.int_to_ptr %base_i64 : i64 -> !tt.ptr<i32>
    %zero = arith.constant dense<0> : tensor<128xi32>
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %mask = arith.cmpi sge, %range, %zero : tensor<128xi32>
    %src_splat = tt.splat %base : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %src_ptr = tt.addptr %src_splat, %range : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %dst_splat = tt.splat %arg0 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %dst_ptr = tt.addptr %dst_splat, %range : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %value = tt.load %src_ptr, %mask, %zero {MixCompileDiscreteMask} : tensor<128x!tt.ptr<i32>>
    tt.store %dst_ptr, %value, %mask {MixCompileDiscreteMask} : tensor<128x!tt.ptr<i32>>
    tt.return
  }
}

// -----
// CHECK-LABEL: func.func @indirect_load_store_in_preceding_scf_if
// CHECK: call @triton_indirect_load(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {isVolatile = true} : (memref<?xi32>, tensor<16xi64>, tensor<16xi1>, tensor<16xi32>) -> tensor<16xi32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @indirect_load_store_in_preceding_scf_if(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32},
                                                          %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32},
                                                          %cond1: i1,
                                                          %cond2: i1) {
    %zero = arith.constant dense<0> : tensor<16xi32>
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %c3 = arith.constant dense<3> : tensor<16xi32>
    %offsets = arith.muli %range, %c3 : tensor<16xi32>
    %src_base = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %src_ptr = tt.addptr %src_base, %offsets : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %mask = arith.cmpi sge, %range, %zero : tensor<16xi32>
    %dst_base = tt.splat %arg1 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %dst_ptr = tt.addptr %dst_base, %range : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    scf.if %cond1 {
      %one = arith.constant dense<1> : tensor<16xi32>
      tt.store %src_ptr, %one, %mask {MixCompileDiscreteMask} : tensor<16x!tt.ptr<i32>>
    }
    %value = scf.if %cond2 -> tensor<16xi32> {
      %loaded = tt.load %src_ptr, %mask, %zero {MixCompileDiscreteMask} : tensor<16x!tt.ptr<i32>>
      scf.yield %loaded : tensor<16xi32>
    } else {
      scf.yield %zero : tensor<16xi32>
    }
    tt.store %dst_ptr, %value, %mask {MixCompileDiscreteMask} : tensor<16x!tt.ptr<i32>>
    tt.return
  }
}

// -----
// CHECK-LABEL: func.func @indirect_load_no_store_in_preceding_scf_if
// CHECK: call @triton_indirect_load(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {isVolatile = false} : (memref<?xi32>, tensor<16xi64>, tensor<16xi1>, tensor<16xi32>) -> tensor<16xi32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @indirect_load_no_store_in_preceding_scf_if(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32},
                                                              %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32},
                                                              %cond1: i1,
                                                              %cond2: i1) {
    %zero = arith.constant dense<0> : tensor<16xi32>
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %c3 = arith.constant dense<3> : tensor<16xi32>
    %offsets = arith.muli %range, %c3 : tensor<16xi32>
    %src_base = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %src_ptr = tt.addptr %src_base, %offsets : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %mask = arith.cmpi sge, %range, %zero : tensor<16xi32>
    %other_base = tt.splat %arg1 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %other_ptr = tt.addptr %other_base, %range : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %dst_base = tt.splat %arg1 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %dst_ptr = tt.addptr %dst_base, %range : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    scf.if %cond1 {
      %one = arith.constant dense<1> : tensor<16xi32>
      tt.store %other_ptr, %one, %mask {MixCompileDiscreteMask} : tensor<16x!tt.ptr<i32>>
    }
    %value = scf.if %cond2 -> tensor<16xi32> {
      %loaded = tt.load %src_ptr, %mask, %zero {MixCompileDiscreteMask} : tensor<16x!tt.ptr<i32>>
      scf.yield %loaded : tensor<16xi32>
    } else {
      scf.yield %zero : tensor<16xi32>
    }
    tt.store %dst_ptr, %value, %mask {MixCompileDiscreteMask} : tensor<16x!tt.ptr<i32>>
    tt.return
  }
}

// -----
// CHECK-LABEL: func.func @indirect_load_store_in_preceding_scf_for
// CHECK: call @triton_indirect_load(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {isVolatile = true} : (memref<?xi32>, tensor<16xi64>, tensor<16xi1>, tensor<16xi32>) -> tensor<16xi32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @indirect_load_store_in_preceding_scf_for(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32},
                                                           %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32},
                                                           %trip: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %zero = arith.constant dense<0> : tensor<16xi32>
    %one = arith.constant dense<1> : tensor<16xi32>
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %src_base = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %src_ptr = tt.addptr %src_base, %range : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %dst_base = tt.splat %arg1 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %dst_ptr = tt.addptr %dst_base, %range : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %mask = arith.cmpi sge, %range, %zero : tensor<16xi32>
    scf.for %i = %c0_i32 to %trip step %c1_i32 : i32 {
      tt.store %src_ptr, %one, %mask {MixCompileDiscreteMask} : tensor<16x!tt.ptr<i32>>
    }
    scf.for %j = %c0_i32 to %trip step %c1_i32 : i32 {
      %loaded = tt.load %src_ptr, %mask, %zero {MixCompileDiscreteMask} : tensor<16x!tt.ptr<i32>>
      tt.store %dst_ptr, %loaded, %mask {MixCompileDiscreteMask} : tensor<16x!tt.ptr<i32>>
    }
    tt.return
  }
}

// -----
// CHECK-LABEL: func.func @indirect_load_nested_loop_carried_store
// CHECK: call @triton_indirect_load(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {isVolatile = true} : (memref<?xi32>, tensor<16xi64>, tensor<16xi1>, tensor<16xi32>) -> tensor<16xi32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @indirect_load_nested_loop_carried_store(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32},
                                                          %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32},
                                                          %outer: i32,
                                                          %inner: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %zero = arith.constant dense<0> : tensor<16xi32>
    %one = arith.constant dense<1> : tensor<16xi32>
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %src_base = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %src_ptr = tt.addptr %src_base, %range : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %dst_base = tt.splat %arg1 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %dst_ptr = tt.addptr %dst_base, %range : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %mask = arith.cmpi sge, %range, %zero : tensor<16xi32>
    scf.for %i = %c0_i32 to %outer step %c1_i32 : i32 {
      scf.for %j = %c0_i32 to %inner step %c1_i32 : i32 {
        %loaded = tt.load %src_ptr, %mask, %zero {MixCompileDiscreteMask} : tensor<16x!tt.ptr<i32>>
        tt.store %src_ptr, %one, %mask {MixCompileDiscreteMask} : tensor<16x!tt.ptr<i32>>
        tt.store %dst_ptr, %loaded, %mask {MixCompileDiscreteMask} : tensor<16x!tt.ptr<i32>>
      }
    }
    tt.return
  }
}

// -----
// CHECK-LABEL: func.func @indirect_load_ignores_loop_store_to_different_memref_root
// CHECK: call @triton_indirect_load_0(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {isVolatile = false} : (memref<?xf32>, tensor<16xi64>, tensor<16xi1>, tensor<16xf32>) -> tensor<16xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @indirect_load_ignores_loop_store_to_different_memref_root(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                                            %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                                            %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                                            %trip: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %zero = arith.constant dense<0> : tensor<16xi32>
    %zero_f = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %c3 = arith.constant dense<3> : tensor<16xi32>
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %mask = arith.cmpi sge, %range, %zero : tensor<16xi32>
    %gm_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %gm_ptr = tt.addptr %gm_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %src_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %src_offsets = arith.muli %range, %c3 : tensor<16xi32>
    %src_ptr = tt.addptr %src_base, %src_offsets : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %dst_base = tt.splat %arg2 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    scf.for %i = %c0_i32 to %trip step %c1_i32 : i32 {
      %local = tt.load %gm_ptr, %mask, %zero_f {MixCompileDiscreteMask} : tensor<16x!tt.ptr<f32>>
      tt.assert %mask, "mask must be true" : tensor<16xi1>
      %indirect = tt.load %src_ptr, %mask, %zero_f {MixCompileDiscreteMask} : tensor<16x!tt.ptr<f32>>
      %sum = arith.addf %local, %indirect : tensor<16xf32>
      tt.store %gm_ptr, %sum, %mask {MixCompileDiscreteMask} : tensor<16x!tt.ptr<f32>>
      tt.store %dst_ptr, %indirect, %mask {MixCompileDiscreteMask} : tensor<16x!tt.ptr<f32>>
    }
    tt.return
  }
}

// -----
// CHECK-LABEL: func.func @indirect_load_broadcast_loop_carried_different_root
// CHECK: call @triton_indirect_load{{.*}}(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {isVolatile = false} : (memref<?xf32>, tensor<4x8xi64>, tensor<4x8xi1>, tensor<4x8xf32>) -> tensor<4x8xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @indirect_load_broadcast_loop_carried_different_root(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                                      %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                                      %trip: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %row_stride = arith.constant dense<8> : tensor<4x1xi32>
    %advance = arith.constant dense<32> : tensor<4x8xi32>
    %zero = arith.constant dense<0.000000e+00> : tensor<4x8xf32>
    %mask = arith.constant dense<true> : tensor<4x8xi1>
    %rows = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %rows_2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %row_offsets = arith.muli %rows_2d, %row_stride : tensor<4x1xi32>
    %cols = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %cols_2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %col_offsets = tt.broadcast %cols_2d : tensor<1x8xi32> -> tensor<4x8xi32>
    %src_base_1d = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %src_base = tt.expand_dims %src_base_1d {axis = 1 : i32} : tensor<4x!tt.ptr<f32>> -> tensor<4x1x!tt.ptr<f32>>
    %src_rows = tt.addptr %src_base, %row_offsets : tensor<4x1x!tt.ptr<f32>>, tensor<4x1xi32>
    %src_broadcast = tt.broadcast %src_rows : tensor<4x1x!tt.ptr<f32>> -> tensor<4x8x!tt.ptr<f32>>
    %src_ptr = tt.addptr %src_broadcast, %col_offsets : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
    %dst_base_1d = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %dst_base = tt.expand_dims %dst_base_1d {axis = 1 : i32} : tensor<4x!tt.ptr<f32>> -> tensor<4x1x!tt.ptr<f32>>
    %dst_rows = tt.addptr %dst_base, %row_offsets : tensor<4x1x!tt.ptr<f32>>, tensor<4x1xi32>
    %dst_broadcast = tt.broadcast %dst_rows : tensor<4x1x!tt.ptr<f32>> -> tensor<4x8x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_broadcast, %col_offsets : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
    %result:2 = scf.for %i = %c0_i32 to %trip step %c1_i32
        iter_args(%src_iter = %src_ptr, %dst_iter = %dst_ptr)
        -> (tensor<4x8x!tt.ptr<f32>>, tensor<4x8x!tt.ptr<f32>>) : i32 {
      %value = tt.load %src_iter, %mask, %zero {MixCompileDiscreteMask} : tensor<4x8x!tt.ptr<f32>>
      tt.store %dst_iter, %value, %mask : tensor<4x8x!tt.ptr<f32>>
      %src_next = tt.addptr %src_iter, %advance : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
      %dst_next = tt.addptr %dst_iter, %advance : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
      scf.yield %src_next, %dst_next : tensor<4x8x!tt.ptr<f32>>, tensor<4x8x!tt.ptr<f32>>
    }
    tt.return
  }
}

// -----
// V1 guard (make_tensor_ptr Load): only one tt.advance layer is supported.
// A nested advance must stay on the legacy block-pointer lowering path.
// CHECK-LABEL: func.func @mtpt_nested_advance_bails
// CHECK-NOT: call @triton_indirect_load
// CHECK-NOT: call @triton_indirect_store
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @mtpt_nested_advance_bails(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                            %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %size_m = arith.constant 64 : i64
    %size_n = arith.constant 256 : i64
    %stride_m = arith.constant 256 : i64
    %stride_n = arith.constant 3 : i64
    %src = tt.make_tensor_ptr %arg0, [%size_m, %size_n], [%stride_m, %stride_n], [%c0, %c0]
           {order = array<i32: 1, 0>} : <tensor<4x8xf32>>
    %adv0 = tt.advance %src, [%c1, %c0] : !tt.ptr<tensor<4x8xf32>>
    %adv1 = tt.advance %adv0, [%c0, %c1] : !tt.ptr<tensor<4x8xf32>>
    %value = tt.load %adv1 : !tt.ptr<tensor<4x8xf32>>
    %out_stride_m = arith.constant 8 : i64
    %out_stride_n = arith.constant 1 : i64
    %dst = tt.make_tensor_ptr %arg1, [%size_m, %size_n], [%out_stride_m, %out_stride_n], [%c0, %c0]
           {order = array<i32: 1, 0>} : <tensor<4x8xf32>>
    tt.store %dst, %value : !tt.ptr<tensor<4x8xf32>>
    tt.return
  }
}

// -----
// V1/V2 rank-5 hit (make_tensor_ptr + one-level tt.advance): ranks 4-5 use
// indirect memory ops, and the advance offsets must be folded into the
// effective block-pointer offsets for both the load and store paths.
// CHECK-LABEL: func.func @mtpt_advance_5d_indirect_load_store
// CHECK: call @triton_indirect_load
// CHECK: tensor<2x2x2x2x4xi64>
// CHECK: call @triton_indirect_store
// CHECK: tensor<2x2x2x2x4xi64>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @mtpt_advance_5d_indirect_load_store(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                       %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %size_0 = arith.constant 4 : i64
    %size_1 = arith.constant 4 : i64
    %size_2 = arith.constant 4 : i64
    %size_3 = arith.constant 4 : i64
    %size_4 = arith.constant 64 : i64
    %src_stride_0 = arith.constant 4096 : i64
    %src_stride_1 = arith.constant 1024 : i64
    %src_stride_2 = arith.constant 256 : i64
    %src_stride_3 = arith.constant 64 : i64
    %src_stride_4 = arith.constant 3 : i64
    %src = tt.make_tensor_ptr %arg0, [%size_0, %size_1, %size_2, %size_3, %size_4], [%src_stride_0, %src_stride_1, %src_stride_2, %src_stride_3, %src_stride_4], [%c0, %c0, %c0, %c0, %c0]
           {order = array<i32: 4, 3, 2, 1, 0>} : <tensor<2x2x2x2x4xf32>>
    %src_adv = tt.advance %src, [%c1, %c0, %c2, %c0, %c3] : !tt.ptr<tensor<2x2x2x2x4xf32>>
    %value = tt.load %src_adv : !tt.ptr<tensor<2x2x2x2x4xf32>>
    %dst_stride_0 = arith.constant 5120 : i64
    %dst_stride_1 = arith.constant 1280 : i64
    %dst_stride_2 = arith.constant 320 : i64
    %dst_stride_3 = arith.constant 80 : i64
    %dst_stride_4 = arith.constant 5 : i64
    %dst = tt.make_tensor_ptr %arg1, [%size_0, %size_1, %size_2, %size_3, %size_4], [%dst_stride_0, %dst_stride_1, %dst_stride_2, %dst_stride_3, %dst_stride_4], [%c0, %c0, %c0, %c0, %c0]
           {order = array<i32: 4, 3, 2, 1, 0>} : <tensor<2x2x2x2x4xf32>>
    %dst_adv = tt.advance %dst, [%c0, %c3, %c0, %c1, %c2] : !tt.ptr<tensor<2x2x2x2x4xf32>>
    tt.store %dst_adv, %value : !tt.ptr<tensor<2x2x2x2x4xf32>>
    tt.return
  }
}

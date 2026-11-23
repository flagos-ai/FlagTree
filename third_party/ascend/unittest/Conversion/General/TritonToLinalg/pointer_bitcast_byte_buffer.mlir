// RUN: triton-opt --triton-to-unstructure --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s
// RUN: triton-opt --triton-to-unstructure --split-input-file %s | FileCheck %s --check-prefix=UNSTRUCTURE
// RUN: triton-opt '--triton-to-unstructure=compile-on-910-95=True compile-mode=simd_simt_template' --split-input-file %s | FileCheck %s --check-prefix=SIMT
// RUN: triton-opt '--triton-to-unstructure=compile-on-910-95=True compile-mode=simd_simt_template' '--triton-to-linalg=compile-on-910-95=True compile-mode=simd_simt_template' --split-input-file %s | FileCheck %s --check-prefix=SIMT-LINALG

// CHECK-LABEL: func.func @widen_tensor_offset
// CHECK: scf.for
// CHECK: hivm.hir.pointer_cast{{.*}} : memref<?xi32>
// CHECK: return
// UNSTRUCTURE-LABEL: tt.func public @widen_tensor_offset
// UNSTRUCTURE: arith.divsi
// UNSTRUCTURE: tt.ptr_to_int
// UNSTRUCTURE: tt.int_to_ptr
// UNSTRUCTURE-NOT: tt.bitcast
// SIMT-LABEL: tt.func public @widen_tensor_offset
// SIMT: ascend.indirect_load
// SIMT-NOT: tt.bitcast
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @widen_tensor_offset(
      %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %four = arith.constant dense<4> : tensor<8xi32>
    %offsets = arith.muli %range, %four : tensor<8xi32>
    %srcs = tt.splat %src : !tt.ptr<i8> -> tensor<8x!tt.ptr<i8>>
    %byte_ptrs = tt.addptr %srcs, %offsets : tensor<8x!tt.ptr<i8>>, tensor<8xi32>
    %i32_ptrs = tt.bitcast %byte_ptrs : tensor<8x!tt.ptr<i8>> -> tensor<8x!tt.ptr<i32>>
    %values = tt.load %i32_ptrs : tensor<8x!tt.ptr<i32>>
    %dsts = tt.splat %dst : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %dst_ptrs = tt.addptr %dsts, %range : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    tt.store %dst_ptrs, %values : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @widen_scalar_multi_addptr
// CHECK: arith.addi
// CHECK: hivm.hir.pointer_cast{{.*}} : memref<?xbf16>
// CHECK: return
// UNSTRUCTURE-LABEL: tt.func public @widen_scalar_multi_addptr
// UNSTRUCTURE: tt.ptr_to_int
// UNSTRUCTURE: arith.addi
// UNSTRUCTURE: tt.int_to_ptr
// UNSTRUCTURE-NOT: tt.bitcast
// SIMT-LINALG-LABEL: func.func @widen_scalar_multi_addptr
// SIMT-LINALG-NOT: tt.int_to_ptr
// SIMT-LINALG: memref.extract_aligned_pointer_as_index
// SIMT-LINALG: memref.reinterpret_cast
// SIMT-LINALG: bufferization.materialize_in_destination
// SIMT-LINALG-NOT: tt.int_to_ptr
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @widen_scalar_multi_addptr(
      %values: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
      %cache: !tt.ptr<i8> {tt.divisibility = 16 : i32},
      %block: i64,
      %position: i64,
      %block_stride: i64 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %c576_i64 = arith.constant 576 : i64
    %c448_i64 = arith.constant 448 : i64
    %block_offset = arith.muli %block, %block_stride : i64
    %block_base = tt.addptr %cache, %block_offset : !tt.ptr<i8>, i64
    %token_offset = arith.muli %position, %c576_i64 : i64
    %token_base = tt.addptr %block_base, %token_offset : !tt.ptr<i8>, i64
    %rope_bytes = tt.addptr %token_base, %c448_i64 : !tt.ptr<i8>, i64
    %rope_bf16 = tt.bitcast %rope_bytes : !tt.ptr<i8> -> !tt.ptr<bf16>
    %rope_bases = tt.splat %rope_bf16 : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>>
    %rope_ptrs = tt.addptr %rope_bases, %range : tensor<8x!tt.ptr<bf16>>, tensor<8xi32>
    %value_bases = tt.splat %values : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>>
    %value_ptrs = tt.addptr %value_bases, %range : tensor<8x!tt.ptr<bf16>>, tensor<8xi32>
    %loaded = tt.load %value_ptrs : tensor<8x!tt.ptr<bf16>>
    tt.store %rope_ptrs, %loaded : tensor<8x!tt.ptr<bf16>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @widen_pointer_inside_loop
// CHECK: scf.for
// CHECK: hivm.hir.pointer_cast{{.*}} : memref<?xbf16>
// CHECK: return
// UNSTRUCTURE-LABEL: tt.func public @widen_pointer_inside_loop
// UNSTRUCTURE: scf.for
// UNSTRUCTURE: tt.ptr_to_int
// UNSTRUCTURE: tt.int_to_ptr
// UNSTRUCTURE: tt.splat
// UNSTRUCTURE: tt.addptr
// UNSTRUCTURE-NOT: tt.bitcast
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @widen_pointer_inside_loop(
      %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
      %count: i64) attributes {noinline = false} {
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c576_i64 = arith.constant 576 : i64
    %c448_i64 = arith.constant 448 : i64
    scf.for %position = %c0_i64 to %count step %c1_i64 : i64 {
      %token_offset = arith.muli %position, %c576_i64 : i64
      %token_base = tt.addptr %src, %token_offset : !tt.ptr<i8>, i64
      %rope_bytes = tt.addptr %token_base, %c448_i64 : !tt.ptr<i8>, i64
      %rope_bf16 = tt.bitcast %rope_bytes : !tt.ptr<i8> -> !tt.ptr<bf16>
      %rope_bases = tt.splat %rope_bf16 : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>>
      %rope_ptrs = tt.addptr %rope_bases, %range : tensor<8x!tt.ptr<bf16>>, tensor<8xi32>
      %values = tt.load %rope_ptrs : tensor<8x!tt.ptr<bf16>>
      %dsts = tt.splat %dst : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>>
      %dst_ptrs = tt.addptr %dsts, %range : tensor<8x!tt.ptr<bf16>>, tensor<8xi32>
      tt.store %dst_ptrs, %values : tensor<8x!tt.ptr<bf16>>
    }
    tt.return
  }
}

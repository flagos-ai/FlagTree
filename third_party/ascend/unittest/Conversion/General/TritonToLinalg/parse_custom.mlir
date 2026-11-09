// RUN: triton-opt --triton-to-unstructure --triton-to-linalg="named-ops=True" --split-input-file %s  | FileCheck %s
// CHECK-LABEL: func.func @parse_custom
// CHECK: %4 = hivm.hir.custom
// CHECK: %reinterpret_cast = memref.reinterpret_cast %4 to offset: [0], sizes: [32], strides: [1] : memref<?xf16> to memref<32xf16, strided<[1]>>
// CHECK: %alloc = memref.alloc() : memref<32xf16>
// CHECK: memref.copy %reinterpret_cast, %alloc : memref<32xf16, strided<[1]>> to memref<32xf16>

module {
  tt.func public @parse_custom(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: i32, %arg3: i32) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i32 = arith.constant 1 : i32
    %0 = hivm.hir.get_sub_block_idx -> i64
    %1 = arith.cmpi eq, %0, %c0_i64 : i64
    scf.if %1 {
      %2 = arith.addi %arg2, %c1_i32 : i32
      %3 = arith.remsi %2, %arg3 : i32
      %4 = hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "foo1"} "foo1" ins(%arg0, %3 : !tt.ptr<f16>, i32) -> !tt.ptr<f16>
      %5 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
      %6 = tt.splat %4 : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
      %7 = tt.addptr %6, %5 : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
      %8 = tt.load %7 : tensor<32x!tt.ptr<f16>>
      %9 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
      %10 = tt.addptr %9, %5 : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
      tt.store %10, %8 : tensor<32x!tt.ptr<f16>>
    }
    hivm.hir.custom {hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "foo2"} "foo2"
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @parse_custom(
// CHECK-SAME: %[[BASE_I64:arg[0-9]+]]: memref<?xi64>, %[[NUM_PROGRAMS_X:arg[0-9]+]]: i32
// CHECK: %[[FOO1:.*]] = hivm.hir.custom {{.*}} "foo1" ins(%[[BASE_I64]], %[[NUM_PROGRAMS_X]], %c0_i64 : memref<?xi64>, i32, i64) -> i32
// CHECK: %[[FOO2:.*]] = hivm.hir.custom {{.*}} "foo2" ins(%arg2, %3 : memref<?xf16>, i32) -> memref<?xf16>
// CHECK: %[[FOO2_VIEW:.*]] = memref.reinterpret_cast %[[FOO2]] to offset: [0], sizes: [32], strides: [1] : memref<?xf16> to memref<32xf16, strided<[1]>>
// CHECK: %[[FOO3:.*]] = hivm.hir.custom {{.*}} "foo3" ins(%[[FOO2_VIEW]], %[[FOO1]] : memref<32xf16, strided<[1]>>, i32) -> memref<32xf16>
// CHECK: %[[FOO3_VIEW:.*]] = memref.reinterpret_cast %[[FOO3]] to offset: [0], sizes: [32], strides: [1] : memref<32xf16> to memref<32xf16, strided<[1]>>
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<32xf16>
// CHECK: memref.copy %[[FOO3_VIEW]], %[[ALLOC]] : memref<32xf16, strided<[1]>> to memref<32xf16>
module {
  tt.func public @parse_custom(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: i32, %arg3: i32, %arg4: !tt.ptr<i64>) {
    %c0_i64 = arith.constant 0 : i64
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = hivm.hir.get_sub_block_idx -> i64
    %1 = arith.cmpi eq, %0, %c0_i64 : i64
    scf.if %1 {
      %2 = arith.addi %arg2, %c1_i32 : i32
      %3 = arith.remsi %2, %arg3 : i32
      %4 = tt.get_num_programs x : i32
      %5 = tt.addptr %arg4, %c0_i32 : !tt.ptr<i64>, i32
      %6 = hivm.hir.custom {hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "foo1"} "foo1" ins(%5, %4, %c0_i64 : !tt.ptr<i64>, i32, i64) -> i32
      %7 = hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "foo2"} "foo2" ins(%arg0, %3 : !tt.ptr<f16>, i32) -> !tt.ptr<f16>
      %8 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
      %9 = tt.splat %7 : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
      %10 = tt.addptr %9, %8 : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
      %11 = tensor.empty() : tensor<32x!tt.ptr<f16>>
      %12 = hivm.hir.custom {SrcPtrIndex = array<i32: 0>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "foo3"} "foo3" ins(%10, %6 : tensor<32x!tt.ptr<f16>>, i32) outs(%11 : tensor<32x!tt.ptr<f16>>) -> tensor<32x!tt.ptr<f16>>
      annotation.mark %12 {ContinuousMemAccess} : tensor<32x!tt.ptr<f16>>
      %13 = tt.load %12 : tensor<32x!tt.ptr<f16>>
      %14 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
      %15 = tt.addptr %14, %8 : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
      tt.store %15, %13 : tensor<32x!tt.ptr<f16>>
    }
    hivm.hir.custom {hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "foo4"} "foo4"
    tt.return
  }
}

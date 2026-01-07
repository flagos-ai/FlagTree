// RUN: triton-adapter-opt --triton-linearize --discrete-mask-access-conversion --triton-to-annotation --triton-to-unstructure --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False' --split-input-file %s | FileCheck %s

// === i8 u8 version ===
tt.func public @fn_npu_i8(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32},
                          %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32},
                          %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32}) {
  %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %1 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<32x!tt.ptr<i8>>
  %2 = tt.addptr %1, %0 : tensor<32x!tt.ptr<i8>>, tensor<32xi32>
  %3 = tt.load %2 : tensor<32x!tt.ptr<i8>>
  %4 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<32x!tt.ptr<i8>>
  %5 = tt.addptr %4, %0 : tensor<32x!tt.ptr<i8>>, tensor<32xi32>
  %6 = tt.load %5 : tensor<32x!tt.ptr<i8>>
  %7 = tt.cat %3, %6 : tensor<32xi8> -> tensor<64xi8>
  %8 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %9 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<64x!tt.ptr<i8>>
  %10 = tt.addptr %9, %8 : tensor<64x!tt.ptr<i8>>, tensor<64xi32>
  tt.store %10, %7 : tensor<64x!tt.ptr<i8>>
  tt.return
}

// CHECK-LABEL: func.func @fn_npu_i8(
// CHECK-NOT: tt.cat
// CHECK: %[[CAST_IN1:.+]] = memref.reinterpret_cast %arg3 to offset: [0], sizes: [32]
// CHECK-SAME: memref<?xi8> to memref<32xi8
// CHECK: %[[ALLOC1:.+]] = memref.alloc() : memref<32xi8>
// CHECK: memref.copy %[[CAST_IN1]], %[[ALLOC1]]
// CHECK: %[[TENSOR1:.+]] = bufferization.to_tensor %[[ALLOC1]]
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<64xi8>
// CHECK: %[[SLICE0:.+]] = tensor.insert_slice %[[TENSOR1]] into %[[EMPTY]][0] [32] [1] : tensor<32xi8> into tensor<64xi8>
// CHECK: %[[SLICE1:.+]] = tensor.insert_slice {{.*}} into %[[SLICE0]][32] [32] [1]
// CHECK: %[[OUT_CAST:.+]] = memref.reinterpret_cast %arg2 to offset: [0], sizes: [64]
// CHECK-SAME: memref<?xi8> to memref<64xi8
// CHECK: bufferization.materialize_in_destination %[[SLICE1]] in writable %[[OUT_CAST]]


// === i16 u16 version ===
tt.func public @fn_npu_i16(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32},
                           %arg1: !tt.ptr<i16> {tt.divisibility = 16 : i32},
                           %arg2: !tt.ptr<i16> {tt.divisibility = 16 : i32}) {
  %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %1 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<32x!tt.ptr<i16>>
  %2 = tt.addptr %1, %0 : tensor<32x!tt.ptr<i16>>, tensor<32xi32>
  %3 = tt.load %2 : tensor<32x!tt.ptr<i16>>
  %4 = tt.splat %arg2 : !tt.ptr<i16> -> tensor<32x!tt.ptr<i16>>
  %5 = tt.addptr %4, %0 : tensor<32x!tt.ptr<i16>>, tensor<32xi32>
  %6 = tt.load %5 : tensor<32x!tt.ptr<i16>>
  %7 = tt.cat %3, %6 : tensor<32xi16> -> tensor<64xi16>
  %8 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %9 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<64x!tt.ptr<i16>>
  %10 = tt.addptr %9, %8 : tensor<64x!tt.ptr<i16>>, tensor<64xi32>
  tt.store %10, %7 : tensor<64x!tt.ptr<i16>>
  tt.return
}

// CHECK-LABEL: func.func @fn_npu_i16(
// CHECK-NOT: tt.cat
// CHECK: %[[CAST_IN1:.+]] = memref.reinterpret_cast %arg3 to offset: [0], sizes: [32]
// CHECK-SAME: memref<?xi16> to memref<32xi16
// CHECK: %[[ALLOC1:.+]] = memref.alloc() : memref<32xi16>
// CHECK: memref.copy %[[CAST_IN1]], %[[ALLOC1]]
// CHECK: %[[TENSOR1:.+]] = bufferization.to_tensor %[[ALLOC1]]
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<64xi16>
// CHECK: %[[SLICE0:.+]] = tensor.insert_slice %[[TENSOR1]] into %[[EMPTY]][0] [32] [1] : tensor<32xi16> into tensor<64xi16>
// CHECK: %[[SLICE1:.+]] = tensor.insert_slice {{.*}} into %[[SLICE0]][32] [32] [1]
// CHECK: %[[OUT_CAST:.+]] = memref.reinterpret_cast %arg2 to offset: [0], sizes: [64]
// CHECK-SAME: memref<?xi16> to memref<64xi16
// CHECK: bufferization.materialize_in_destination %[[SLICE1]] in writable %[[OUT_CAST]]


// === i32 u32 version ===
tt.func public @fn_npu_i32(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32},
                           %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32},
                           %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
  %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %1 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>>
  %2 = tt.addptr %1, %0 : tensor<32x!tt.ptr<i32>>, tensor<32xi32>
  %3 = tt.load %2 : tensor<32x!tt.ptr<i32>>
  %4 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>>
  %5 = tt.addptr %4, %0 : tensor<32x!tt.ptr<i32>>, tensor<32xi32>
  %6 = tt.load %5 : tensor<32x!tt.ptr<i32>>
  %7 = tt.cat %3, %6 : tensor<32xi32> -> tensor<64xi32>
  %8 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %9 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>>
  %10 = tt.addptr %9, %8 : tensor<64x!tt.ptr<i32>>, tensor<64xi32>
  tt.store %10, %7 : tensor<64x!tt.ptr<i32>>
  tt.return
}

// CHECK-LABEL: func.func @fn_npu_i32(
// CHECK-NOT: tt.cat
// CHECK: %[[CAST_IN1:.+]] = memref.reinterpret_cast %arg3 to offset: [0], sizes: [32]
// CHECK-SAME: memref<?xi32> to memref<32xi32
// CHECK: %[[ALLOC1:.+]] = memref.alloc() : memref<32xi32>
// CHECK: memref.copy %[[CAST_IN1]], %[[ALLOC1]]
// CHECK: %[[TENSOR1:.+]] = bufferization.to_tensor %[[ALLOC1]]
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<64xi32>
// CHECK: %[[SLICE0:.+]] = tensor.insert_slice %[[TENSOR1]] into %[[EMPTY]][0] [32] [1] : tensor<32xi32> into tensor<64xi32>
// CHECK: %[[SLICE1:.+]] = tensor.insert_slice {{.*}} into %[[SLICE0]][32] [32] [1]
// CHECK: %[[OUT_CAST:.+]] = memref.reinterpret_cast %arg2 to offset: [0], sizes: [64]
// CHECK-SAME: memref<?xi32> to memref<64xi32
// CHECK: bufferization.materialize_in_destination %[[SLICE1]] in writable %[[OUT_CAST]]


// === i64 u64 version ===
tt.func public @fn_npu_i64(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32},
                           %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32},
                           %arg2: !tt.ptr<i64> {tt.divisibility = 16 : i32}) {
  %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %1 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<32x!tt.ptr<i64>>
  %2 = tt.addptr %1, %0 : tensor<32x!tt.ptr<i64>>, tensor<32xi32>
  %3 = tt.load %2 : tensor<32x!tt.ptr<i64>>
  %4 = tt.splat %arg2 : !tt.ptr<i64> -> tensor<32x!tt.ptr<i64>>
  %5 = tt.addptr %4, %0 : tensor<32x!tt.ptr<i64>>, tensor<32xi32>
  %6 = tt.load %5 : tensor<32x!tt.ptr<i64>>
  %7 = tt.cat %3, %6 : tensor<32xi64> -> tensor<64xi64>
  %8 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %9 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<64x!tt.ptr<i64>>
  %10 = tt.addptr %9, %8 : tensor<64x!tt.ptr<i64>>, tensor<64xi32>
  tt.store %10, %7 : tensor<64x!tt.ptr<i64>>
  tt.return
}

// CHECK-LABEL: func.func @fn_npu_i64(
// CHECK-NOT: tt.cat
// CHECK: %[[CAST_IN1:.+]] = memref.reinterpret_cast %arg3 to offset: [0], sizes: [32]
// CHECK-SAME: memref<?xi64> to memref<32xi64
// CHECK: %[[ALLOC1:.+]] = memref.alloc() : memref<32xi64>
// CHECK: memref.copy %[[CAST_IN1]], %[[ALLOC1]]
// CHECK: %[[TENSOR1:.+]] = bufferization.to_tensor %[[ALLOC1]]
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<64xi64>
// CHECK: %[[SLICE0:.+]] = tensor.insert_slice %[[TENSOR1]] into %[[EMPTY]][0] [32] [1] : tensor<32xi64> into tensor<64xi64>
// CHECK: %[[SLICE1:.+]] = tensor.insert_slice {{.*}} into %[[SLICE0]][32] [32] [1]
// CHECK: %[[OUT_CAST:.+]] = memref.reinterpret_cast %arg2 to offset: [0], sizes: [64]
// CHECK-SAME: memref<?xi64> to memref<64xi64
// CHECK: bufferization.materialize_in_destination %[[SLICE1]] in writable %[[OUT_CAST]]


// === float8_e4m3fn version ===
tt.func public @fn_npu_f8E4M3FN(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32},
                           %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32},
                           %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}) {
  %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %1 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<32x!tt.ptr<f8E4M3FN>>
  %2 = tt.addptr %1, %0 : tensor<32x!tt.ptr<f8E4M3FN>>, tensor<32xi32>
  %3 = tt.load %2 : tensor<32x!tt.ptr<f8E4M3FN>>
  %4 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<32x!tt.ptr<f8E4M3FN>>
  %5 = tt.addptr %4, %0 : tensor<32x!tt.ptr<f8E4M3FN>>, tensor<32xi32>
  %6 = tt.load %5 : tensor<32x!tt.ptr<f8E4M3FN>>
  %7 = tt.cat %3, %6 : tensor<32xf8E4M3FN> -> tensor<64xf8E4M3FN>
  %8 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %9 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<64x!tt.ptr<f8E4M3FN>>
  %10 = tt.addptr %9, %8 : tensor<64x!tt.ptr<f8E4M3FN>>, tensor<64xi32>
  tt.store %10, %7 : tensor<64x!tt.ptr<f8E4M3FN>>
  tt.return
}

// CHECK-LABEL: func.func @fn_npu_f8E4M3FN(
// CHECK-NOT: tt.cat
// CHECK: %[[CAST_IN1:.+]] = memref.reinterpret_cast %arg3 to offset: [0], sizes: [32]
// CHECK-SAME: memref<?xf8E4M3FN> to memref<32xf8E4M3FN
// CHECK: %[[ALLOC1:.+]] = memref.alloc() : memref<32xf8E4M3FN>
// CHECK: memref.copy %[[CAST_IN1]], %[[ALLOC1]]
// CHECK: %[[TENSOR1:.+]] = bufferization.to_tensor %[[ALLOC1]]
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<64xf8E4M3FN>
// CHECK: %[[SLICE0:.+]] = tensor.insert_slice %[[TENSOR1]] into %[[EMPTY]][0] [32] [1] : tensor<32xf8E4M3FN> into tensor<64xf8E4M3FN>
// CHECK: %[[SLICE1:.+]] = tensor.insert_slice {{.*}} into %[[SLICE0]][32] [32] [1]
// CHECK: %[[OUT_CAST:.+]] = memref.reinterpret_cast %arg2 to offset: [0], sizes: [64]
// CHECK-SAME: memref<?xf8E4M3FN> to memref<64xf8E4M3FN
// CHECK: bufferization.materialize_in_destination %[[SLICE1]] in writable %[[OUT_CAST]]


// === float8_e5m2 version ===
tt.func public @fn_npu_f8E5M2(%arg0: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32},
                           %arg1: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32},
                           %arg2: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}) {
  %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %1 = tt.splat %arg1 : !tt.ptr<f8E5M2> -> tensor<32x!tt.ptr<f8E5M2>>
  %2 = tt.addptr %1, %0 : tensor<32x!tt.ptr<f8E5M2>>, tensor<32xi32>
  %3 = tt.load %2 : tensor<32x!tt.ptr<f8E5M2>>
  %4 = tt.splat %arg2 : !tt.ptr<f8E5M2> -> tensor<32x!tt.ptr<f8E5M2>>
  %5 = tt.addptr %4, %0 : tensor<32x!tt.ptr<f8E5M2>>, tensor<32xi32>
  %6 = tt.load %5 : tensor<32x!tt.ptr<f8E5M2>>
  %7 = tt.cat %3, %6 : tensor<32xf8E5M2> -> tensor<64xf8E5M2>
  %8 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %9 = tt.splat %arg0 : !tt.ptr<f8E5M2> -> tensor<64x!tt.ptr<f8E5M2>>
  %10 = tt.addptr %9, %8 : tensor<64x!tt.ptr<f8E5M2>>, tensor<64xi32>
  tt.store %10, %7 : tensor<64x!tt.ptr<f8E5M2>>
  tt.return
}

// CHECK-LABEL: func.func @fn_npu_f8E5M2(
// CHECK-NOT: tt.cat
// CHECK: %[[CAST_IN1:.+]] = memref.reinterpret_cast %arg3 to offset: [0], sizes: [32]
// CHECK-SAME: memref<?xf8E5M2> to memref<32xf8E5M2
// CHECK: %[[ALLOC1:.+]] = memref.alloc() : memref<32xf8E5M2>
// CHECK: memref.copy %[[CAST_IN1]], %[[ALLOC1]]
// CHECK: %[[TENSOR1:.+]] = bufferization.to_tensor %[[ALLOC1]]
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<64xf8E5M2>
// CHECK: %[[SLICE0:.+]] = tensor.insert_slice %[[TENSOR1]] into %[[EMPTY]][0] [32] [1] : tensor<32xf8E5M2> into tensor<64xf8E5M2>
// CHECK: %[[SLICE1:.+]] = tensor.insert_slice {{.*}} into %[[SLICE0]][32] [32] [1]
// CHECK: %[[OUT_CAST:.+]] = memref.reinterpret_cast %arg2 to offset: [0], sizes: [64]
// CHECK-SAME: memref<?xf8E5M2> to memref<64xf8E5M2
// CHECK: bufferization.materialize_in_destination %[[SLICE1]] in writable %[[OUT_CAST]]
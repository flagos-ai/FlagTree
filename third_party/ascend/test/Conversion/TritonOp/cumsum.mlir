// RUN:  triton-adapter-opt --triton-linearize --discrete-mask-access-conversion --triton-to-annotation --triton-to-unstructure --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False' --split-input-file %s | FileCheck %s

// === i8 u8 version ===
module {
  tt.func public @fn_npu_u8(
    %arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}
  ) {
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<8x!tt.ptr<i8>>
    %2 = tt.addptr %1, %0 : tensor<8x!tt.ptr<i8>>, tensor<8xi32>
    %3 = tt.load %2 : tensor<8x!tt.ptr<i8>>
    %4 = "tt.scan"(%3) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%arg2: i8, %arg3: i8):
      %7 = arith.addi %arg2, %arg3 : i8
      tt.scan.return %7 : i8
    }) : (tensor<8xi8>) -> tensor<8xi8>
    %5 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<8x!tt.ptr<i8>>
    %6 = tt.addptr %5, %0 : tensor<8x!tt.ptr<i8>>, tensor<8xi32>
    tt.store %6, %4 : tensor<8x!tt.ptr<i8>>
    tt.return
  }
}

// -----

// CHECK: func.func private @triton_cumsum_0(tensor<8xi8>, i32, i1) -> tensor<8xi8>
// CHECK: %false = arith.constant false
// CHECK: %c0_i32 = arith.constant 0 : i32
// CHECK: %[[INPUT_BUF:.*]] = memref.alloc() : memref<8xi8>
// CHECK: memref.copy {{.*}}, %[[INPUT_BUF]] : memref<8xi8{{.*}}> to memref<8xi8>
// CHECK: %[[TENSOR:.*]] = bufferization.to_tensor %[[INPUT_BUF]] restrict writable : memref<8xi8>
// CHECK: %{{.*}} = call @triton_cumsum_0(%[[TENSOR]], %c0_i32, %false) : (tensor<8xi8>, i32, i1) -> tensor<8xi8>
// CHECK: bufferization.materialize_in_destination


// === i16 u16 version ===
module {
  tt.func public @fn_npu_u16(
    %arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<i16> {tt.divisibility = 16 : i32}
  ) {
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<8x!tt.ptr<i16>>
    %2 = tt.addptr %1, %0 : tensor<8x!tt.ptr<i16>>, tensor<8xi32>
    %3 = tt.load %2 : tensor<8x!tt.ptr<i16>>
    %4 = "tt.scan"(%3) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%arg2: i16, %arg3: i16):
      %7 = arith.addi %arg2, %arg3 : i16
      tt.scan.return %7 : i16
    }) : (tensor<8xi16>) -> tensor<8xi16>
    %5 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<8x!tt.ptr<i16>>
    %6 = tt.addptr %5, %0 : tensor<8x!tt.ptr<i16>>, tensor<8xi32>
    tt.store %6, %4 : tensor<8x!tt.ptr<i16>>
    tt.return
  }
}

// -----

// CHECK: func.func private @triton_cumsum_0(tensor<8xi16>, i32, i1) -> tensor<8xi16>
// CHECK: %false = arith.constant false
// CHECK: %c0_i32 = arith.constant 0 : i32
// CHECK: %[[INPUT_BUF:.*]] = memref.alloc() : memref<8xi16>
// CHECK: memref.copy {{.*}}, %[[INPUT_BUF]] : memref<8xi16{{.*}}> to memref<8xi16>
// CHECK: %[[TENSOR:.*]] = bufferization.to_tensor %[[INPUT_BUF]] restrict writable : memref<8xi16>
// CHECK: %{{.*}} = call @triton_cumsum_0(%[[TENSOR]], %c0_i32, %false) : (tensor<8xi16>, i32, i1) -> tensor<8xi16>
// CHECK: bufferization.materialize_in_destination


// === i32 u32 version ===
module {
  tt.func public @fn_npu_u32(
    %arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}
  ) {
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %2 = tt.addptr %1, %0 : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %3 = tt.load %2 : tensor<8x!tt.ptr<i32>>
    %4 = "tt.scan"(%3) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %7 = arith.addi %arg2, %arg3 : i32
      tt.scan.return %7 : i32
    }) : (tensor<8xi32>) -> tensor<8xi32>
    %5 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %6 = tt.addptr %5, %0 : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    tt.store %6, %4 : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// CHECK: func.func private @triton_cumsum_0(tensor<8xi32>, i32, i1) -> tensor<8xi32>
// CHECK: %false = arith.constant false
// CHECK: %c0_i32 = arith.constant 0 : i32
// CHECK: %[[INPUT_BUF:.*]] = memref.alloc() : memref<8xi32>
// CHECK: memref.copy {{.*}}, %[[INPUT_BUF]] : memref<8xi32{{.*}}> to memref<8xi32>
// CHECK: %[[TENSOR:.*]] = bufferization.to_tensor %[[INPUT_BUF]] restrict writable : memref<8xi32>
// CHECK: %{{.*}} = call @triton_cumsum_0(%[[TENSOR]], %c0_i32, %false) : (tensor<8xi32>, i32, i1) -> tensor<8xi32>
// CHECK: bufferization.materialize_in_destination


// === i64 u64 version ===
module {
  tt.func public @fn_npu_u64(
    %arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}
  ) {
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<8x!tt.ptr<i64>>
    %2 = tt.addptr %1, %0 : tensor<8x!tt.ptr<i64>>, tensor<8xi32>
    %3 = tt.load %2 : tensor<8x!tt.ptr<i64>>
    %4 = "tt.scan"(%3) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%arg2: i64, %arg3: i64):
      %7 = arith.addi %arg2, %arg3 : i64
      tt.scan.return %7 : i64
    }) : (tensor<8xi64>) -> tensor<8xi64>
    %5 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<8x!tt.ptr<i64>>
    %6 = tt.addptr %5, %0 : tensor<8x!tt.ptr<i64>>, tensor<8xi32>
    tt.store %6, %4 : tensor<8x!tt.ptr<i64>>
    tt.return
  }
}

// -----

// CHECK: func.func private @triton_cumsum_0(tensor<8xi64>, i32, i1) -> tensor<8xi64>
// CHECK: %false = arith.constant false
// CHECK: %c0_i32 = arith.constant 0 : i32
// CHECK: %[[INPUT_BUF:.*]] = memref.alloc() : memref<8xi64>
// CHECK: memref.copy {{.*}}, %[[INPUT_BUF]] : memref<8xi64{{.*}}> to memref<8xi64>
// CHECK: %[[TENSOR:.*]] = bufferization.to_tensor %[[INPUT_BUF]] restrict writable : memref<8xi64>
// CHECK: %{{.*}} = call @triton_cumsum_0(%[[TENSOR]], %c0_i32, %false) : (tensor<8xi64>, i32, i1) -> tensor<8xi64>
// CHECK: bufferization.materialize_in_destination


// === f8E4M3FN version ===
module {
  tt.func public @fn_npu_f8E4M3FN(
    %arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}
  ) {
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<8x!tt.ptr<f8E4M3FN>>
    %2 = tt.addptr %1, %0 : tensor<8x!tt.ptr<f8E4M3FN>>, tensor<8xi32>
    %3 = tt.load %2 : tensor<8x!tt.ptr<f8E4M3FN>>
    %4 = "tt.scan"(%3) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%arg2: f8E4M3FN, %arg3: f8E4M3FN):
      %7 = arith.addf %arg2, %arg3 : f8E4M3FN
      tt.scan.return %7 : f8E4M3FN
    }) : (tensor<8xf8E4M3FN>) -> tensor<8xf8E4M3FN>
    %5 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<8x!tt.ptr<f8E4M3FN>>
    %6 = tt.addptr %5, %0 : tensor<8x!tt.ptr<f8E4M3FN>>, tensor<8xi32>
    tt.store %6, %4 : tensor<8x!tt.ptr<f8E4M3FN>>
    tt.return
  }
}

// -----

// CHECK: func.func private @triton_cumsum_0(tensor<8xf8E4M3FN>, i32, i1) -> tensor<8xf8E4M3FN>
// CHECK: %false = arith.constant false
// CHECK: %c0_i32 = arith.constant 0 : i32
// CHECK: %[[INPUT_BUF:.+]] = memref.alloc() : memref<8xf8E4M3FN>
// CHECK: memref.copy {{.*}}, %[[INPUT_BUF]] : memref<8xf8E4M3FN{{.*}}> to memref<8xf8E4M3FN>
// CHECK: %[[TENSOR:.+]] = bufferization.to_tensor %[[INPUT_BUF]] restrict writable : memref<8xf8E4M3FN>
// CHECK: %{{.*}} = call @triton_cumsum_0(%[[TENSOR]], %c0_i32, %false) : (tensor<8xf8E4M3FN>, i32, i1) -> tensor<8xf8E4M3FN>
// CHECK: bufferization.materialize_in_destination


// === f8E5M2 version ===
module {
  tt.func public @fn_npu_f8E5M2(
    %arg0: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}
  ) {
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.splat %arg1 : !tt.ptr<f8E5M2> -> tensor<8x!tt.ptr<f8E5M2>>
    %2 = tt.addptr %1, %0 : tensor<8x!tt.ptr<f8E5M2>>, tensor<8xi32>
    %3 = tt.load %2 : tensor<8x!tt.ptr<f8E5M2>>
    %4 = "tt.scan"(%3) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%arg2: f8E5M2, %arg3: f8E5M2):
      %7 = arith.addf %arg2, %arg3 : f8E5M2
      tt.scan.return %7 : f8E5M2
    }) : (tensor<8xf8E5M2>) -> tensor<8xf8E5M2>
    %5 = tt.splat %arg0 : !tt.ptr<f8E5M2> -> tensor<8x!tt.ptr<f8E5M2>>
    %6 = tt.addptr %5, %0 : tensor<8x!tt.ptr<f8E5M2>>, tensor<8xi32>
    tt.store %6, %4 : tensor<8x!tt.ptr<f8E5M2>>
    tt.return
  }
}

// -----

// CHECK: func.func private @triton_cumsum_0(tensor<8xf8E5M2>, i32, i1) -> tensor<8xf8E5M2>
// CHECK: %false = arith.constant false
// CHECK: %c0_i32 = arith.constant 0 : i32
// CHECK: %[[INPUT_BUF:.+]] = memref.alloc() : memref<8xf8E5M2>
// CHECK: memref.copy {{.*}}, %[[INPUT_BUF]] : memref<8xf8E5M2{{.*}}> to memref<8xf8E5M2>
// CHECK: %[[TENSOR:.+]] = bufferization.to_tensor %[[INPUT_BUF]] restrict writable : memref<8xf8E5M2>
// CHECK: %{{.*}} = call @triton_cumsum_0(%[[TENSOR]], %c0_i32, %false) : (tensor<8xf8E5M2>, i32, i1) -> tensor<8xf8E5M2>
// CHECK: bufferization.materialize_in_destination
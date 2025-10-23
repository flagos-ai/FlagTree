// RUN: triton-adapter-opt --triton-to-linalg -split-input-file %s | FileCheck %s

module {
  tt.func public @load_deinterleave(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>) {
    %cst_1 = arith.constant dense<1> : tensor<32xi32>
    %cst_2 = arith.constant dense<2> : tensor<32xi32>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    // Pay attention here: `multiply 2` tells that last dimension stride is 2
    %1 = arith.muli %0, %cst_2 : tensor<32xi32>
    %2 = arith.addi %1, %cst_1 : tensor<32xi32>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
    %4 = tt.addptr %3, %1 : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
    // even index
    // CHECK: %[[VAL_0:.*]] = bufferization.to_tensor
    // CHECK: tensor.extract_slice %[[VAL_0]][0] [32] [2] : tensor<64xf16> to tensor<32xf16>
    %5 = tt.load %4 : tensor<32x!tt.ptr<f16>>
    %6 = tt.addptr %3, %2 : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
    // odd index
    // CHECK: %[[VAL_1:.*]] = bufferization.to_tensor
    // CHECK: tensor.extract_slice %[[VAL_1]][1] [32] [2] : tensor<64xf16> to tensor<32xf16>
    %7 = tt.load %6 : tensor<32x!tt.ptr<f16>>
    %8 = tt.make_range {end = 64 : i32, start = 32 : i32} : tensor<32xi32>
    %9 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
    %10 = tt.addptr %9, %0 : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
    tt.store %10, %5 : tensor<32x!tt.ptr<f16>>
    %11 = tt.addptr %9, %8 : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
    tt.store %11, %7 : tensor<32x!tt.ptr<f16>>
    tt.return
  }
}


// -----

module {
  tt.func public @store_interleave(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>) {
    %cst_1 = arith.constant dense<1> : tensor<32xi32>
    %cst_2 = arith.constant dense<2> : tensor<32xi32>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
    %2 = tt.addptr %1, %0 : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
    %3 = tt.load %2 : tensor<32x!tt.ptr<f16>>
    %4 = tt.make_range {end = 64 : i32, start = 32 : i32} : tensor<32xi32>
    %5 = tt.addptr %1, %4 : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
    %6 = tt.load %5 : tensor<32x!tt.ptr<f16>>
    // Pay attention here: `multiply 2` tells that last dimension stride is 2
    %7 = arith.muli %0, %cst_2 : tensor<32xi32>
    %8 = arith.addi %7, %cst_1 : tensor<32xi32>
    %9 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
    %10 = tt.addptr %9, %7 : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
    tt.store %10, %3 : tensor<32x!tt.ptr<f16>>
    %11 = tt.addptr %9, %8 : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
    // CHECK: %[[LOAD_FIRST:.*]] = bufferization.to_tensor
    // CHECK: %[[LOAD_SECOND:.*]] = bufferization.to_tensor
    // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<64xf16>
    // CHECK: %[[INSERT_FIRST:.*]] = tensor.insert_slice %[[LOAD_FIRST]] into %[[EMPTY]][0] [32] [2] : tensor<32xf16> into tensor<64xf16>
    // CHECK: %[[INSERT_SECOND:.*]] = tensor.insert_slice %[[LOAD_SECOND]] into %[[INSERT_FIRST]][1] [32] [2] : tensor<32xf16> into tensor<64xf16>
    // CHECK: bufferization.materialize_in_destination %[[INSERT_SECOND]]
    tt.store %11, %6 : tensor<32x!tt.ptr<f16>>
    tt.return
  }
}
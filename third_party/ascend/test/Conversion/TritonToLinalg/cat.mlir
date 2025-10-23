// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s
tt.func public @fn_npu_cat(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: !tt.ptr<f32>, %arg4: !tt.ptr<f32>) attributes {noinline = false} {
  %0 = tt.make_range {end = 8192 : i32, start = 0 : i32} : tensor<8192xi32>
  %1 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<8192x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<8192x!tt.ptr<f32>>, tensor<8192xi32>
  %3 = tt.load %2 : tensor<8192x!tt.ptr<f32>>
  %4 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<8192x!tt.ptr<f32>>
  %5 = tt.addptr %4, %0 : tensor<8192x!tt.ptr<f32>>, tensor<8192xi32>
  %6 = tt.load %5 : tensor<8192x!tt.ptr<f32>>
  %7 = tt.cat %3, %6 : tensor<8192xf32> -> tensor<16384xf32>
  %8 = tt.make_range {end = 16384 : i32, start = 0 : i32} : tensor<16384xi32>
  %9 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16384x!tt.ptr<f32>>
  %10 = tt.addptr %9, %8 : tensor<16384x!tt.ptr<f32>>, tensor<16384xi32>
  tt.store %10, %7 : tensor<16384x!tt.ptr<f32>>
  tt.return
}
//CHECK-LABEL: @fn_npu_cat
//CHECK-NOT: tt.cat
//CHECK: %[[VAL0:.*]] = bufferization.to_tensor %[[ADDR0:.*]] restrict writable : memref<8192xf32>
//CHECK: %[[VAL1:.*]] = bufferization.to_tensor %[[ADDR1:.*]] restrict writable : memref<8192xf32>
//CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<16384xf32>
//CHECK: %[[INSERT0:.*]] = tensor.insert_slice %[[VAL0]] into %[[EMPTY]][0] [8192] [1] : tensor<8192xf32> into tensor<16384xf32>
//CHECK: %[[INSERT1:.*]] = tensor.insert_slice %[[VAL1]] into %[[INSERT0]][8192] [8192] [1] : tensor<8192xf32> into tensor<16384xf32>

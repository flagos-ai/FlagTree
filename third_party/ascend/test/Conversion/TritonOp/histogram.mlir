// RUN: triton-adapter-opt %s --triton-linearize --discrete-mask-access-conversion --triton-to-annotation '--triton-to-unstructure=compile-on-910-95=False force-simt-template=False' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False compile-on-910-95=False' --split-input-file %s | FileCheck %s

module {
  tt.func public @histogram_kernel(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32>
    %1 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %2 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<2048x!tt.ptr<i64>>
    %3 = tt.addptr %2, %0 : tensor<2048x!tt.ptr<i64>>, tensor<2048xi32>
    %4 = tt.load %3 : tensor<2048x!tt.ptr<i64>>
    %5 = tt.histogram %4 : tensor<2048xi64> -> tensor<2xi32>
    %6 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<2x!tt.ptr<i64>>
    %7 = tt.addptr %6, %1 : tensor<2x!tt.ptr<i64>>, tensor<2xi32>
    %8 = arith.extsi %5 : tensor<2xi32> to tensor<2xi64>
    tt.store %7, %8 : tensor<2x!tt.ptr<i64>>
    tt.return
  }
}

// CHECK-LABEL: func.func @histogram_kernel
// CHECK: %[[INPUT:.+]] = bufferization.to_tensor
// CHECK: hfusion.histogram %[[INPUT]], 2 : tensor<2048xi64> -> tensor<2xi32>
// CHECK-NOT: tt.histogram


// -----

module {
  tt.func public @histogram_kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32>
    %1 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %2 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<2048x!tt.ptr<i32>>
    %3 = tt.addptr %2, %0 : tensor<2048x!tt.ptr<i32>>, tensor<2048xi32>
    %4 = tt.load %3 : tensor<2048x!tt.ptr<i32>>
    %5 = tt.histogram %4 : tensor<2048xi32> -> tensor<2xi32>
    %6 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<2x!tt.ptr<i32>>
    %7 = tt.addptr %6, %1 : tensor<2x!tt.ptr<i32>>, tensor<2xi32>
    tt.store %7, %5 : tensor<2x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK-LABEL: func.func @histogram_kernel
// CHECK: %[[INPUT:.+]] = bufferization.to_tensor
// CHECK: hfusion.histogram %[[INPUT]], 2 : tensor<2048xi32> -> tensor<2xi32>
// CHECK-NOT: tt.histogram

// RUN: triton-adapter-opt %s --discrete-mask-access-conversion --split-input-file %s | FileCheck %s
// RUN: triton-adapter-opt %s --triton-linearize --discrete-mask-access-conversion --triton-to-annotation '--triton-to-unstructure=compile-on-910-95=False force-simt-template=False' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False compile-on-910-95=False'

// CHECK-LABEL: tt.func @discrete_load
// CHECK: %[[loaded_value:.*]] = tt.load %[[load_ptr:.*]]
// CHECK: %[[value:.*]] = arith.select %[[mask:.*]], %[[loaded_value]], %[[other:.*]]
// CHECK: tt.store %[[store_ptr:.*]], %[[value]]
tt.func @discrete_load(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) {
  %cst = arith.constant dense<0> : tensor<1024xi32>
  %cst_0 = arith.constant dense<200> : tensor<1024xi32>
  %cst_1 = arith.constant dense<400> : tensor<1024xi32>
  %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  %1 = arith.cmpi slt, %0, %cst_0 : tensor<1024xi32>
  %2 = arith.cmpi sgt, %0, %cst_1 : tensor<1024xi32>
  %3 = arith.ori %1, %2 : tensor<1024xi1>
  %4 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %5 = tt.addptr %4, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %6 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %7 = tt.addptr %6, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %8 = tt.load %5, %3, %cst : tensor<1024x!tt.ptr<i32>>
  tt.store %7, %8 : tensor<1024x!tt.ptr<i32>>
  tt.return
}

// CHECK-LABEL: tt.func @discrete_load_without_other
// CHECK: %[[other:.*]] = arith.constant dense<0>
// CHECK: %[[loaded_value:.*]] = tt.load %[[load_ptr:.*]]
// CHECK: %[[value:.*]] = arith.select %[[mask:.*]], %[[loaded_value]], %[[other]]
// CHECK: tt.store %[[store_ptr:.*]], %[[value]]
tt.func @discrete_load_without_other(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) {
  %cst = arith.constant dense<0> : tensor<1024xi32>
  %cst_0 = arith.constant dense<200> : tensor<1024xi32>
  %cst_1 = arith.constant dense<400> : tensor<1024xi32>
  %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  %1 = arith.cmpi slt, %0, %cst_0 : tensor<1024xi32>
  %2 = arith.cmpi sgt, %0, %cst_1 : tensor<1024xi32>
  %3 = arith.ori %1, %2 : tensor<1024xi1>
  %4 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %5 = tt.addptr %4, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %6 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %7 = tt.addptr %6, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %8 = tt.load %5, %3 : tensor<1024x!tt.ptr<i32>>
  tt.store %7, %8 : tensor<1024x!tt.ptr<i32>>
  tt.return
}

// CHECK-LABEL: tt.func @discrete_store
// CHECK: %[[loaded_value:.*]] = tt.load %[[load_ptr:.*]] : tensor<1024x!tt.ptr<i32>>
// CHECK: %[[origin_value:.*]] = tt.load %[[store_ptr:.*]] : tensor<1024x!tt.ptr<i32>>
// CHECK: %[[store_value:.*]] = arith.select %[[mask:.*]], %[[loaded_value]], %[[origin_value]]
// CHECK: tt.store %[[store_ptr]], %[[store_value]] {DiscreteMask}
tt.func @discrete_store(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) {
  %cst = arith.constant dense<0> : tensor<1024xi32>
  %cst_0 = arith.constant dense<200> : tensor<1024xi32>
  %cst_1 = arith.constant dense<400> : tensor<1024xi32>
  %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  %1 = arith.cmpi slt, %0, %cst_0 : tensor<1024xi32>
  %2 = arith.cmpi sgt, %0, %cst_1 : tensor<1024xi32>
  %3 = arith.ori %1, %2 : tensor<1024xi1>
  %4 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %5 = tt.addptr %4, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %6 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %7 = tt.addptr %6, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %8 = tt.load %5 : tensor<1024x!tt.ptr<i32>>
  tt.store %7, %8, %3 : tensor<1024x!tt.ptr<i32>>
  tt.return
}

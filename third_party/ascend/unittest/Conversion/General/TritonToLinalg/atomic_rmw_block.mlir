// RUN: triton-opt -allow-unregistered-dialect --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @moe_align_block_size_stage4(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32} , %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32} , %arg3: !tt.ptr<i32> {tt.divisibility = 16 : i32} , %arg4: !tt.ptr<i32> {tt.divisibility = 16 : i32} , %arg5: i32) attributes {noinline = false} {
    %cst = arith.constant dense<1> : tensor<1xi32>
    %cst_0 = arith.constant dense<0> : tensor<1xi32>
    %c250_i32 = arith.constant 250 : i32
    %c16_i32 = arith.constant 16 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.addptr %arg4, %0 : !tt.ptr<i32>, i32
    %2 = tt.load %1 : !tt.ptr<i32>
    %3 = tt.addptr %1, %c1_i32 : !tt.ptr<i32>, i32
    %4 = tt.load %3 : !tt.ptr<i32>
    scf.for %arg6 = %2 to %4 step %c16_i32  : i32 {
      %22 = arith.divsi %arg6, %c16_i32 : i32
      %23 = tt.addptr %arg2, %22 : !tt.ptr<i32>, i32
      tt.store %23, %0 : !tt.ptr<i32>
    }
    %5 = arith.muli %0, %c250_i32 : i32
    %6 = tt.splat %0 : i32 -> tensor<1xi32>
    %7 = arith.cmpi slt, %0, %arg5 : i32
    %8 = tt.splat %7 : i1 -> tensor<1xi1>
    %9 = tt.addptr %arg0, %0 : !tt.ptr<i32>, i32
    %10 = tt.splat %9 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
    %11 = tt.load %10, %8, %cst_0 : tensor<1x!tt.ptr<i32>>
    %12 = tt.addptr %arg3, %5 : !tt.ptr<i32>, i32
    %13 = tt.splat %12 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
    %14 = tt.addptr %13, %11 : tensor<1x!tt.ptr<i32>>, tensor<1xi32>
    %15 = tt.atomic_rmw add, acq_rel, gpu, %14, %cst, %8 : (tensor<1x!tt.ptr<i32>>, tensor<1xi32>, tensor<1xi1>) -> tensor<1xi32>
    %16 = tt.splat %arg4 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
    %17 = tt.addptr %16, %11 : tensor<1x!tt.ptr<i32>>, tensor<1xi32>
    %18 = tt.load %17, %8, %cst_0 : tensor<1x!tt.ptr<i32>>
    %19 = arith.addi %15, %18 : tensor<1xi32>
    %20 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
    %21 = tt.addptr %20, %19 : tensor<1x!tt.ptr<i32>>, tensor<1xi32>
    tt.store %21, %6, %8 : tensor<1x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK-LABEL: func.func @moe_align_block_size_stage4

// CHECK: %[[MASK:.*]] = arith.cmpi slt
// CHECK: %[[MASK_SIZE:.*]] = arith.index_castui %[[MASK]] : i1 to index
// CHECK: %[[ATOMIC_PTR:.*]] = memref.reinterpret_cast %{{.*}} to offset: [%{{.*}}], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
// CHECK: %[[ATOMIC_VIEW:.*]] = memref.subview %[[ATOMIC_PTR]][0] [%[[MASK_SIZE]]] [1] : memref<1xi32, strided<[1], offset: ?>> to memref<?xi32, strided<[1], offset: ?>>
// CHECK: %[[ATOMIC_VALUE:.*]] = tensor.extract_slice %{{.*}}[0] [%[[MASK_SIZE]]] [1] : tensor<1xi32> to tensor<?xi32>
// CHECK: %[[RESULT_VIEW:.*]] = memref.subview %{{.*}}[0] [%[[MASK_SIZE]]] [1] : memref<1xi32> to memref<?xi32, strided<[1]>>
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: memref.copy %[[ATOMIC_VIEW]], %[[RESULT_VIEW]] :
// CHECK: hivm.hir.store ins(%[[ATOMIC_VALUE]] : tensor<?xi32>) outs(%[[ATOMIC_VIEW]] : memref<?xi32, strided<[1], offset: ?>>) {already_sync} atomic = <add>
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[LOCK]] : memref<1xi64>)

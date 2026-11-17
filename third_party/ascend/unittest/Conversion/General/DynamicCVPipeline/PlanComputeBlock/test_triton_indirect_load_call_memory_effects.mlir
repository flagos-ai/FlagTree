// RUN: triton-opt --debug-only=memory-effects-tracker --plan-compute-block %s 2>&1 | FileCheck %s

module {
  func.func private @triton_indirect_load_0(%arg0: memref<4x4xf32>) -> tensor<4x4xf32>

  // CHECK-LABEL: func.func @triton_indirect_load_call_is_not_memory_barrier
  // CHECK: %[[ALLOC:[A-Za-z0-9_]+]] = memref.alloc()
  // CHECK: linalg.fill {{.*}} outs(%[[ALLOC]]
  // CHECK: Analyzing op %{{.*}} = func.call @triton_indirect_load_0(%[[ALLOC]])
  // CHECK-NEXT: [memory-effects-tracker] Defs:
  // CHECK-NEXT: [memory-effects-tracker] Preds:
  // CHECK-NEXT: [memory-effects-tracker] Analyzing op %{{.*}} = bufferization.to_tensor %[[ALLOC]]
  // CHECK-NEXT: [memory-effects-tracker] Defs:
  // CHECK-NEXT: [memory-effects-tracker] linalg.fill {{.*}} outs(%[[ALLOC]]
  // CHECK-NEXT: [memory-effects-tracker] Preds:
  // CHECK-NEXT: [memory-effects-tracker] linalg.fill {{.*}} outs(%[[ALLOC]]
  func.func @triton_indirect_load_call_is_not_memory_barrier(
    %arg0: memref<4xf32>,
    %rhs: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %unused = memref.load %arg0[%c0] : memref<4xf32>
    %alloc = memref.alloc() : memref<4x4xf32>
    linalg.fill ins(%zero : f32) outs(%alloc : memref<4x4xf32>)
    %ignored = func.call @triton_indirect_load_0(%alloc) : (memref<4x4xf32>) -> tensor<4x4xf32>
    %lhs = bufferization.to_tensor %alloc restrict writable : memref<4x4xf32> to tensor<4x4xf32>
    %out = tensor.empty() : tensor<4x4xf32>
    %init = linalg.fill ins(%zero : f32) outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
    %mm = linalg.matmul ins(%lhs, %rhs : tensor<4x4xf32>, tensor<4x4xf32>)
      outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %mm : tensor<4x4xf32>
  }
}

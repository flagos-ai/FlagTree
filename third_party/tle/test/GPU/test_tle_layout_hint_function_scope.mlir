// RUN: triton-opt %s -convert-triton-to-tritongpu='target=cuda:90 num-warps=8' | FileCheck %s

// An unrelated function's explicit layout must not split pure sources in a
// function with no layout hints. Check conversion alone, before CSE can hide
// the redundant cloning. Shared splats/ranges and unrelated scalar effects
// remain shared/unique even when another function contains set_layout.
#layout = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>

module {
  // CHECK-LABEL: tt.func @unhinted
  tt.func @unhinted(%output: !tt.ptr<i32>, %stride: i32) {
    // CHECK: %[[RANGE:.*]] = tt.make_range
    // CHECK-NOT: tt.make_range
    %range = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32>
    // CHECK: %[[SPLAT:.*]] = tt.splat %arg1
    %strides = tt.splat %stride : i32 -> tensor<1024xi32>
    // CHECK: %[[OFFSET:.*]] = arith.muli %[[RANGE]], %[[SPLAT]]
    %offset = arith.muli %range, %strides : tensor<1024xi32>
    // CHECK-NOT: tt.make_range
    // CHECK-NOT: tt.splat %arg1
    // CHECK: %[[VALUE:.*]] = arith.addi %[[OFFSET]], %[[SPLAT]]
    %value = arith.addi %offset, %strides : tensor<1024xi32>
    %base = tt.splat %output : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
    // CHECK: %[[PTR:.*]] = tt.addptr %{{.*}}, %[[RANGE]]
    %ptr = tt.addptr %base, %range : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
    // CHECK: tt.store %[[PTR]], %[[VALUE]]
    tt.store %ptr, %value : tensor<1024x!tt.ptr<i32>>
    // CHECK-NOT: tt.make_range
    // CHECK-NOT: tt.store
    // CHECK: tt.return
    tt.return
  }

  // CHECK-LABEL: tt.func @hinted
  tt.func @hinted(%output: !tt.ptr<i32>) {
    %range = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32>
    %value = tle.gpu.set_layout %range {target_encoding = #layout} : tensor<1024xi32> -> tensor<1024xi32>
    %base = tt.splat %output : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
    %ptr = tt.addptr %base, %value : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
    // CHECK: tt.store {{.*}}tle.explicit_memory_encoding
    tt.store %ptr, %value : tensor<1024x!tt.ptr<i32>>
    tt.return
  }
}

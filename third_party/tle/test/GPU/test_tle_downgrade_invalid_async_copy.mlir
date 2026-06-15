// RUN: triton-opt %s -split-input-file -triton-tle-downgrade-invalid-async-copy | FileCheck %s

// CHECK-LABEL: tt.func @downgrade_bf16
// CHECK-NOT: ttg.async_copy_global_to_local
// CHECK-NOT: ttg.async_commit_group
// CHECK-NOT: ttg.async_wait
// CHECK: %[[LOAD:.*]] = tt.load %{{.*}} : tensor<32x512x!tt.ptr<bf16>, #{{.*}}>
// CHECK: ttg.local_store %[[LOAD]], %{{.*}} : tensor<32x512xbf16, #{{.*}}> -> !ttg.memdesc<32x512xbf16
// CHECK: %{{.*}} = ttg.local_load %{{.*}} : !ttg.memdesc<32x512xbf16
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
tt.func @downgrade_bf16(%input: tensor<32x512x!tt.ptr<bf16>, #blocked>,
                        %view: !ttg.memdesc<32x512xbf16, #shared, #smem, mutable>)
                        -> tensor<32x512xbf16, #blocked> {
  %token = ttg.async_copy_global_to_local %input, %view : tensor<32x512x!tt.ptr<bf16>, #blocked> -> <32x512xbf16, #shared, #smem, mutable>
  %commit = ttg.async_commit_group tokens %token
  %wait = ttg.async_wait %commit {num = 0 : i32}
  %loaded = ttg.local_load %view token %wait : !ttg.memdesc<32x512xbf16, #shared, #smem, mutable> -> tensor<32x512xbf16, #blocked>
  tt.return %loaded : tensor<32x512xbf16, #blocked>
}
}

// -----

// CHECK-LABEL: tt.func @coalesced_bf16_fallback
// CHECK: %[[PTR_CVT:.*]] = ttg.convert_layout %{{.*}} : tensor<64x512x!tt.ptr<bf16>, #{{.*}}> -> tensor<64x512x!tt.ptr<bf16>, #[[VEC:.*]]>
// CHECK: %[[LOAD:.*]] = tt.load %[[PTR_CVT]] : tensor<64x512x!tt.ptr<bf16>, #[[VEC]]>
// CHECK: %[[VAL_CVT:.*]] = ttg.convert_layout %[[LOAD]] : tensor<64x512xbf16, #[[VEC]]> -> tensor<64x512xbf16, #{{.*}}>
// CHECK: ttg.local_store %[[VAL_CVT]], %{{.*}} : tensor<64x512xbf16, #{{.*}}> -> !ttg.memdesc<64x512xbf16
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
tt.func @coalesced_bf16_fallback(
    %ptrs: tensor<64x512x!tt.ptr<bf16>, #blocked> {tt.contiguity = dense<[1, 512]> : tensor<2xi32>, tt.divisibility = dense<[1, 16]> : tensor<2xi32>},
    %view: !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>) {
  %token = ttg.async_copy_global_to_local %ptrs, %view {tle.local_ptr_async_store} : tensor<64x512x!tt.ptr<bf16>, #blocked> -> <64x512xbf16, #shared, #smem, mutable>
  %commit = ttg.async_commit_group tokens %token
  ttg.async_wait %commit {num = 0 : i32}
  tt.return
}
}

// -----

// CHECK-LABEL: tt.func @drop_loop_free_partition_eviction_policy
// CHECK: partition0
// CHECK: ttg.async_copy_global_to_local
// CHECK-SAME: {tle.local_ptr_async_store}
// CHECK-NOT: evictionPolicy = evict_last
// CHECK: ttg.warp_return
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
tt.func @drop_loop_free_partition_eviction_policy(%ptrs: tensor<64x128x!tt.ptr<f32>, #blocked>, %view: !ttg.memdesc<64x128xf32, #shared, #smem, mutable>) {
  ttg.warp_specialize(%view, %ptrs) attributes {requestedRegisters = array<i32: 72>}
  default {
    ttg.warp_yield
  }
  partition0(%arg0: !ttg.memdesc<64x128xf32, #shared, #smem, mutable>, %arg1: tensor<64x128x!tt.ptr<f32>, #blocked>) num_warps(4) {
    %token = ttg.async_copy_global_to_local %arg1, %arg0 evictionPolicy = evict_last {tle.local_ptr_async_store} : tensor<64x128x!tt.ptr<f32>, #blocked> -> <64x128xf32, #shared, #smem, mutable>
    ttg.warp_return
  } : (!ttg.memdesc<64x128xf32, #shared, #smem, mutable>, tensor<64x128x!tt.ptr<f32>, #blocked>) -> ()
  tt.return
}
}

// -----

// CHECK-LABEL: tt.func @preserve_loop_partition_eviction_policy
// CHECK: partition0
// CHECK: scf.for
// CHECK: ttg.async_copy_global_to_local
// CHECK-SAME: evictionPolicy = evict_last
// CHECK-SAME: {tle.local_ptr_async_store}
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
tt.func @preserve_loop_partition_eviction_policy(%ptrs: tensor<64x128x!tt.ptr<f32>, #blocked>, %view: !ttg.memdesc<64x128xf32, #shared, #smem, mutable>) {
  ttg.warp_specialize(%view, %ptrs) attributes {requestedRegisters = array<i32: 72>}
  default {
    ttg.warp_yield
  }
  partition0(%arg0: !ttg.memdesc<64x128xf32, #shared, #smem, mutable>, %arg1: tensor<64x128x!tt.ptr<f32>, #blocked>) num_warps(4) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %i = %c0 to %c2 step %c1 {
      %token = ttg.async_copy_global_to_local %arg1, %arg0 evictionPolicy = evict_last {tle.local_ptr_async_store} : tensor<64x128x!tt.ptr<f32>, #blocked> -> <64x128xf32, #shared, #smem, mutable>
    }
    ttg.warp_return
  } : (!ttg.memdesc<64x128xf32, #shared, #smem, mutable>, tensor<64x128x!tt.ptr<f32>, #blocked>) -> ()
  tt.return
}
}

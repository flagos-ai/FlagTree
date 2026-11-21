// RUN: triton-opt %s --triton-tle-optimize-local-pointer-async-stores | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [2, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @explicit_sync_store
  // CHECK-NOT: async_copy_global_to_local
  // CHECK: %[[V:.*]] = tt.load
  // CHECK-SAME: tt.load.async = false
  // CHECK: tt.store %{{.*}}, %[[V]]
  // CHECK-NOT: async_copy_global_to_local
  tt.func @explicit_sync_store(%gptr: tensor<16x64x!tt.ptr<bf16>, #blocked>) {
    %smem = ttg.local_alloc : () -> !ttg.memdesc<16x64xbf16, #shared, #smem, mutable>
    %row = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %r = tt.expand_dims %row {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    %rr = tt.broadcast %r : tensor<16x1xi32, #blocked> -> tensor<16x64xi32, #blocked>
    %col = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %c = tt.expand_dims %col {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %cc = tt.broadcast %c : tensor<1x64xi32, #blocked> -> tensor<16x64xi32, #blocked>
    %ptr = "tle.local_pointers"(%smem, %rr, %cc) : (!ttg.memdesc<16x64xbf16, #shared, #smem, mutable>, tensor<16x64xi32, #blocked>, tensor<16x64xi32, #blocked>) -> tensor<16x64x!tt.ptr<bf16, 3>, #blocked>
    %v = tt.load %gptr {tt.load.async = false} : tensor<16x64x!tt.ptr<bf16>, #blocked>
    tt.store %ptr, %v : tensor<16x64x!tt.ptr<bf16, 3>, #blocked>
    tt.return
  }
}

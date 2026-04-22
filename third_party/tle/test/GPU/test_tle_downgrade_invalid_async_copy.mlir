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

// CHECK-DAG: #[[VEC:.*]] = #ttg.blocked<{{.*}}sizePerThread = [1, 8]{{.*}}order = [1, 0]{{.*}}>
// CHECK-LABEL: tt.func @coalesced_bf16_fallback
// CHECK: %[[PTR_CVT:.*]] = ttg.convert_layout %{{.*}} : tensor<64x512x!tt.ptr<bf16>, #{{.*}}> -> tensor<64x512x!tt.ptr<bf16>, #[[VEC]]>
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

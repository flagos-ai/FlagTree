// RUN: triton-opt %s -triton-tle-downgrade-invalid-async-copy | FileCheck %s

// CHECK-LABEL: tt.func @legalize_required_bf16_1x256
// CHECK: %[[PTRS:.*]] = ttg.convert_layout %{{.*}} : tensor<1x256x!tt.ptr<bf16>, #{{.*}}> -> tensor<1x256x!tt.ptr<bf16>, #[[LEGAL:.*]]>
// CHECK: %[[TOKEN:.*]] = ttg.async_copy_global_to_local %[[PTRS]], %{{.*}} {contiguity = 2 : i32, tle.required_async_copy}
// CHECK: %[[COMMIT:.*]] = ttg.async_commit_group tokens %[[TOKEN]]
// CHECK: ttg.async_wait %[[COMMIT]] {num = 0 : i32}
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
tt.func @legalize_required_bf16_1x256(
    %ptrs: tensor<1x256x!tt.ptr<bf16>, #blocked> {tt.contiguity = dense<[1, 256]> : tensor<2xi32>, tt.divisibility = dense<[1, 16]> : tensor<2xi32>},
    %view: !ttg.memdesc<1x256xbf16, #shared, #smem, mutable>) {
  %token = ttg.async_copy_global_to_local %ptrs, %view {tle.required_async_copy} : tensor<1x256x!tt.ptr<bf16>, #blocked> -> <1x256xbf16, #shared, #smem, mutable>
  %commit = ttg.async_commit_group tokens %token
  ttg.async_wait %commit {num = 0 : i32}
  tt.return
}
}

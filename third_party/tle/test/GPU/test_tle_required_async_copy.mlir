// RUN: not triton-opt %s -triton-tle-downgrade-invalid-async-copy 2>&1 | FileCheck %s

// CHECK: error: explicit tle.gpu.copy(is_async=True) cannot be represented by a legal NVIDIA cp.async width; refusing synchronous downgrade
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
tt.func @required_async_too_narrow(
    %ptrs: tensor<1x!tt.ptr<bf16>, #blocked>,
    %view: !ttg.memdesc<1xbf16, #shared, #smem, mutable>) {
  %token = ttg.async_copy_global_to_local %ptrs, %view {tle.required_async_copy} : tensor<1x!tt.ptr<bf16>, #blocked> -> <1xbf16, #shared, #smem, mutable>
  %commit = ttg.async_commit_group tokens %token
  ttg.async_wait %commit {num = 0 : i32}
  tt.return
}
}

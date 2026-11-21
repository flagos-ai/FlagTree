// RUN: triton-opt %s --verify-each --tritonhcugpu-update-async-wait-count="arch-generation-name=gfx936" | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"} {
  // CHECK-LABEL: @wait_credit_requires_every_wave(
  tt.func public @wait_credit_requires_every_wave(%A: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) {
    %full = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32, #blocked>
    %half = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32, #blocked>
    %ap = tt.splat %A : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #blocked>
    %p = tt.addptr %ap, %half : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi32, #blocked>
    %buf0 = ttg.local_alloc : () -> !ttg.memdesc<512xbf16, #shared, #smem, mutable>
    %buf1 = ttg.local_alloc : () -> !ttg.memdesc<256xbf16, #shared, #smem, mutable>
    %buf2 = ttg.local_alloc : () -> !ttg.memdesc<512xbf16, #shared, #smem, mutable>
    %buf3 = ttg.local_alloc : () -> !ttg.memdesc<256xbf16, #shared, #smem, mutable>

    %a = hcug.buffer_load_to_local %A[%full] into %buf0 : <bf16>[tensor<512xi32, #blocked>] -> <512xbf16, #shared, #smem, mutable>
    %b = hcug.buffer_load_to_local %A[%half] into %buf1 : <bf16>[tensor<256xi32, #blocked>] -> <256xbf16, #shared, #smem, mutable>
    // The second buffer copy only issues in two of the four waves.
    // CHECK: hcug.async_wait{{.*}}num_inst = 0
    ttg.async_wait %a {num = 0 : i32}
    %c = hcug.buffer_load_to_local %A[%full] into %buf2 : <bf16>[tensor<512xi32, #blocked>] -> <512xbf16, #shared, #smem, mutable>
    // A later copy that issues in every wave still permits a partial wait.
    // CHECK: hcug.async_wait{{.*}}num_inst = 1
    ttg.async_wait %b {num = 0 : i32}
    %d = ttg.async_copy_global_to_local %p, %buf3 : tensor<256x!tt.ptr<bf16>, #blocked> -> <256xbf16, #shared, #smem, mutable>
    // The global-pointer lowering also predicates out redundant waves.
    // CHECK: hcug.async_wait{{.*}}num_inst = 0
    ttg.async_wait %c {num = 0 : i32}
    tt.return
  }
}

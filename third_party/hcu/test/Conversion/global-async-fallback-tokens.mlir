// RUN: triton-opt %s --split-input-file --verify-each --tritonhcugpu-coalesce-async-copy="arch-generation-name=gfx936" | FileCheck %s --check-prefix=FB --enable-var-scope
// RUN: triton-opt %s --split-input-file --verify-each --tritonhcugpu-coalesce-async-copy="arch-generation-name=gfx936" --test-tritonhcugpu-membar --tritonhcugpu-update-async-wait-count="arch-generation-name=gfx936" | FileCheck %s --check-prefix=PUB --enable-var-scope
// RUN: triton-opt %s --split-input-file --verify-each --tritonhcugpu-coalesce-async-copy="arch-generation-name=gfx936" --tritonhcugpu-update-async-wait-count="arch-generation-name=gfx936" --allocate-hcugpu-shared-memory --convert-triton-hcugpu-to-llvm="arch=gfx936" --convert-cf-to-llvm --convert-arith-to-llvm --canonicalize --cse | FileCheck %s --check-prefix=LLVM --enable-var-scope

// gfx936 GLOBAL direct-to-LDS legality and conservative register fallback.
// CPU compiler regression inputs; these are not runtime test kernels.
// Reuses AMD coalesce/update-wait and HCU async-copy-war public test shapes.
// Empty commits are zero-instruction IR boundaries, not hardware completion.
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"} {
// FB-LABEL: @valid_global32_reuse(
// PUB-LABEL: @valid_global32_reuse(
// LLVM-LABEL: llvm.func @valid_global32_reuse(
// FB-NOT: tt.load
// FB: ttg.async_copy_global_to_local
// FB: ttg.async_copy_global_to_local
// FB-NOT: tt.load
// FB: tt.return
// PUB: hcug.async_wait{{.*}}num_inst = 0
// PUB-NEXT: ttg.local_barrier
// PUB: ttg.local_load
// PUB: tt.store
// PUB-NEXT: ttg.local_barrier
// PUB-NEXT: {{.*}}ttg.async_copy_global_to_local
// PUB: hcug.async_wait{{.*}}num_inst = 0
// PUB-NEXT: ttg.local_barrier
// LLVM: rocdl.global.load.lds
// LLVM: rocdl.s.waitcnt -49168
// LLVM: rocdl.s.barrier
// LLVM: llvm.load {{.*}}!llvm.ptr<3>
// LLVM: rocdl.s.barrier
// LLVM: rocdl.global.load.lds
// LLVM: rocdl.s.waitcnt -49168
  tt.func public @valid_global32_reuse(%A: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %O: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %i = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32, #blocked>
    %baseA = tt.splat %A : !tt.ptr<f16> -> tensor<512x!tt.ptr<f16>, #blocked>
    %p = tt.addptr %baseA, %i : tensor<512x!tt.ptr<f16>, #blocked>, tensor<512xi32, #blocked>
    %baseO = tt.splat %O : !tt.ptr<f16> -> tensor<512x!tt.ptr<f16>, #blocked>
    %out = tt.addptr %baseO, %i : tensor<512x!tt.ptr<f16>, #blocked>, tensor<512xi32, #blocked>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %a0 = ttg.async_copy_global_to_local %p, %buf : tensor<512x!tt.ptr<f16>, #blocked> -> <512xf16, #shared, #smem, mutable>
    %g0 = ttg.async_commit_group tokens %a0
    %w0 = ttg.async_wait %g0 {num = 0 : i32}
    %v0 = ttg.local_load %buf token %w0 : !ttg.memdesc<512xf16, #shared, #smem, mutable> -> tensor<512xf16, #blocked>
    tt.store %out, %v0 : tensor<512x!tt.ptr<f16>, #blocked>
    %a1 = ttg.async_copy_global_to_local %p, %buf : tensor<512x!tt.ptr<f16>, #blocked> -> <512xf16, #shared, #smem, mutable>
    %g1 = ttg.async_commit_group tokens %a1
    %w1 = ttg.async_wait %g1 {num = 0 : i32}
    %v1 = ttg.local_load %buf token %w1 : !ttg.memdesc<512xf16, #shared, #smem, mutable> -> tensor<512xf16, #blocked>
    tt.store %out, %v1 : tensor<512x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"} {
// FB-LABEL: @merged_valid_and_fallback(
// PUB-LABEL: @merged_valid_and_fallback(
// LLVM-LABEL: llvm.func @merged_valid_and_fallback(
// FB: %[[A:.*]] = ttg.async_copy_global_to_local
// FB: %[[LOAD:.*]] = tt.load
// FB-SAME: isVolatile = true
// FB-SAME: tt.load.async = false
// FB: ttg.local_store %[[LOAD]],
// FB: %[[DONE:.*]] = ttg.async_commit_group{{$}}
// FB: %[[MERGED:.*]] = ttg.async_commit_group tokens %[[A]], %[[DONE]]
// FB: %[[C:.*]] = ttg.async_copy_global_to_local
// FB: %[[CG:.*]] = ttg.async_commit_group tokens %[[C]]
// FB: ttg.async_wait %[[MERGED]]
// FB: ttg.local_load
// FB: ttg.local_load
// FB: ttg.async_wait %[[MERGED]], %[[CG]]
// PUB: hcug.async_wait{{.*}}num_inst = 1
// PUB-NEXT: ttg.local_barrier
// PUB: ttg.local_load
// PUB: ttg.local_load
// PUB: hcug.async_wait{{.*}}num_inst = 0
// PUB-NEXT: ttg.local_barrier
// LLVM: rocdl.global.load.lds
// LLVM: llvm.store {{.*}}!llvm.ptr<3>
// LLVM: rocdl.global.load.lds
// LLVM: rocdl.s.waitcnt -49167
// LLVM: rocdl.s.barrier
// LLVM: llvm.load {{.*}}!llvm.ptr<3>
// LLVM: rocdl.s.waitcnt -49168
// LLVM: rocdl.s.barrier
  tt.func public @merged_valid_and_fallback(%A: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %O: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %i = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32, #blocked>
    %baseA = tt.splat %A : !tt.ptr<f16> -> tensor<512x!tt.ptr<f16>, #blocked>
    %p = tt.addptr %baseA, %i : tensor<512x!tt.ptr<f16>, #blocked>, tensor<512xi32, #blocked>
    %baseO = tt.splat %O : !tt.ptr<f16> -> tensor<512x!tt.ptr<f16>, #blocked>
    %out = tt.addptr %baseO, %i : tensor<512x!tt.ptr<f16>, #blocked>, tensor<512xi32, #blocked>
    %a = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %c = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %ca = ttg.async_copy_global_to_local %p, %a : tensor<512x!tt.ptr<f16>, #blocked> -> <512xf16, #shared, #smem, mutable>
    %cb = ttg.async_copy_global_to_local %p, %b {isVolatile = true} : tensor<512x!tt.ptr<f16>, #blocked> -> <512xf16, #shared, #smem, mutable>
    %g = ttg.async_commit_group tokens %ca, %cb
    %cc = ttg.async_copy_global_to_local %p, %c : tensor<512x!tt.ptr<f16>, #blocked> -> <512xf16, #shared, #smem, mutable>
    %cg = ttg.async_commit_group tokens %cc
    %w = ttg.async_wait %g {num = 0 : i32}
    %av = ttg.local_load %a token %w : !ttg.memdesc<512xf16, #shared, #smem, mutable> -> tensor<512xf16, #blocked>
    tt.store %out, %av : tensor<512x!tt.ptr<f16>, #blocked>
    %bv = ttg.local_load %b token %w : !ttg.memdesc<512xf16, #shared, #smem, mutable> -> tensor<512xf16, #blocked>
    tt.store %out, %bv : tensor<512x!tt.ptr<f16>, #blocked>
    %wc = ttg.async_wait %g, %cg {num = 0 : i32}
    %cv = ttg.local_load %c token %wc : !ttg.memdesc<512xf16, #shared, #smem, mutable> -> tensor<512xf16, #blocked>
    tt.store %out, %cv : tensor<512x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"} {
// FB-LABEL: @tokenless_partial_empty_boundaries(
// PUB-LABEL: @tokenless_partial_empty_boundaries(
// LLVM-LABEL: llvm.func @tokenless_partial_empty_boundaries(
// FB: ttg.async_copy_global_to_local
// FB: tt.load
// FB: ttg.local_store
// FB: ttg.async_commit_group{{$}}
// FB: ttg.async_commit_group tokens
// FB: ttg.async_copy_global_to_local
// FB: ttg.async_wait {num = 1
// FB: ttg.async_wait {num = 2
// FB: ttg.async_wait {num = 0
// FB: tt.load
// FB: ttg.local_store
// FB: ttg.async_commit_group{{$}}
// FB: ttg.async_commit_group tokens
// FB: ttg.async_wait {num = 1
// PUB: hcug.async_wait{{.*}}num_inst = 1
// PUB-NEXT: ttg.local_barrier
// PUB: hcug.async_wait{{.*}}num_inst = 1
// PUB-NEXT: ttg.local_barrier
// PUB: hcug.async_wait{{.*}}num_inst = 0
// PUB-NEXT: ttg.local_barrier
// PUB: hcug.async_wait{{.*}}num_inst = 0
// PUB-NEXT: ttg.local_barrier
// LLVM: rocdl.global.load.lds
// LLVM: rocdl.global.load.lds
// LLVM: rocdl.s.waitcnt -49167
// LLVM: rocdl.s.barrier
// LLVM: rocdl.s.waitcnt -49168
// LLVM: rocdl.s.barrier
  tt.func public @tokenless_partial_empty_boundaries(%A: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %O: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %i = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32, #blocked>
    %baseA = tt.splat %A : !tt.ptr<f16> -> tensor<512x!tt.ptr<f16>, #blocked>
    %p = tt.addptr %baseA, %i : tensor<512x!tt.ptr<f16>, #blocked>, tensor<512xi32, #blocked>
    %baseO = tt.splat %O : !tt.ptr<f16> -> tensor<512x!tt.ptr<f16>, #blocked>
    %out = tt.addptr %baseO, %i : tensor<512x!tt.ptr<f16>, #blocked>, tensor<512xi32, #blocked>
    %a = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %c = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %d = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %ca = ttg.async_copy_global_to_local %p, %a : tensor<512x!tt.ptr<f16>, #blocked> -> <512xf16, #shared, #smem, mutable>
    %ag = ttg.async_commit_group tokens %ca
    %cb = ttg.async_copy_global_to_local %p, %b {isVolatile = true} : tensor<512x!tt.ptr<f16>, #blocked> -> <512xf16, #shared, #smem, mutable>
    %bg = ttg.async_commit_group tokens %cb
    %cc = ttg.async_copy_global_to_local %p, %c : tensor<512x!tt.ptr<f16>, #blocked> -> <512xf16, #shared, #smem, mutable>
    %cg = ttg.async_commit_group tokens %cc
    %w1 = ttg.async_wait {num = 1 : i32}
    %w2 = ttg.async_wait {num = 2 : i32}
    %av = ttg.local_load %a token %w2 : !ttg.memdesc<512xf16, #shared, #smem, mutable> -> tensor<512xf16, #blocked>
    tt.store %out, %av : tensor<512x!tt.ptr<f16>, #blocked>
    %bv = ttg.local_load %b token %w2 : !ttg.memdesc<512xf16, #shared, #smem, mutable> -> tensor<512xf16, #blocked>
    tt.store %out, %bv : tensor<512x!tt.ptr<f16>, #blocked>
    %w0 = ttg.async_wait {num = 0 : i32}
    %cv = ttg.local_load %c token %w0 : !ttg.memdesc<512xf16, #shared, #smem, mutable> -> tensor<512xf16, #blocked>
    tt.store %out, %cv : tensor<512x!tt.ptr<f16>, #blocked>
    %cd = ttg.async_copy_global_to_local %p, %d {isVolatile = true} : tensor<512x!tt.ptr<f16>, #blocked> -> <512xf16, #shared, #smem, mutable>
    %dg = ttg.async_commit_group tokens %cd
    %wd = ttg.async_wait {num = 1 : i32}
    %dv = ttg.local_load %d token %wd : !ttg.memdesc<512xf16, #shared, #smem, mutable> -> tensor<512xf16, #blocked>
    tt.store %out, %dv : tensor<512x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

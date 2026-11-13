// RUN: triton-opt %s --split-input-file --verify-each --tritonhcugpu-coalesce-async-copy="arch-generation-name=gfx936" | FileCheck %s --check-prefix=FB --enable-var-scope --implicit-check-not=ttg.async_copy_global_to_local
// RUN: triton-opt %s --split-input-file --verify-each --tritonhcugpu-coalesce-async-copy="arch-generation-name=gfx936" --test-tritonhcugpu-membar --tritonhcugpu-update-async-wait-count="arch-generation-name=gfx936" | FileCheck %s --check-prefix=PUB --enable-var-scope
// RUN: triton-opt %s --split-input-file --verify-each --tritonhcugpu-coalesce-async-copy="arch-generation-name=gfx936" --tritonhcugpu-update-async-wait-count="arch-generation-name=gfx936" --allocate-hcugpu-shared-memory --convert-triton-hcugpu-to-llvm="arch=gfx936" --convert-cf-to-llvm --convert-arith-to-llvm --canonicalize --cse | FileCheck %s --check-prefix=LLVM --enable-var-scope --implicit-check-not=rocdl.global.load.lds

// gfx936 GLOBAL direct-to-LDS legality and conservative register fallback.
// CPU compiler regression inputs; these are not runtime test kernels.
// Reuses AMD coalesce/update-wait and HCU async-copy-war public test shapes.
// Empty commits are zero-instruction IR boundaries, not hardware completion.
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"} {
// FB-LABEL: @low_mask(
// PUB-LABEL: @low_mask(
// LLVM-LABEL: llvm.func @low_mask(
// FB: %[[MASK:.*]] = arith.cmpi
// FB: %[[BUF:.*]] = ttg.local_alloc
// FB: %[[LOAD:.*]] = tt.load {{.*}}, %[[MASK]]
// FB-SAME: tt.load.async = false
// FB: ttg.local_store %[[LOAD]], %[[BUF]]
// FB: %[[DONE:.*]] = ttg.async_commit_group{{$}}
// FB: %[[GROUP:.*]] = ttg.async_commit_group tokens %[[DONE]]
// FB: %[[WAIT:.*]] = ttg.async_wait %[[GROUP]]
// FB: ttg.local_load %[[BUF]] token %[[WAIT]]
// FB-NOT: ttg.async_copy_global_to_local
// FB: tt.return
// PUB: ttg.local_store
// PUB: hcug.async_wait{{.*}}num_inst = 0
// PUB-NEXT: ttg.local_barrier
// PUB: ttg.local_load
// LLVM-NOT: rocdl.global.load.lds
// LLVM: llvm.load {{.*}}!llvm.ptr<1>
// LLVM: llvm.store {{.*}}!llvm.ptr<3>
// LLVM: rocdl.s.waitcnt -49168
// LLVM: rocdl.s.barrier
// LLVM: llvm.load {{.*}}!llvm.ptr<3>
  tt.func public @low_mask(%A: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %O: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %i = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32, #blocked>
    %baseA = tt.splat %A : !tt.ptr<f16> -> tensor<512x!tt.ptr<f16>, #blocked>
    %p = tt.addptr %baseA, %i : tensor<512x!tt.ptr<f16>, #blocked>, tensor<512xi32, #blocked>
    %baseO = tt.splat %O : !tt.ptr<f16> -> tensor<512x!tt.ptr<f16>, #blocked>
    %out = tt.addptr %baseO, %i : tensor<512x!tt.ptr<f16>, #blocked>, tensor<512xi32, #blocked>
    %one = arith.constant dense<1> : tensor<512xi32, #blocked>
    %zero = arith.constant dense<0> : tensor<512xi32, #blocked>
    %parity = arith.andi %i, %one : tensor<512xi32, #blocked>
    %mask = arith.cmpi eq, %parity, %zero : tensor<512xi32, #blocked>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %a0 = ttg.async_copy_global_to_local %p, %buf mask %mask : tensor<512x!tt.ptr<f16>, #blocked> -> <512xf16, #shared, #smem, mutable>
    %g0 = ttg.async_commit_group tokens %a0
    %w0 = ttg.async_wait %g0 {num = 0 : i32}
    %v0 = ttg.local_load %buf token %w0 : !ttg.memdesc<512xf16, #shared, #smem, mutable> -> tensor<512xf16, #blocked>
    tt.store %out, %v0 : tensor<512x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"} {
// FB-LABEL: @narrow_i8(
// PUB-LABEL: @narrow_i8(
// LLVM-LABEL: llvm.func @narrow_i8(
// FB: tt.load
// FB-SAME: tt.load.async = false
// FB: ttg.local_store
// FB-NOT: ttg.async_copy_global_to_local
// FB: tt.return
// PUB: hcug.async_wait{{.*}}num_inst = 0
// PUB-NEXT: ttg.local_barrier
// LLVM-NOT: rocdl.global.load.lds
// LLVM: llvm.load {{.*}}!llvm.ptr<1>
// LLVM: llvm.store {{.*}}!llvm.ptr<3>
// LLVM: rocdl.s.waitcnt -49168
// LLVM: rocdl.s.barrier
  tt.func public @narrow_i8(%A: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %O: !tt.ptr<i8> {tt.divisibility = 16 : i32}) {
    %i = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32, #blocked>
    %baseA = tt.splat %A : !tt.ptr<i8> -> tensor<256x!tt.ptr<i8>, #blocked>
    %p = tt.addptr %baseA, %i : tensor<256x!tt.ptr<i8>, #blocked>, tensor<256xi32, #blocked>
    %baseO = tt.splat %O : !tt.ptr<i8> -> tensor<256x!tt.ptr<i8>, #blocked>
    %out = tt.addptr %baseO, %i : tensor<256x!tt.ptr<i8>, #blocked>, tensor<256xi32, #blocked>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<256xi8, #shared, #smem, mutable>
    %a0 = ttg.async_copy_global_to_local %p, %buf : tensor<256x!tt.ptr<i8>, #blocked> -> <256xi8, #shared, #smem, mutable>
    %g0 = ttg.async_commit_group tokens %a0
    %w0 = ttg.async_wait %g0 {num = 0 : i32}
    %v0 = ttg.local_load %buf token %w0 : !ttg.memdesc<256xi8, #shared, #smem, mutable> -> tensor<256xi8, #blocked>
    tt.store %out, %v0 : tensor<256x!tt.ptr<i8>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"} {
// FB-LABEL: @other7_reuse(
// PUB-LABEL: @other7_reuse(
// LLVM-LABEL: llvm.func @other7_reuse(
// FB: %[[OTHER:.*]] = arith.constant dense<7.000000e+00>
// FB: %[[MASK:.*]] = arith.cmpi
// FB: %[[BUF:.*]] = ttg.local_alloc
// FB: %[[LOAD:.*]] = tt.load {{.*}}, %[[MASK]], %[[OTHER]]
// FB-SAME: tt.load.async = false
// FB: ttg.local_store %[[LOAD]], %[[BUF]]
// FB: %[[DONE:.*]] = ttg.async_commit_group{{$}}
// FB: %[[GROUP:.*]] = ttg.async_commit_group tokens %[[DONE]]
// FB: %[[WAIT:.*]] = ttg.async_wait %[[GROUP]]
// FB: ttg.local_load %[[BUF]] token %[[WAIT]]
// FB-NOT: ttg.async_copy_global_to_local
// FB: tt.return
// PUB: ttg.local_store
// PUB: hcug.async_wait{{.*}}num_inst = 0
// PUB-NEXT: ttg.local_barrier
// PUB: ttg.local_load
// LLVM-NOT: rocdl.global.load.lds
// LLVM: llvm.load {{.*}}!llvm.ptr<1>
// LLVM: llvm.store {{.*}}!llvm.ptr<3>
// LLVM: rocdl.s.waitcnt -49168
// LLVM: rocdl.s.barrier
// LLVM: llvm.load {{.*}}!llvm.ptr<3>
// PUB: tt.store
// PUB: tt.load
// PUB-NEXT: ttg.local_barrier
// PUB-NEXT: ttg.local_store
// PUB: hcug.async_wait{{.*}}num_inst = 0
// PUB-NEXT: ttg.local_barrier
// LLVM: rocdl.s.barrier
// LLVM: llvm.store {{.*}}!llvm.ptr<3>
// LLVM: rocdl.s.waitcnt -49168
// LLVM: rocdl.s.barrier
  tt.func public @other7_reuse(%A: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %O: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %i = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32, #blocked>
    %baseA = tt.splat %A : !tt.ptr<f16> -> tensor<512x!tt.ptr<f16>, #blocked>
    %p = tt.addptr %baseA, %i : tensor<512x!tt.ptr<f16>, #blocked>, tensor<512xi32, #blocked>
    %baseO = tt.splat %O : !tt.ptr<f16> -> tensor<512x!tt.ptr<f16>, #blocked>
    %out = tt.addptr %baseO, %i : tensor<512x!tt.ptr<f16>, #blocked>, tensor<512xi32, #blocked>
    %limit = arith.constant dense<256> : tensor<512xi32, #blocked>
    %mask = arith.cmpi slt, %i, %limit : tensor<512xi32, #blocked>
    %other = arith.constant dense<7.0> : tensor<512xf16, #blocked>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %a0 = ttg.async_copy_global_to_local %p, %buf mask %mask other %other : tensor<512x!tt.ptr<f16>, #blocked> -> <512xf16, #shared, #smem, mutable>
    %g0 = ttg.async_commit_group tokens %a0
    %w0 = ttg.async_wait %g0 {num = 0 : i32}
    %v0 = ttg.local_load %buf token %w0 : !ttg.memdesc<512xf16, #shared, #smem, mutable> -> tensor<512xf16, #blocked>
    tt.store %out, %v0 : tensor<512x!tt.ptr<f16>, #blocked>
    %a1 = ttg.async_copy_global_to_local %p, %buf mask %mask other %other : tensor<512x!tt.ptr<f16>, #blocked> -> <512xf16, #shared, #smem, mutable>
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
// FB-LABEL: @volatile(
// PUB-LABEL: @volatile(
// LLVM-LABEL: llvm.func @volatile(
// FB: %[[BUF:.*]] = ttg.local_alloc
// FB: %[[LOAD:.*]] = tt.load
// FB-SAME: isVolatile = true
// FB-SAME: tt.load.async = false
// FB: ttg.local_store %[[LOAD]], %[[BUF]]
// FB: %[[DONE:.*]] = ttg.async_commit_group{{$}}
// FB: %[[GROUP:.*]] = ttg.async_commit_group tokens %[[DONE]]
// FB: %[[WAIT:.*]] = ttg.async_wait %[[GROUP]]
// FB: ttg.local_load %[[BUF]] token %[[WAIT]]
// FB-NOT: ttg.async_copy_global_to_local
// FB: tt.return
// PUB: ttg.local_store
// PUB: hcug.async_wait{{.*}}num_inst = 0
// PUB-NEXT: ttg.local_barrier
// PUB: ttg.local_load
// LLVM-NOT: rocdl.global.load.lds
// LLVM: llvm.load {{.*}}!llvm.ptr<1>
// LLVM: llvm.store {{.*}}!llvm.ptr<3>
// LLVM: rocdl.s.waitcnt -49168
// LLVM: rocdl.s.barrier
// LLVM: llvm.load {{.*}}!llvm.ptr<3>
  tt.func public @volatile(%A: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %O: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %i = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32, #blocked>
    %baseA = tt.splat %A : !tt.ptr<f16> -> tensor<512x!tt.ptr<f16>, #blocked>
    %p = tt.addptr %baseA, %i : tensor<512x!tt.ptr<f16>, #blocked>, tensor<512xi32, #blocked>
    %baseO = tt.splat %O : !tt.ptr<f16> -> tensor<512x!tt.ptr<f16>, #blocked>
    %out = tt.addptr %baseO, %i : tensor<512x!tt.ptr<f16>, #blocked>, tensor<512xi32, #blocked>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %a0 = ttg.async_copy_global_to_local %p, %buf {isVolatile = true} : tensor<512x!tt.ptr<f16>, #blocked> -> <512xf16, #shared, #smem, mutable>
    %g0 = ttg.async_commit_group tokens %a0
    %w0 = ttg.async_wait %g0 {num = 0 : i32}
    %v0 = ttg.local_load %buf token %w0 : !ttg.memdesc<512xf16, #shared, #smem, mutable> -> tensor<512xf16, #blocked>
    tt.store %out, %v0 : tensor<512x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"} {
// FB-LABEL: @strided_subview(
// FB: %[[VIEW:.*]] = ttg.memdesc_subslice
// FB: %[[LOAD:.*]] = tt.load
// FB-SAME: tt.load.async = false
// FB: ttg.local_store %[[LOAD]], %[[VIEW]]
// FB-NOT: ttg.async_copy_global_to_local
// FB: tt.return
// PUB-LABEL: @strided_subview(
// PUB: ttg.local_store
// PUB: hcug.async_wait{{.*}}num_inst = 0
// PUB-NEXT: ttg.local_barrier
// PUB: ttg.local_load
// LLVM-LABEL: llvm.func @strided_subview(
// LLVM-NOT: rocdl.global.load.lds
// LLVM: llvm.store {{.*}}!llvm.ptr<3>
// LLVM: rocdl.s.waitcnt -49168
// LLVM: rocdl.s.barrier
// LLVM: llvm.load {{.*}}!llvm.ptr<3>
  tt.func public @strided_subview(%A: tensor<32x32x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<[64, 16]> : tensor<2xi32>}, %O: tensor<32x32x!tt.ptr<f16>, #blocked>) {
    %root = ttg.local_alloc : () -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    %view = ttg.memdesc_subslice %root [0, 32] : !ttg.memdesc<32x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable, 32x64>
    %copy = ttg.async_copy_global_to_local %A, %view : tensor<32x32x!tt.ptr<f16>, #blocked> -> <32x32xf16, #shared, #smem, mutable, 32x64>
    %group = ttg.async_commit_group tokens %copy
    %wait = ttg.async_wait %group {num = 0 : i32}
    %v = ttg.local_load %view token %wait : !ttg.memdesc<32x32xf16, #shared, #smem, mutable, 32x64> -> tensor<32x32xf16, #blocked>
    tt.store %O, %v : tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

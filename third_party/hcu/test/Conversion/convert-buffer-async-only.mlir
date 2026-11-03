// RUN: triton-opt %s --tritonhcugpu-convert-buffer-ops="arch-generation-name=gfx936" --verify-each | FileCheck %s --check-prefix=FULL
// RUN: triton-opt %s --tritonhcugpu-convert-buffer-ops="arch-generation-name=gfx936 async-only=false" --verify-each | FileCheck %s --check-prefix=FULL
// RUN: triton-opt %s --tritonhcugpu-convert-buffer-ops="arch-generation-name=gfx936 async-only=true" --verify-each | FileCheck %s --check-prefix=ASYNC
//
// Reuse the existing all-memory fixture. The omitted/default-false option
// retains conversion of every supported memory family. The early async-only
// mode converts the producer but leaves all synchronous families intact.
// Its commit must still consume the converted producer's exact result.
//
// FULL-LABEL: @all_memory(
// FULL: %[[COPY:[a-zA-Z0-9_]+]] = hcug.buffer_load_to_local
// FULL: ttg.async_commit_group tokens %[[COPY]]
// FULL: %[[LOAD:[a-zA-Z0-9_]+]] = hcug.buffer_load
// FULL: hcug.buffer_store %[[LOAD]],
// FULL: hcug.buffer_atomic_rmw add, acq_rel, gpu,
// FULL: hcug.buffer_atomic_cas acq_rel, gpu,
// FULL: tt.return
//
// ASYNC-LABEL: @all_memory(
// ASYNC-NOT: hcug.buffer_load {{[^t]}}
// ASYNC: %[[COPY:[a-zA-Z0-9_]+]] = hcug.buffer_load_to_local
// ASYNC: ttg.async_commit_group tokens %[[COPY]]
// ASYNC: %[[LOAD:[a-zA-Z0-9_]+]] = tt.load
// ASYNC: tt.store {{.*}}, %[[LOAD]] :
// ASYNC: tt.atomic_rmw add, acq_rel, gpu,
// ASYNC: tt.atomic_cas acq_rel, gpu,
// ASYNC: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [8], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"} {
 tt.func public @all_memory(%A: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %B: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
  %r = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32, #blocked>
  %a = tt.splat %A : !tt.ptr<i32> -> tensor<512x!tt.ptr<i32>, #blocked>
  %b = tt.splat %B : !tt.ptr<i32> -> tensor<512x!tt.ptr<i32>, #blocked>
  %ap = tt.addptr %a, %r : tensor<512x!tt.ptr<i32>, #blocked>, tensor<512xi32, #blocked>
  %bp = tt.addptr %b, %r : tensor<512x!tt.ptr<i32>, #blocked>, tensor<512xi32, #blocked>
  %zero = arith.constant dense<0> : tensor<512xi32, #blocked>
  %one = arith.constant dense<1> : tensor<512xi32, #blocked>
  %dest = ttg.local_alloc : () -> !ttg.memdesc<512xi32, #shared, #smem, mutable>
  %copy = ttg.async_copy_global_to_local %ap, %dest {contiguity = 1 : i32} : tensor<512x!tt.ptr<i32>, #blocked> -> <512xi32, #shared, #smem, mutable>
  %commit = ttg.async_commit_group tokens %copy
  %sync = tt.load %ap {tt.load.async = false} : tensor<512x!tt.ptr<i32>, #blocked>
  tt.store %bp, %sync : tensor<512x!tt.ptr<i32>, #blocked>
  %old = tt.atomic_rmw add, acq_rel, gpu, %bp, %one : (tensor<512x!tt.ptr<i32>, #blocked>, tensor<512xi32, #blocked>) -> tensor<512xi32, #blocked>
  %cas = tt.atomic_cas acq_rel, gpu, %bp, %zero, %one : (tensor<512x!tt.ptr<i32>, #blocked>, tensor<512xi32, #blocked>, tensor<512xi32, #blocked>) -> tensor<512xi32, #blocked>
  tt.return
 }
}

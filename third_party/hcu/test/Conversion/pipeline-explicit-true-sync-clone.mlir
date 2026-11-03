// RUN: triton-opt %s --verify-each --tritonhcugpu-schedule-loops="num_stages=3" --tritonhcugpu-pipeline="use_async_copy=false" --canonicalize --triton-tle-lower-async-load --tritonhcugpu-coalesce-async-copy="arch-generation-name=gfx936" | FileCheck %s --check-prefix=CLONE --implicit-check-not="tt.load.async = true" --implicit-check-not=ttg.async_copy_global_to_local
// Pipeline-policy regression; input True is not hand-rewritten False.
// The alternating K mask prevents a >=32bit async load. Existing single-dot
// shell must actually create a stream clone before this counts as coverage.
// The pipeline must select the synchronous clone before TLE lowering.
// CLONE-LABEL: @true_ineligible_clone_stays_sync(
// CLONE-NOT: tt.load.async = true
// CLONE-NOT: ttg.async_copy_global_to_local
// CLONE: tt.load
// CLONE-SAME: tt.load.async = false
// CLONE: ttg.local_store
// CLONE: ttg.local_load
// CLONE: tt.dot
// CLONE-NOT: tt.load.async = true
// CLONE-NOT: ttg.async_copy_global_to_local
// CLONE: tt.return
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 1, tilesPerWarp = [1, 1]}>
#da = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#db = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.target" = "hip:gfx936"} {
  tt.func public @true_ineligible_clone_stays_sync(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %b: tensor<32x32xbf16, #db>) -> tensor<32x32xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : i32
    %zero = arith.constant dense<0.0> : tensor<32x32xf32, #mma>
    %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %row = tt.expand_dims %r {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %idx = tt.broadcast %row : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %base = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>, #blocked>
    %ptr = tt.addptr %base, %idx : tensor<32x32x!tt.ptr<bf16>, #blocked>, tensor<32x32xi32, #blocked>
    %out = scf.for %i = %c0 to %c16 step %c1 iter_args(%acc = %zero) -> tensor<32x32xf32, #mma> : i32 {
      %one = arith.constant dense<1> : tensor<32x32xi32, #blocked>
      %zero_i = arith.constant dense<0> : tensor<32x32xi32, #blocked>
      %parity = arith.andi %idx, %one : tensor<32x32xi32, #blocked>
      %mask = arith.cmpi eq, %parity, %zero_i : tensor<32x32xi32, #blocked>
      %a = tt.load %ptr, %mask {tt.load.async = true} : tensor<32x32x!tt.ptr<bf16>, #blocked>
      %ad = ttg.convert_layout %a : tensor<32x32xbf16, #blocked> -> tensor<32x32xbf16, #da>
      %d = tt.dot %ad, %b, %acc : tensor<32x32xbf16, #da> * tensor<32x32xbf16, #db> -> tensor<32x32xf32, #mma>
      scf.yield %d : tensor<32x32xf32, #mma>
    }
    tt.return %out : tensor<32x32xf32, #mma>
  }
}

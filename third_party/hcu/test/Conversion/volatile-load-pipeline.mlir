// RUN: triton-opt %s --tritonhcugpu-schedule-loops="num_stages=3" --tritonhcugpu-pipeline="use_async_copy=false" --triton-tle-lower-async-load --canonicalize --verify-each | FileCheck %s --check-prefixes=COMMON,SYNC
// RUN: triton-opt %s --tritonhcugpu-schedule-loops="num_stages=3" --tritonhcugpu-pipeline="use_async_copy=true" --triton-tle-lower-async-load --canonicalize --verify-each | FileCheck %s --check-prefixes=COMMON,ASYNC
// One volatile load is tied to each of the 16 dot iterations. Only the ordinary
// B load may be pipelined: 14 main iterations plus two tail iterations.
// COMMON-LABEL: tt.func public @mixed_volatile_and_normal
// COMMON-NOT: isVolatile = true
// SYNC-COUNT-2: ttg.local_store
// ASYNC-COUNT-2: ttg.async_copy_global_to_local
// COMMON-NOT: isVolatile = true
// COMMON: scf.for
// COMMON: tt.load {{.*}}isVolatile = true
// COMMON-NOT: isVolatile = true
// COMMON: tt.dot
// COMMON-NOT: isVolatile = true
// COMMON: scf.yield
// COMMON-COUNT-2: tt.load {{.*}}isVolatile = true
// COMMON-NOT: isVolatile = true
// COMMON: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 1, tilesPerWarp = [1, 1]}>
#da = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#db = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.target" = "hip:gfx936"} {
  tt.func public @mixed_volatile_and_normal(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %argB: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) -> tensor<32x32xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : i32
    %zero = arith.constant dense<0.0> : tensor<32x32xf32, #mma>
    %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %row = tt.expand_dims %r {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %idx = tt.broadcast %row : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %base = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>, #blocked>
    %ptr = tt.addptr %base, %idx : tensor<32x32x!tt.ptr<bf16>, #blocked>, tensor<32x32xi32, #blocked>
    %baseB = tt.splat %argB : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>, #blocked>
    %ptrB = tt.addptr %baseB, %idx : tensor<32x32x!tt.ptr<bf16>, #blocked>, tensor<32x32xi32, #blocked>
    %out = scf.for %i = %c0 to %c16 step %c1 iter_args(%acc = %zero) -> tensor<32x32xf32, #mma> : i32 {
      %a = tt.load %ptr {isVolatile = true, tt.load.async = true} : tensor<32x32x!tt.ptr<bf16>, #blocked>
      %ad = ttg.convert_layout %a : tensor<32x32xbf16, #blocked> -> tensor<32x32xbf16, #da>
      %bLoad = tt.load %ptrB : tensor<32x32x!tt.ptr<bf16>, #blocked>
      %b = ttg.convert_layout %bLoad : tensor<32x32xbf16, #blocked> -> tensor<32x32xbf16, #db>
      %d = tt.dot %ad, %b, %acc : tensor<32x32xbf16, #da> * tensor<32x32xbf16, #db> -> tensor<32x32xf32, #mma>
      scf.yield %d : tensor<32x32xf32, #mma>
    }
    tt.return %out : tensor<32x32xf32, #mma>
  }
}

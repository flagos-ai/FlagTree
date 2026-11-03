// RUN: triton-opt %s --split-input-file --verify-each --tritonhcugpu-update-async-wait-count="arch-generation-name=gfx936" --allocate-hcugpu-shared-memory --convert-triton-hcugpu-to-llvm="arch=gfx936" --convert-cf-to-llvm --convert-arith-to-llvm --canonicalize --cse | FileCheck %s --enable-var-scope

// Address-only regression: gfx936 BUFFER direct-to-LDS, 16B packets.
// Ordinary legal Swizzled layout; no private pipeline/native/test passes.
// Positive cases rebuild the explicit varying DAG at the selected physical
// lane, then add the uniform splat in wrapping i32 element-offset arithmetic.
// Unsupported inner leaves keep factoring U but retain the data shuffle;
// unsupported outer expressions retain the original whole-offset shuffle.
// Mask ballot/false-lane OOB selection and wait/publication remain intact.
// These checks are compiler IR checks, not numerical/GPU validation.

// CHECK-LABEL: llvm.func @masked_uniform_i32(
// CHECK-SAME: %[[U:[a-zA-Z0-9_]+]]: i32 {tt.divisibility = 8 : i32}
// CHECK-NOT: rocdl.ds_bpermute
// CHECK: %[[OOB:[a-zA-Z0-9_]+]] = llvm.mlir.constant(-2147483648 : i32) : i32
// CHECK-NOT: rocdl.ds_bpermute
// CHECK: %[[WIDTH:[a-zA-Z0-9_]+]] = llvm.mlir.constant(16 : i32) : i32
// CHECK-NOT: rocdl.ds_bpermute
// CHECK: %[[TWO:[a-zA-Z0-9_]+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NOT: rocdl.ds_bpermute
// CHECK: %[[BALLOT:[a-zA-Z0-9_]+]] = rocdl.ballot {{%[a-zA-Z0-9_]+}} : i64
// CHECK-NOT: rocdl.ds_bpermute
// CHECK: %[[SHIFT:[a-zA-Z0-9_]+]] = llvm.lshr %[[BALLOT]], {{%[a-zA-Z0-9_]+}} : i64
// CHECK-NOT: rocdl.ds_bpermute
// CHECK: %[[PRED:[a-zA-Z0-9_]+]] = llvm.trunc %[[SHIFT]] : i64 to i1
// CHECK-NOT: rocdl.ds_bpermute
// CHECK: %[[SUM:[a-zA-Z0-9_]+]] = llvm.add {{%[a-zA-Z0-9_]+}}, %[[U]] : i32
// CHECK-NOT: rocdl.ds_bpermute
// CHECK: %[[BYTES:[a-zA-Z0-9_]+]] = llvm.mul %[[SUM]], %[[TWO]] : i32
// CHECK-NOT: rocdl.ds_bpermute
// CHECK: %[[VOFF:[a-zA-Z0-9_]+]] = llvm.select %[[PRED]], %[[BYTES]], %[[OOB]] : i1, i32
// CHECK-NOT: rocdl.ds_bpermute
// CHECK: rocdl.raw.ptr.buffer.load.lds {{%[a-zA-Z0-9_]+}}, {{%[a-zA-Z0-9_]+}}, %[[WIDTH]], %[[VOFF]],
// CHECK-NOT: rocdl.ds_bpermute
// CHECK: rocdl.s.waitcnt -49168
// CHECK-NOT: rocdl.ds_bpermute
// CHECK: rocdl.s.barrier
// CHECK-NOT: rocdl.ds_bpermute
// CHECK: llvm.load {{.*}}!llvm.ptr<3>
// CHECK-NOT: rocdl.ds_bpermute
// CHECK: llvm.return

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [1, 0]}>
#sM = #ttg.slice<{dim = 1, parent = #blocked}>
#sK = #ttg.slice<{dim = 0, parent = #blocked}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx936", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @masked_uniform_i32(%A: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %O: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %U: i32 {tt.divisibility = 8 : i32}) {
    %m = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #sM>
    %k = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #sK>
    %m1 = tt.expand_dims %m {axis = 1 : i32} : tensor<256xi32, #sM> -> tensor<256x1xi32, #blocked>
    %k1 = tt.expand_dims %k {axis = 0 : i32} : tensor<16xi32, #sK> -> tensor<1x16xi32, #blocked>
    %mf = tt.broadcast %m1 : tensor<256x1xi32, #blocked> -> tensor<256x16xi32, #blocked>
    %kf = tt.broadcast %k1 : tensor<1x16xi32, #blocked> -> tensor<256x16xi32, #blocked>
    %pitch = arith.constant dense<1024> : tensor<256x16xi32, #blocked>
    %sixteen = arith.constant dense<16> : tensor<256x16xi32, #blocked>
    %eight = arith.constant dense<8> : tensor<256x16xi32, #blocked>
    %zero = arith.constant dense<0> : tensor<256x16xi32, #blocked>
    %row = arith.muli %mf, %pitch : tensor<256x16xi32, #blocked>
    %v0 = arith.addi %row, %kf : tensor<256x16xi32, #blocked>
    %us = tt.splat %U : i32 -> tensor<256x16xi32, #blocked>
    %sum = arith.addi %us, %v0 : tensor<256x16xi32, #blocked>
    %kb = arith.andi %kf, %eight : tensor<256x16xi32, #blocked>
    %mask = arith.cmpi eq, %kb, %zero : tensor<256x16xi32, #blocked>
    %init = arith.constant dense<-3.000000e+00> : tensor<256x16xbf16, #blocked>
    %dst = ttg.local_alloc %init : (tensor<256x16xbf16, #blocked>) -> !ttg.memdesc<256x16xbf16, #shared, #smem, mutable>
    ttg.local_barrier
    %t = hcug.buffer_load_to_local %A[%sum] mask = %mask into %dst {contiguity = 8 : i32} : <bf16>[tensor<256x16xi32, #blocked>] -> <256x16xbf16, #shared, #smem, mutable>
    %w = ttg.async_wait %t {num = 0 : i32}
    ttg.local_barrier
    %got = ttg.local_load %dst token %w : !ttg.memdesc<256x16xbf16, #shared, #smem, mutable> -> tensor<256x16xbf16, #blocked>
    %orow = arith.muli %mf, %sixteen : tensor<256x16xi32, #blocked>
    %oi = arith.addi %orow, %kf : tensor<256x16xi32, #blocked>
    %op = tt.splat %O : !tt.ptr<bf16> -> tensor<256x16x!tt.ptr<bf16>, #blocked>
    %optrs = tt.addptr %op, %oi : tensor<256x16x!tt.ptr<bf16>, #blocked>, tensor<256x16xi32, #blocked>
    tt.store %optrs, %got : tensor<256x16x!tt.ptr<bf16>, #blocked>
    ttg.local_dealloc %dst : !ttg.memdesc<256x16xbf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// CHECK-LABEL: llvm.func @trunc_wrap_uniform_i64(
// CHECK-SAME: %[[U:[a-zA-Z0-9_]+]]: i64 {tt.divisibility = 8 : i32}
// CHECK-NOT: rocdl.ds_bpermute
// CHECK: %[[OOB:[a-zA-Z0-9_]+]] = llvm.mlir.constant(-2147483648 : i32) : i32
// CHECK-NOT: rocdl.ds_bpermute
// CHECK: %[[WIDTH:[a-zA-Z0-9_]+]] = llvm.mlir.constant(16 : i32) : i32
// CHECK-NOT: rocdl.ds_bpermute
// CHECK: %[[TWO:[a-zA-Z0-9_]+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NOT: rocdl.ds_bpermute
// CHECK: %[[U32:[a-zA-Z0-9_]+]] = llvm.trunc %[[U]] : i64 to i32
// CHECK-NOT: rocdl.ds_bpermute
// CHECK: %[[BALLOT:[a-zA-Z0-9_]+]] = rocdl.ballot {{%[a-zA-Z0-9_]+}} : i64
// CHECK-NOT: rocdl.ds_bpermute
// CHECK: %[[SHIFT:[a-zA-Z0-9_]+]] = llvm.lshr %[[BALLOT]], {{%[a-zA-Z0-9_]+}} : i64
// CHECK-NOT: rocdl.ds_bpermute
// CHECK: %[[PRED:[a-zA-Z0-9_]+]] = llvm.trunc %[[SHIFT]] : i64 to i1
// CHECK-NOT: rocdl.ds_bpermute
// CHECK: %[[SUM:[a-zA-Z0-9_]+]] = llvm.add {{%[a-zA-Z0-9_]+}}, %[[U32]] : i32
// CHECK-NOT: rocdl.ds_bpermute
// CHECK: %[[BYTES:[a-zA-Z0-9_]+]] = llvm.mul %[[SUM]], %[[TWO]] : i32
// CHECK-NOT: rocdl.ds_bpermute
// CHECK: %[[VOFF:[a-zA-Z0-9_]+]] = llvm.select %[[PRED]], %[[BYTES]], %[[OOB]] : i1, i32
// CHECK-NOT: rocdl.ds_bpermute
// CHECK: rocdl.raw.ptr.buffer.load.lds {{%[a-zA-Z0-9_]+}}, {{%[a-zA-Z0-9_]+}}, %[[WIDTH]], %[[VOFF]],
// CHECK-NOT: rocdl.ds_bpermute
// CHECK: rocdl.s.waitcnt -49168
// CHECK-NOT: rocdl.ds_bpermute
// CHECK: rocdl.s.barrier
// CHECK-NOT: rocdl.ds_bpermute
// CHECK: llvm.load {{.*}}!llvm.ptr<3>
// CHECK-NOT: rocdl.ds_bpermute
// CHECK: llvm.return

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [1, 0]}>
#sM = #ttg.slice<{dim = 1, parent = #blocked}>
#sK = #ttg.slice<{dim = 0, parent = #blocked}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx936", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @trunc_wrap_uniform_i64(%A: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %O: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %U: i64 {tt.divisibility = 8 : i32}) {
    %m = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #sM>
    %k = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #sK>
    %m1 = tt.expand_dims %m {axis = 1 : i32} : tensor<256xi32, #sM> -> tensor<256x1xi32, #blocked>
    %k1 = tt.expand_dims %k {axis = 0 : i32} : tensor<16xi32, #sK> -> tensor<1x16xi32, #blocked>
    %mf = tt.broadcast %m1 : tensor<256x1xi32, #blocked> -> tensor<256x16xi32, #blocked>
    %kf = tt.broadcast %k1 : tensor<1x16xi32, #blocked> -> tensor<256x16xi32, #blocked>
    %pitch = arith.constant dense<1024> : tensor<256x16xi32, #blocked>
    %sixteen = arith.constant dense<16> : tensor<256x16xi32, #blocked>
    %eight = arith.constant dense<8> : tensor<256x16xi32, #blocked>
    %zero = arith.constant dense<0> : tensor<256x16xi32, #blocked>
    %row = arith.muli %mf, %pitch : tensor<256x16xi32, #blocked>
    %v0 = arith.addi %row, %kf : tensor<256x16xi32, #blocked>
    %vwrap = arith.addi %v0, %sixteen : tensor<256x16xi32, #blocked>
    %v64 = arith.extui %vwrap : tensor<256x16xi32, #blocked> to tensor<256x16xi64, #blocked>
    %us = tt.splat %U : i64 -> tensor<256x16xi64, #blocked>
    %sum = arith.addi %v64, %us : tensor<256x16xi64, #blocked>
    %narrow = arith.trunci %sum : tensor<256x16xi64, #blocked> to tensor<256x16xi32, #blocked>
    %kb = arith.andi %kf, %eight : tensor<256x16xi32, #blocked>
    %mask = arith.cmpi eq, %kb, %zero : tensor<256x16xi32, #blocked>
    %init = arith.constant dense<-3.000000e+00> : tensor<256x16xbf16, #blocked>
    %dst = ttg.local_alloc %init : (tensor<256x16xbf16, #blocked>) -> !ttg.memdesc<256x16xbf16, #shared, #smem, mutable>
    ttg.local_barrier
    %t = hcug.buffer_load_to_local %A[%narrow] into %dst {contiguity = 8 : i32} : <bf16>[tensor<256x16xi32, #blocked>] -> <256x16xbf16, #shared, #smem, mutable>
    %w = ttg.async_wait %t {num = 0 : i32}
    ttg.local_barrier
    %got = ttg.local_load %dst token %w : !ttg.memdesc<256x16xbf16, #shared, #smem, mutable> -> tensor<256x16xbf16, #blocked>
    %orow = arith.muli %mf, %sixteen : tensor<256x16xi32, #blocked>
    %oi = arith.addi %orow, %kf : tensor<256x16xi32, #blocked>
    %op = tt.splat %O : !tt.ptr<bf16> -> tensor<256x16x!tt.ptr<bf16>, #blocked>
    %optrs = tt.addptr %op, %oi : tensor<256x16x!tt.ptr<bf16>, #blocked>, tensor<256x16xi32, #blocked>
    tt.store %optrs, %got : tensor<256x16x!tt.ptr<bf16>, #blocked>
    ttg.local_dealloc %dst : !ttg.memdesc<256x16xbf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// CHECK-LABEL: llvm.func @inner_stride_fallback(
// CHECK-SAME: %[[U:[a-zA-Z0-9_]+]]: i32 {tt.divisibility = 8 : i32}
// CHECK-SAME: %[[STRIDE:[a-zA-Z0-9_]+]]: i32 {tt.divisibility = 16 : i32}
// CHECK: %[[OOB:[a-zA-Z0-9_]+]] = llvm.mlir.constant(-2147483648 : i32) : i32
// CHECK: %[[WIDTH:[a-zA-Z0-9_]+]] = llvm.mlir.constant(16 : i32) : i32
// CHECK: %[[TWO:[a-zA-Z0-9_]+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK: %[[ROW:[a-zA-Z0-9_]+]] = llvm.mul {{%[a-zA-Z0-9_]+}}, %[[STRIDE]] : i32
// CHECK: %[[VARYING:[a-zA-Z0-9_]+]] = llvm.add %[[ROW]], {{%[a-zA-Z0-9_]+}} : i32
// CHECK: %[[SHUFFLED:[a-zA-Z0-9_]+]] = rocdl.ds_bpermute {{%[a-zA-Z0-9_]+}}, %[[VARYING]] : (i32, i32) -> i32
// CHECK: %[[BALLOT:[a-zA-Z0-9_]+]] = rocdl.ballot {{%[a-zA-Z0-9_]+}} : i64
// CHECK: %[[SHIFT:[a-zA-Z0-9_]+]] = llvm.lshr %[[BALLOT]], {{%[a-zA-Z0-9_]+}} : i64
// CHECK: %[[PRED:[a-zA-Z0-9_]+]] = llvm.trunc %[[SHIFT]] : i64 to i1
// CHECK: %[[SUM:[a-zA-Z0-9_]+]] = llvm.add %[[SHUFFLED]], %[[U]] : i32
// CHECK: %[[BYTES:[a-zA-Z0-9_]+]] = llvm.mul %[[SUM]], %[[TWO]] : i32
// CHECK: %[[VOFF:[a-zA-Z0-9_]+]] = llvm.select %[[PRED]], %[[BYTES]], %[[OOB]] : i1, i32
// CHECK: rocdl.raw.ptr.buffer.load.lds {{%[a-zA-Z0-9_]+}}, {{%[a-zA-Z0-9_]+}}, %[[WIDTH]], %[[VOFF]],
// CHECK: rocdl.s.waitcnt -49168
// CHECK: rocdl.s.barrier
// CHECK: llvm.load {{.*}}!llvm.ptr<3>
// CHECK: llvm.return

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [1, 0]}>
#sM = #ttg.slice<{dim = 1, parent = #blocked}>
#sK = #ttg.slice<{dim = 0, parent = #blocked}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx936", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @inner_stride_fallback(%A: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %O: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %U: i32 {tt.divisibility = 8 : i32}, %stride: i32 {tt.divisibility = 16 : i32}) {
    %m = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #sM>
    %k = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #sK>
    %m1 = tt.expand_dims %m {axis = 1 : i32} : tensor<256xi32, #sM> -> tensor<256x1xi32, #blocked>
    %k1 = tt.expand_dims %k {axis = 0 : i32} : tensor<16xi32, #sK> -> tensor<1x16xi32, #blocked>
    %mf = tt.broadcast %m1 : tensor<256x1xi32, #blocked> -> tensor<256x16xi32, #blocked>
    %kf = tt.broadcast %k1 : tensor<1x16xi32, #blocked> -> tensor<256x16xi32, #blocked>
    %pitch = arith.constant dense<1024> : tensor<256x16xi32, #blocked>
    %sixteen = arith.constant dense<16> : tensor<256x16xi32, #blocked>
    %eight = arith.constant dense<8> : tensor<256x16xi32, #blocked>
    %zero = arith.constant dense<0> : tensor<256x16xi32, #blocked>
    %stridef = tt.splat %stride : i32 -> tensor<256x16xi32, #blocked>
    %row = arith.muli %mf, %stridef : tensor<256x16xi32, #blocked>
    %v0 = arith.addi %row, %kf : tensor<256x16xi32, #blocked>
    %us = tt.splat %U : i32 -> tensor<256x16xi32, #blocked>
    %sum = arith.addi %us, %v0 : tensor<256x16xi32, #blocked>
    %kb = arith.andi %kf, %eight : tensor<256x16xi32, #blocked>
    %mask = arith.cmpi eq, %kb, %zero : tensor<256x16xi32, #blocked>
    %init = arith.constant dense<-3.000000e+00> : tensor<256x16xbf16, #blocked>
    %dst = ttg.local_alloc %init : (tensor<256x16xbf16, #blocked>) -> !ttg.memdesc<256x16xbf16, #shared, #smem, mutable>
    ttg.local_barrier
    %t = hcug.buffer_load_to_local %A[%sum] into %dst {contiguity = 8 : i32} : <bf16>[tensor<256x16xi32, #blocked>] -> <256x16xbf16, #shared, #smem, mutable>
    %w = ttg.async_wait %t {num = 0 : i32}
    ttg.local_barrier
    %got = ttg.local_load %dst token %w : !ttg.memdesc<256x16xbf16, #shared, #smem, mutable> -> tensor<256x16xbf16, #blocked>
    %orow = arith.muli %mf, %sixteen : tensor<256x16xi32, #blocked>
    %oi = arith.addi %orow, %kf : tensor<256x16xi32, #blocked>
    %op = tt.splat %O : !tt.ptr<bf16> -> tensor<256x16x!tt.ptr<bf16>, #blocked>
    %optrs = tt.addptr %op, %oi : tensor<256x16x!tt.ptr<bf16>, #blocked>, tensor<256x16xi32, #blocked>
    tt.store %optrs, %got : tensor<256x16x!tt.ptr<bf16>, #blocked>
    ttg.local_dealloc %dst : !ttg.memdesc<256x16xbf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// CHECK-LABEL: llvm.func @outer_unknown_fallback(
// CHECK-SAME: %[[U:[a-zA-Z0-9_]+]]: i32 {tt.divisibility = 8 : i32}
// CHECK: %[[OOB:[a-zA-Z0-9_]+]] = llvm.mlir.constant(-2147483648 : i32) : i32
// CHECK: %[[WIDTH:[a-zA-Z0-9_]+]] = llvm.mlir.constant(16 : i32) : i32
// CHECK: %[[TWO:[a-zA-Z0-9_]+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK: %[[EIGHT:[a-zA-Z0-9_]+]] = llvm.mlir.constant(8 : i32) : i32
// CHECK: %[[PRE:[a-zA-Z0-9_]+]] = llvm.add %[[U]], {{%[a-zA-Z0-9_]+}} : i32
// CHECK: %[[OFFSET:[a-zA-Z0-9_]+]] = llvm.xor %[[PRE]], %[[EIGHT]] : i32
// CHECK: %[[SHUFFLED:[a-zA-Z0-9_]+]] = rocdl.ds_bpermute {{%[a-zA-Z0-9_]+}}, %[[OFFSET]] : (i32, i32) -> i32
// CHECK: %[[BALLOT:[a-zA-Z0-9_]+]] = rocdl.ballot {{%[a-zA-Z0-9_]+}} : i64
// CHECK: %[[SHIFT:[a-zA-Z0-9_]+]] = llvm.lshr %[[BALLOT]], {{%[a-zA-Z0-9_]+}} : i64
// CHECK: %[[PRED:[a-zA-Z0-9_]+]] = llvm.trunc %[[SHIFT]] : i64 to i1
// CHECK: %[[BYTES:[a-zA-Z0-9_]+]] = llvm.mul %[[SHUFFLED]], %[[TWO]] : i32
// CHECK: %[[VOFF:[a-zA-Z0-9_]+]] = llvm.select %[[PRED]], %[[BYTES]], %[[OOB]] : i1, i32
// CHECK: rocdl.raw.ptr.buffer.load.lds {{%[a-zA-Z0-9_]+}}, {{%[a-zA-Z0-9_]+}}, %[[WIDTH]], %[[VOFF]],
// CHECK: rocdl.s.waitcnt -49168
// CHECK: rocdl.s.barrier
// CHECK: llvm.load {{.*}}!llvm.ptr<3>
// CHECK: llvm.return

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [1, 0]}>
#sM = #ttg.slice<{dim = 1, parent = #blocked}>
#sK = #ttg.slice<{dim = 0, parent = #blocked}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx936", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @outer_unknown_fallback(%A: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %O: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %U: i32 {tt.divisibility = 8 : i32}) {
    %m = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #sM>
    %k = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #sK>
    %m1 = tt.expand_dims %m {axis = 1 : i32} : tensor<256xi32, #sM> -> tensor<256x1xi32, #blocked>
    %k1 = tt.expand_dims %k {axis = 0 : i32} : tensor<16xi32, #sK> -> tensor<1x16xi32, #blocked>
    %mf = tt.broadcast %m1 : tensor<256x1xi32, #blocked> -> tensor<256x16xi32, #blocked>
    %kf = tt.broadcast %k1 : tensor<1x16xi32, #blocked> -> tensor<256x16xi32, #blocked>
    %pitch = arith.constant dense<1024> : tensor<256x16xi32, #blocked>
    %sixteen = arith.constant dense<16> : tensor<256x16xi32, #blocked>
    %eight = arith.constant dense<8> : tensor<256x16xi32, #blocked>
    %zero = arith.constant dense<0> : tensor<256x16xi32, #blocked>
    %row = arith.muli %mf, %pitch : tensor<256x16xi32, #blocked>
    %v0 = arith.addi %row, %kf : tensor<256x16xi32, #blocked>
    %us = tt.splat %U : i32 -> tensor<256x16xi32, #blocked>
    %sum = arith.addi %us, %v0 : tensor<256x16xi32, #blocked>
    %unknown = arith.xori %sum, %eight : tensor<256x16xi32, #blocked>
    %kb = arith.andi %kf, %eight : tensor<256x16xi32, #blocked>
    %mask = arith.cmpi eq, %kb, %zero : tensor<256x16xi32, #blocked>
    %init = arith.constant dense<-3.000000e+00> : tensor<256x16xbf16, #blocked>
    %dst = ttg.local_alloc %init : (tensor<256x16xbf16, #blocked>) -> !ttg.memdesc<256x16xbf16, #shared, #smem, mutable>
    ttg.local_barrier
    %t = hcug.buffer_load_to_local %A[%unknown] into %dst {contiguity = 8 : i32} : <bf16>[tensor<256x16xi32, #blocked>] -> <256x16xbf16, #shared, #smem, mutable>
    %w = ttg.async_wait %t {num = 0 : i32}
    ttg.local_barrier
    %got = ttg.local_load %dst token %w : !ttg.memdesc<256x16xbf16, #shared, #smem, mutable> -> tensor<256x16xbf16, #blocked>
    %orow = arith.muli %mf, %sixteen : tensor<256x16xi32, #blocked>
    %oi = arith.addi %orow, %kf : tensor<256x16xi32, #blocked>
    %op = tt.splat %O : !tt.ptr<bf16> -> tensor<256x16x!tt.ptr<bf16>, #blocked>
    %optrs = tt.addptr %op, %oi : tensor<256x16x!tt.ptr<bf16>, #blocked>, tensor<256x16xi32, #blocked>
    tt.store %optrs, %got : tensor<256x16x!tt.ptr<bf16>, #blocked>
    ttg.local_dealloc %dst : !ttg.memdesc<256x16xbf16, #shared, #smem, mutable>
    tt.return
  }
}

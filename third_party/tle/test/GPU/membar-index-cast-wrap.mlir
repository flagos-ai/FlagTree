// RUN: triton-opt %s --allocate-shared-memory-nv='compute-capability=120 ptx-version=88' --convert-triton-gpu-to-llvm='compute-capability=120 ptx-version=88' -reconcile-unrealized-casts | FileCheck %s
// Independent neighbouring-contract regression: cast before new interval arithmetic.
// Actual write/read index sets are both {0,2,...,254}; layouts change ownership.
#write = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#read = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cast_wrap_raw(%P: !tt.ptr<i32>, %Y: !tt.ptr<i32>) {
    %wr = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #write>
    %Pbase = tt.splat %P : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #write>
    %input = tt.addptr %Pbase, %wr : tensor<128x!tt.ptr<i32>, #write>, tensor<128xi32, #write>
    %value = tt.load %input : tensor<128x!tt.ptr<i32>, #write>
    %s = ttg.local_alloc : () -> !ttg.memdesc<1024xi32, #shared, #smem, mutable>
    %r = tt.make_range {start = 256 : i32, end = 384 : i32} : tensor<128xi32, #write>
    %narrow = arith.trunci %r : tensor<128xi32, #write> to tensor<128xi8, #write>
    %wide = arith.extui %narrow : tensor<128xi8, #write> to tensor<128xi32, #write>
    %two = arith.constant dense<2> : tensor<128xi32, #write>
    %i = arith.muli %wide, %two : tensor<128xi32, #write>
    %wp = "tle.local_pointers"(%s, %i) : (!ttg.memdesc<1024xi32, #shared, #smem, mutable>, tensor<128xi32, #write>) -> tensor<128x!tt.ptr<i32, 3>, #write>
    tt.store %wp, %value : tensor<128x!tt.ptr<i32, 3>, #write>
    %rr = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #read>
    %two_r = arith.constant dense<2> : tensor<128xi32, #read>
    %ri = arith.muli %rr, %two_r : tensor<128xi32, #read>
    %rp = "tle.local_pointers"(%s, %ri) : (!ttg.memdesc<1024xi32, #shared, #smem, mutable>, tensor<128xi32, #read>) -> tensor<128x!tt.ptr<i32, 3>, #read>
    %v = tt.load %rp : tensor<128x!tt.ptr<i32, 3>, #read>
    %Ybase = tt.splat %Y : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #read>
    %output = tt.addptr %Ybase, %rr : tensor<128x!tt.ptr<i32>, #read>, tensor<128xi32, #read>
    tt.store %output, %v : tensor<128x!tt.ptr<i32>, #read>
    tt.return
  }
}

// CHECK-LABEL: llvm.func @cast_wrap_raw
// CHECK: st.shared
// CHECK: nvvm.barrier0
// CHECK: ld.shared

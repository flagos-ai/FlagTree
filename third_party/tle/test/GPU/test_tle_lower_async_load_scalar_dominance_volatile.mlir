// RUN: triton-opt %s --split-input-file --triton-tle-lower-async-load --verify-each | FileCheck %s

// Scope: scalar fallback, direct/nested/CFG dominance, and volatile fallback.

// CHECK-LABEL: tt.func public @scalar
// CHECK-NOT: async_copy_global_to_local
// CHECK: tt.load
// CHECK-NOT: ttg.local_load
// CHECK: tt.store
// CHECK: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.target" = "hip:gfx936"} {
  tt.func public @scalar(%arg0: !tt.ptr<f32>) {
    %v = tt.load %arg0 {tt.load.async = false} : !tt.ptr<f32>
    tt.store %arg0, %v : !tt.ptr<f32>
    tt.return
  }
}

// -----

// CHECK-LABEL: tt.func public @scalar
// CHECK-NOT: async_copy_global_to_local
// CHECK: tt.load
// CHECK-NOT: ttg.local_load
// CHECK: tt.store
// CHECK: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.target" = "hip:gfx936"} {
  tt.func public @scalar(%arg0: !tt.ptr<f32>) {
    %v = tt.load %arg0 {tt.load.async = true} : !tt.ptr<f32>
    tt.store %arg0, %v : !tt.ptr<f32>
    tt.return
  }
}

// -----

// CHECK-LABEL: tt.func public @vector
// CHECK: ttg.async_copy_global_to_local
// CHECK: ttg.async_commit_group
// CHECK: ttg.async_wait
// CHECK: ttg.local_load
// CHECK: tt.store

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.target" = "hip:gfx936"} {
  tt.func public @vector(%arg0: !tt.ptr<f32>, %pred: i1) {
    %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #blocked>
    %p = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %ptr = tt.addptr %p, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %v = tt.load %ptr {tt.load.async = true} : tensor<128x!tt.ptr<f32>, #blocked>
    tt.store %ptr, %v : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: tt.func public @vector
// CHECK: ttg.async_copy_global_to_local
// CHECK: ttg.async_commit_group
// CHECK: ttg.async_wait
// CHECK: ttg.local_load
// CHECK: scf.if

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.target" = "hip:gfx936"} {
  tt.func public @vector(%arg0: !tt.ptr<f32>, %pred: i1) {
    %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #blocked>
    %p = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %ptr = tt.addptr %p, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %v = tt.load %ptr {tt.load.async = true} : tensor<128x!tt.ptr<f32>, #blocked>
    %out = scf.if %pred -> (tensor<128xf32, #blocked>) {
      scf.yield %v : tensor<128xf32, #blocked>
    } else {
      %twice = arith.addf %v, %v : tensor<128xf32, #blocked>
      scf.yield %twice : tensor<128xf32, #blocked>
    }
    tt.store %ptr, %out : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: tt.func public @vector
// CHECK: ttg.async_copy_global_to_local
// CHECK: ttg.async_commit_group
// CHECK: ttg.async_wait
// CHECK: ttg.local_load
// CHECK: arith.addf

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.target" = "hip:gfx936"} {
  tt.func public @vector(%arg0: !tt.ptr<f32>, %pred: i1) {
    %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #blocked>
    %p = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %ptr = tt.addptr %p, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %v = tt.load %ptr {tt.load.async = true} : tensor<128x!tt.ptr<f32>, #blocked>
    %direct = arith.addf %v, %v : tensor<128xf32, #blocked>
    %out = scf.if %pred -> (tensor<128xf32, #blocked>) {
      scf.yield %v : tensor<128xf32, #blocked>
    } else {
      %twice = arith.addf %v, %v : tensor<128xf32, #blocked>
      scf.yield %twice : tensor<128xf32, #blocked>
    }
    %sum = arith.addf %out, %direct : tensor<128xf32, #blocked>
    tt.store %ptr, %sum : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: tt.func public @vector
// CHECK: ttg.async_copy_global_to_local
// CHECK: ttg.async_commit_group
// CHECK: ttg.async_wait
// CHECK: ttg.local_load
// CHECK: cf.cond_br

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.target" = "hip:gfx936"} {
  tt.func public @vector(%arg0: !tt.ptr<f32>, %pred: i1) {
    %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #blocked>
    %p = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %ptr = tt.addptr %p, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %v = tt.load %ptr {tt.load.async = true} : tensor<128x!tt.ptr<f32>, #blocked>
    cf.cond_br %pred, ^then, ^else
  ^then:
    tt.store %ptr, %v : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  ^else:
    %twice = arith.addf %v, %v : tensor<128xf32, #blocked>
    tt.store %ptr, %twice : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: tt.func public @vector
// CHECK-NOT: ttg.async_copy_global_to_local
// CHECK-NOT: ttg.local_alloc
// CHECK: tt.load {{.*}}isVolatile = true
// CHECK-NOT: = tt.load 
// CHECK-NOT: ttg.async_copy_global_to_local
// CHECK: tt.store
// CHECK: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.target" = "hip:gfx936"} {
  tt.func public @vector(%arg0: !tt.ptr<f32>, %pred: i1) {
    %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #blocked>
    %p = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %ptr = tt.addptr %p, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %v = tt.load %ptr {tt.load.async = true, isVolatile = true} : tensor<128x!tt.ptr<f32>, #blocked>
    tt.store %ptr, %v : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

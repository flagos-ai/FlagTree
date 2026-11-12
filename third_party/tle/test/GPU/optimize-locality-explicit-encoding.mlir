// RUN: triton-opt %s -split-input-file -tritongpu-optimize-thread-locality | FileCheck %s

// Exercise the shared pass with CUDA sm80's 32-lane warps and AMD gfx942's
// 64-lane wavefronts. Layout constraints must preserve tensor types as well as
// attributes, independently of the target. Iluvatar's 64-lane cases also live
// in third_party/iluvatar/python/test/unit/tle/inputs/set_layout_thread_locality.mlir.
// Unconstrained reshape, gather and loop reductions are covered by optimize-locality.mlir.

// CHECK-DAG: #[[$RESHAPE_LAYOUT:blocked[0-9]*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
// CHECK-LABEL: tt.func @explicit_reshape_warp32
// CHECK-NEXT: %[[VIEW:.*]] = tt.reshape %arg0 allow_reorder {tle.explicit_encoding.0 = #[[$RESHAPE_LAYOUT]]} : tensor<8x128xf32, #{{.*}}> -> tensor<64x16xf32, #[[$RESHAPE_LAYOUT]]>
// CHECK-NEXT: %[[SUM:.*]] = "tt.reduce"(%[[VIEW]]) <{axis = 1 : i32}>
// CHECK: }) : (tensor<64x16xf32, #[[$RESHAPE_LAYOUT]]>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #[[$RESHAPE_LAYOUT]]}>>
// CHECK-NEXT: tt.return %[[SUM]]
#src = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [2, 1], order = [1, 0]}>
#dst = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @explicit_reshape_warp32(%arg0: tensor<8x128xf32, #src>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #dst}>> {
    %view = tt.reshape %arg0 allow_reorder {tle.explicit_encoding.0 = #dst} : tensor<8x128xf32, #src> -> tensor<64x16xf32, #dst>
    %sum = "tt.reduce"(%view) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %max = arith.maximumf %a, %b : f32
      tt.reduce.return %max : f32
    }) : (tensor<64x16xf32, #dst>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #dst}>>
    tt.return %sum : tensor<64xf32, #ttg.slice<{dim = 1, parent = #dst}>>
  }
}

// -----

// CHECK-DAG: #[[$GATHER_LAYOUT:blocked[0-9]*]] = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [16, 2], warpsPerCTA = [2, 2], order = [1, 0]}>
// CHECK-LABEL: tt.func @explicit_gather_warp32
// CHECK-NEXT: %[[GATHER:.*]] = tt.gather %arg0[%arg1] {axis = 0 : i32, tle.explicit_encoding.0 = #[[$GATHER_LAYOUT]]} : (tensor<64x64xf32, #[[$GATHER_LAYOUT]]>, tensor<64x64xi32, #[[$GATHER_LAYOUT]]>) -> tensor<64x64xf32, #[[$GATHER_LAYOUT]]>
// CHECK-NEXT: tt.return %[[GATHER]]
#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [16, 2], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @explicit_gather_warp32(%arg0: tensor<64x64xf32, #blocked>, %arg1: tensor<64x64xi32, #blocked>) -> tensor<64x64xf32, #blocked> {
    %gather = tt.gather %arg0[%arg1] {axis = 0 : i32, tle.explicit_encoding.0 = #blocked} : (tensor<64x64xf32, #blocked>, tensor<64x64xi32, #blocked>) -> tensor<64x64xf32, #blocked>
    tt.return %gather : tensor<64x64xf32, #blocked>
  }
}

// -----

// CHECK-LABEL: tt.func @explicit_reduction_in_loop_warp32
// CHECK: %[[LOOP:.*]] = scf.for {{.*}} -> (tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>)
// CHECK-NEXT: %[[LOAD:.*]] = tt.load
// CHECK-NEXT: %[[SUM:.*]] = "tt.reduce"(%[[LOAD]]) <{axis = 1 : i32}>
// CHECK: }) {tle.explicit_encoding.0 = #ttg.slice<{dim = 1, parent = #blocked}>} : (tensor<32x128xf32, #blocked>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
// CHECK-NEXT: %[[NEXT:.*]] = arith.addf %{{.*}}, %[[SUM]] : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
// CHECK-NEXT: scf.yield %[[NEXT]]
// CHECK-NEXT: {{^ *}}}
// CHECK-NEXT: tt.return %[[LOOP]]
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @explicit_reduction_in_loop_warp32(%arg0: tensor<32x128x!tt.ptr<f32>, #blocked>, %n: i32) -> tensor<32xf32, #slice> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<32xf32, #slice>
    %result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %zero) -> (tensor<32xf32, #slice>) : i32 {
      %load = tt.load %arg0 : tensor<32x128x!tt.ptr<f32>, #blocked>
      %sum = "tt.reduce"(%load) <{axis = 1 : i32}> ({
      ^bb0(%a: f32, %b: f32):
        %add = arith.addf %a, %b : f32
        tt.reduce.return %add : f32
      }) {tle.explicit_encoding.0 = #slice} : (tensor<32x128xf32, #blocked>) -> tensor<32xf32, #slice>
      %next = arith.addf %acc, %sum : tensor<32xf32, #slice>
      scf.yield %next : tensor<32xf32, #slice>
    }
    tt.return %result : tensor<32xf32, #slice>
  }
}

// -----

// CHECK-LABEL: tt.func @explicit_accumulator_in_loop_warp32
// CHECK: %[[LOOP:.*]] = scf.for {{.*}} -> (tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>)
// CHECK-NEXT: %[[LOAD:.*]] = tt.load
// CHECK-NEXT: %[[SUM:.*]] = "tt.reduce"(%[[LOAD]]) <{axis = 1 : i32}>
// CHECK: }) : (tensor<32x128xf32, #blocked>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
// CHECK-NEXT: %[[NEXT:.*]] = arith.addf %{{.*}}, %[[SUM]] {tle.explicit_encoding.0 = #ttg.slice<{dim = 1, parent = #blocked}>} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
// CHECK-NEXT: scf.yield %[[NEXT]]
// CHECK-NEXT: {{^ *}}}
// CHECK-NEXT: tt.return %[[LOOP]]
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @explicit_accumulator_in_loop_warp32(%arg0: tensor<32x128x!tt.ptr<f32>, #blocked>, %n: i32) -> tensor<32xf32, #slice> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<32xf32, #slice>
    %result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %zero) -> (tensor<32xf32, #slice>) : i32 {
      %load = tt.load %arg0 : tensor<32x128x!tt.ptr<f32>, #blocked>
      %sum = "tt.reduce"(%load) <{axis = 1 : i32}> ({
      ^bb0(%a: f32, %b: f32):
        %add = arith.addf %a, %b : f32
        tt.reduce.return %add : f32
      }) : (tensor<32x128xf32, #blocked>) -> tensor<32xf32, #slice>
      %next = arith.addf %acc, %sum {tle.explicit_encoding.0 = #slice} : tensor<32xf32, #slice>
      scf.yield %next : tensor<32xf32, #slice>
    }
    tt.return %result : tensor<32xf32, #slice>
  }
}

// -----

// CHECK-DAG: #[[$RESHAPE_LAYOUT:blocked[0-9]*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
// CHECK-LABEL: tt.func @explicit_reshape_warp64
// CHECK-NEXT: %[[VIEW:.*]] = tt.reshape %arg0 allow_reorder {tle.explicit_encoding.0 = #[[$RESHAPE_LAYOUT]]} : tensor<8x128xf32, #{{.*}}> -> tensor<64x16xf32, #[[$RESHAPE_LAYOUT]]>
// CHECK-NEXT: %[[SUM:.*]] = "tt.reduce"(%[[VIEW]]) <{axis = 1 : i32}>
// CHECK: }) : (tensor<64x16xf32, #[[$RESHAPE_LAYOUT]]>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #[[$RESHAPE_LAYOUT]]}>>
// CHECK-NEXT: tt.return %[[SUM]]
#src = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 8], warpsPerCTA = [2, 1], order = [1, 0]}>
#dst = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @explicit_reshape_warp64(%arg0: tensor<8x128xf32, #src>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #dst}>> {
    %view = tt.reshape %arg0 allow_reorder {tle.explicit_encoding.0 = #dst} : tensor<8x128xf32, #src> -> tensor<64x16xf32, #dst>
    %sum = "tt.reduce"(%view) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %max = arith.maximumf %a, %b : f32
      tt.reduce.return %max : f32
    }) : (tensor<64x16xf32, #dst>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #dst}>>
    tt.return %sum : tensor<64xf32, #ttg.slice<{dim = 1, parent = #dst}>>
  }
}

// -----

// CHECK-DAG: #[[$GATHER_LAYOUT:blocked[0-9]*]] = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [32, 2], warpsPerCTA = [2, 2], order = [1, 0]}>
// CHECK-LABEL: tt.func @explicit_gather_warp64
// CHECK-NEXT: %[[GATHER:.*]] = tt.gather %arg0[%arg1] {axis = 0 : i32, tle.explicit_encoding.0 = #[[$GATHER_LAYOUT]]} : (tensor<64x64xf32, #[[$GATHER_LAYOUT]]>, tensor<64x64xi32, #[[$GATHER_LAYOUT]]>) -> tensor<64x64xf32, #[[$GATHER_LAYOUT]]>
// CHECK-NEXT: tt.return %[[GATHER]]
#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [32, 2], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @explicit_gather_warp64(%arg0: tensor<64x64xf32, #blocked>, %arg1: tensor<64x64xi32, #blocked>) -> tensor<64x64xf32, #blocked> {
    %gather = tt.gather %arg0[%arg1] {axis = 0 : i32, tle.explicit_encoding.0 = #blocked} : (tensor<64x64xf32, #blocked>, tensor<64x64xi32, #blocked>) -> tensor<64x64xf32, #blocked>
    tt.return %gather : tensor<64x64xf32, #blocked>
  }
}

// -----

// CHECK-LABEL: tt.func @explicit_reduction_in_loop_warp64
// CHECK: %[[LOOP:.*]] = scf.for {{.*}} -> (tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>)
// CHECK-NEXT: %[[LOAD:.*]] = tt.load
// CHECK-NEXT: %[[SUM:.*]] = "tt.reduce"(%[[LOAD]]) <{axis = 1 : i32}>
// CHECK: }) {tle.explicit_encoding.0 = #ttg.slice<{dim = 1, parent = #blocked}>} : (tensor<32x128xf32, #blocked>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
// CHECK-NEXT: %[[NEXT:.*]] = arith.addf %{{.*}}, %[[SUM]] : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
// CHECK-NEXT: scf.yield %[[NEXT]]
// CHECK-NEXT: {{^ *}}}
// CHECK-NEXT: tt.return %[[LOOP]]
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @explicit_reduction_in_loop_warp64(%arg0: tensor<32x128x!tt.ptr<f32>, #blocked>, %n: i32) -> tensor<32xf32, #slice> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<32xf32, #slice>
    %result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %zero) -> (tensor<32xf32, #slice>) : i32 {
      %load = tt.load %arg0 : tensor<32x128x!tt.ptr<f32>, #blocked>
      %sum = "tt.reduce"(%load) <{axis = 1 : i32}> ({
      ^bb0(%a: f32, %b: f32):
        %add = arith.addf %a, %b : f32
        tt.reduce.return %add : f32
      }) {tle.explicit_encoding.0 = #slice} : (tensor<32x128xf32, #blocked>) -> tensor<32xf32, #slice>
      %next = arith.addf %acc, %sum : tensor<32xf32, #slice>
      scf.yield %next : tensor<32xf32, #slice>
    }
    tt.return %result : tensor<32xf32, #slice>
  }
}

// -----

// CHECK-LABEL: tt.func @explicit_accumulator_in_loop_warp64
// CHECK: %[[LOOP:.*]] = scf.for {{.*}} -> (tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>)
// CHECK-NEXT: %[[LOAD:.*]] = tt.load
// CHECK-NEXT: %[[SUM:.*]] = "tt.reduce"(%[[LOAD]]) <{axis = 1 : i32}>
// CHECK: }) : (tensor<32x128xf32, #blocked>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
// CHECK-NEXT: %[[NEXT:.*]] = arith.addf %{{.*}}, %[[SUM]] {tle.explicit_encoding.0 = #ttg.slice<{dim = 1, parent = #blocked}>} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
// CHECK-NEXT: scf.yield %[[NEXT]]
// CHECK-NEXT: {{^ *}}}
// CHECK-NEXT: tt.return %[[LOOP]]
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @explicit_accumulator_in_loop_warp64(%arg0: tensor<32x128x!tt.ptr<f32>, #blocked>, %n: i32) -> tensor<32xf32, #slice> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<32xf32, #slice>
    %result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %zero) -> (tensor<32xf32, #slice>) : i32 {
      %load = tt.load %arg0 : tensor<32x128x!tt.ptr<f32>, #blocked>
      %sum = "tt.reduce"(%load) <{axis = 1 : i32}> ({
      ^bb0(%a: f32, %b: f32):
        %add = arith.addf %a, %b : f32
        tt.reduce.return %add : f32
      }) : (tensor<32x128xf32, #blocked>) -> tensor<32xf32, #slice>
      %next = arith.addf %acc, %sum {tle.explicit_encoding.0 = #slice} : tensor<32xf32, #slice>
      scf.yield %next : tensor<32xf32, #slice>
    }
    tt.return %result : tensor<32xf32, #slice>
  }
}

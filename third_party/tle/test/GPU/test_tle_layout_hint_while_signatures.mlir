// RUN: triton-opt %s -convert-triton-to-tritongpu='target=cuda:90 num-warps=4' -verify-each | FileCheck %s

#flat = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#matrix = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>

// CHECK-DAG: #[[$FLAT:.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK-DAG: #[[$MATRIX:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module {
  // A hint on an initial value reaches only the before/yield signature. The
  // original propagation accessed result zero even though no result exists.
  // CHECK-LABEL: tt.func @hinted_initial_without_results
  // CHECK: scf.while {{.*}} : (tensor<4xi64, #[[$FLAT]]>) -> ()
  // CHECK: scf.condition({{%.*}})
  // CHECK: scf.yield {{.*}} : tensor<4xi64, #[[$FLAT]]>
  // CHECK: tt.return
  tt.func @hinted_initial_without_results(%base: !tt.ptr<i64>, %again: i1) {
    %indices = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %bases = tt.splat %base : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %ptrs = tt.addptr %bases, %indices : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %initial = tt.load %ptrs : tensor<4x!tt.ptr<i64>>
    %hint = tle.gpu.set_layout %initial {target_encoding = #flat} : tensor<4xi64> -> tensor<4xi64>
    scf.while (%carried = %hint) : (tensor<4xi64>) -> () {
      scf.condition(%again)
    } do {
      %step = tt.load %ptrs : tensor<4x!tt.ptr<i64>>
      scf.yield %step : tensor<4xi64>
    }
    tt.return
  }

  // A hint first discovered through an after-region yield has the same before
  // signature, including the initial value. It is not a while result hint.
  // CHECK-LABEL: tt.func @hinted_yield_without_results
  // CHECK: scf.while {{.*}} : (tensor<4xi64, #[[$FLAT]]>) -> ()
  // CHECK: scf.condition({{%.*}})
  // CHECK: scf.yield {{.*}} : tensor<4xi64, #[[$FLAT]]>
  // CHECK: tt.return
  tt.func @hinted_yield_without_results(%base: !tt.ptr<i64>, %again: i1) {
    %indices = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %bases = tt.splat %base : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %ptrs = tt.addptr %bases, %indices : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %initial = tt.load %ptrs : tensor<4x!tt.ptr<i64>>
    scf.while (%carried = %initial) : (tensor<4xi64>) -> () {
      scf.condition(%again)
    } do {
      %step = tt.load %ptrs : tensor<4x!tt.ptr<i64>>
      %hint = tle.gpu.set_layout %step {target_encoding = #flat} : tensor<4xi64> -> tensor<4xi64>
      scf.yield %hint : tensor<4xi64>
    }
    tt.return
  }

  // The two signatures may differ in rank even with equal arity. Input/yield
  // values retain the flat hint, while condition/after/result values retain the
  // matrix hint. The after argument is itself hinted to exercise block handling.
  // CHECK-LABEL: tt.func @independent_region_encodings
  // CHECK: scf.while {{.*}} : (tensor<4xi64, #[[$FLAT]]>) -> tensor<2x2xi64, #[[$MATRIX]]>
  // CHECK: scf.condition({{%.*}}) {{.*}} : tensor<2x2xi64, #[[$MATRIX]]>
  // CHECK: ^bb0({{%.*}}: tensor<2x2xi64, #[[$MATRIX]]>)
  // CHECK: scf.yield {{.*}} : tensor<4xi64, #[[$FLAT]]>
  // CHECK: tt.store {{.*}} : tensor<2x2x!tt.ptr<i64>, #[[$MATRIX]]>
  // CHECK: tt.return
  tt.func @independent_region_encodings(%base: !tt.ptr<i64>, %output_base: !tt.ptr<i64>, %again: i1) {
    %indices = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %bases = tt.splat %base : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %ptrs = tt.addptr %bases, %indices : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %initial = tt.load %ptrs : tensor<4x!tt.ptr<i64>>
    %matrix_indices = tt.reshape %indices : tensor<4xi32> -> tensor<2x2xi32>
    %matrix_bases = tt.splat %base : !tt.ptr<i64> -> tensor<2x2x!tt.ptr<i64>>
    %matrix_ptrs = tt.addptr %matrix_bases, %matrix_indices : tensor<2x2x!tt.ptr<i64>>, tensor<2x2xi32>
    %output_bases = tt.splat %output_base : !tt.ptr<i64> -> tensor<2x2x!tt.ptr<i64>>
    %output = tt.addptr %output_bases, %matrix_indices : tensor<2x2x!tt.ptr<i64>>, tensor<2x2xi32>
    %initial_hint = tle.gpu.set_layout %initial {target_encoding = #flat} : tensor<4xi64> -> tensor<4xi64>
    %result = scf.while (%carried = %initial_hint) : (tensor<4xi64>) -> tensor<2x2xi64> {
      %forwarded = tt.load %matrix_ptrs : tensor<2x2x!tt.ptr<i64>>
      %condition_hint = tle.gpu.set_layout %forwarded {target_encoding = #matrix} : tensor<2x2xi64> -> tensor<2x2xi64>
      scf.condition(%again) %condition_hint : tensor<2x2xi64>
    } do {
    ^bb0(%matrix_value: tensor<2x2xi64>):
      %after_hint = tle.gpu.set_layout %matrix_value {target_encoding = #matrix} : tensor<2x2xi64> -> tensor<2x2xi64>
      %step = tt.load %ptrs : tensor<4x!tt.ptr<i64>>
      scf.yield %step : tensor<4xi64>
    }
    %result_hint = tle.gpu.set_layout %result {target_encoding = #matrix} : tensor<2x2xi64> -> tensor<2x2xi64>
    tt.store %output, %result_hint : tensor<2x2x!tt.ptr<i64>>
    tt.return
  }

  // Conversely, the before signature can be empty while the result/after
  // signature has a tensor. A result hint must not index an initial operand.
  // CHECK-LABEL: tt.func @hinted_result_without_inputs
  // CHECK: scf.while : () -> tensor<4xi64, #[[$FLAT]]>
  // CHECK: scf.condition({{%.*}}) {{.*}} : tensor<4xi64, #[[$FLAT]]>
  // CHECK: ^bb0({{%.*}}: tensor<4xi64, #[[$FLAT]]>)
  // CHECK: scf.yield
  // CHECK: tt.store {{.*}} : tensor<4x!tt.ptr<i64>, #[[$FLAT]]>
  // CHECK: tt.return
  tt.func @hinted_result_without_inputs(%base: !tt.ptr<i64>, %output_base: !tt.ptr<i64>, %again: i1) {
    %indices = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %bases = tt.splat %base : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %ptrs = tt.addptr %bases, %indices : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %output_bases = tt.splat %output_base : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %output = tt.addptr %output_bases, %indices : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %result = scf.while : () -> tensor<4xi64> {
      %forwarded = tt.load %ptrs : tensor<4x!tt.ptr<i64>>
      scf.condition(%again) %forwarded : tensor<4xi64>
    } do {
    ^bb0(%value: tensor<4xi64>):
      scf.yield
    }
    %hint = tle.gpu.set_layout %result {target_encoding = #flat} : tensor<4xi64> -> tensor<4xi64>
    tt.store %output, %hint : tensor<4x!tt.ptr<i64>>
    tt.return
  }
}

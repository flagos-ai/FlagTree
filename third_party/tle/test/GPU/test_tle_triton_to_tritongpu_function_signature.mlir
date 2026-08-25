// RUN: triton-opt %s -convert-triton-to-tritongpu='target=cuda:90 num-warps=4' | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#explicit = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-DAG: #[[$BLOCKED:.*]] = #ttg.blocked
  // CHECK-DAG: #[[$EXPLICIT:.*]] = #ttg.blocked<{{.*}}sizePerThread = [1, 2]{{.*}}threadsPerWarp = [2, 16]{{.*}}warpsPerCTA = [4, 1]{{.*}}>
  // CHECK-LABEL: tt.func private @tensor_memdesc_worker(
  // CHECK-SAME: tensor<64xi32, #[[$BLOCKED]]>
  // CHECK-SAME: !ttg.memdesc<128xi8, #shared, #smem, mutable>
  // CHECK-SAME: i32
  // CHECK-SAME: -> tensor<64xi32, #[[$BLOCKED]]>
  // CHECK-NOT: builtin.unrealized_conversion_cast
  tt.func private @tensor_memdesc_worker(
      %value: tensor<64xi32>,
      %arena: !ttg.memdesc<128xi8, #shared, #smem, mutable>,
      %index: i32) -> tensor<64xi32> attributes {noinline = true} {
    %alias = tle.memdesc_alias %arena {offset_bytes = 64 : i64} :
        !ttg.memdesc<128xi8, #shared, #smem, mutable> ->
        !ttg.memdesc<64xi8, #shared, #smem, mutable>
    %ptr = "tle.local_pointers"(%alias, %index) :
        (!ttg.memdesc<64xi8, #shared, #smem, mutable>, i32) -> !tt.ptr<i8, 3>
    %result = arith.addi %value, %value : tensor<64xi32>
    tt.return %result : tensor<64xi32>
  }

  // CHECK-LABEL: tt.func public @kernel(
  // CHECK: tt.call @tensor_memdesc_worker
  // CHECK-SAME: tensor<64xi32, #[[$BLOCKED]]>, !ttg.memdesc<128xi8, #shared, #smem, mutable>, i32
  // CHECK-NOT: builtin.unrealized_conversion_cast
  tt.func public @kernel(%output: !tt.ptr<i32>) {
    %index = arith.constant 0 : i32
    %arena = ttg.local_alloc : () ->
        !ttg.memdesc<128xi8, #shared, #smem, mutable>
    %value = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %result = tt.call @tensor_memdesc_worker(%value, %arena, %index) :
        (tensor<64xi32>, !ttg.memdesc<128xi8, #shared, #smem, mutable>, i32) ->
        tensor<64xi32>
    %output_splat = tt.splat %output : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>>
    %offsets = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %output_ptrs = tt.addptr %output_splat, %offsets :
        tensor<64x!tt.ptr<i32>>, tensor<64xi32>
    tt.store %output_ptrs, %result : tensor<64x!tt.ptr<i32>>
    tt.return
  }

  // An explicit result layout is a noinline function ABI contract. Keep the
  // callee signature and every call result synchronized with the return value.
  // CHECK-LABEL: tt.func private @explicit_result_worker
  // CHECK-SAME: -> tensor<8x32xf32, #[[$EXPLICIT]]>
  tt.func private @explicit_result_worker() -> tensor<8x32xf32> attributes {noinline = true} {
    %zero = arith.constant dense<0.0> : tensor<8x32xf32>
    %encoded = tle.encoding %zero {target_encoding = #explicit} : tensor<8x32xf32> -> tensor<8x32xf32>
    // CHECK: tt.return {{.*}} : tensor<8x32xf32, #[[$EXPLICIT]]>
    tt.return %encoded : tensor<8x32xf32>
  }

  // CHECK-LABEL: tt.func public @explicit_result_caller
  tt.func public @explicit_result_caller() attributes {noinline = false} {
    // CHECK: tt.call @explicit_result_worker() : () -> tensor<8x32xf32, #[[$EXPLICIT]]>
    %result = tt.call @explicit_result_worker() : () -> tensor<8x32xf32>
    tt.return
  }

  // An entry encoding on a noinline tensor argument is the call ABI, not an
  // internal layout conversion.
  // CHECK-LABEL: tt.func private @explicit_argument_worker
  // CHECK-SAME: (%arg0: tensor<8x32xf32, #[[$EXPLICIT]]>)
  tt.func private @explicit_argument_worker(%arg0: tensor<8x32xf32>) attributes {noinline = true} {
    %encoded = tle.encoding %arg0 {target_encoding = #explicit} : tensor<8x32xf32> -> tensor<8x32xf32>
    // CHECK: tt.return
    tt.return
  }

  // CHECK-LABEL: tt.func public @explicit_argument_caller
  tt.func public @explicit_argument_caller() attributes {noinline = false} {
    %zero = arith.constant dense<0.0> : tensor<8x32xf32>
    %encoded = tle.encoding %zero {target_encoding = #explicit} : tensor<8x32xf32> -> tensor<8x32xf32>
    // CHECK: tt.call @explicit_argument_worker(%{{.*}}) : (tensor<8x32xf32, #[[$EXPLICIT]]>) -> ()
    tt.call @explicit_argument_worker(%encoded) : (tensor<8x32xf32>) -> ()
    tt.return
  }
}

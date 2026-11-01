// RUN: triton-opt %s -convert-triton-to-tritongpu='target=cuda:90 num-warps=4 enable-source-remat=false' | FileCheck %s

// A gather index layout propagates to the dot result and accumulator, but
// neither dot input has a dot-operand encoding yet. Dot conversion must keep
// the propagated C/D layout instead of replacing it with its default layout.
#gather = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [0, 1]}>
// CHECK: #[[$GATHER:.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [0, 1]}>

module {
  // CHECK-LABEL: tt.func @dot_to_gather
  tt.func @dot_to_gather(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>, %indices: !tt.ptr<i32>, %output: !tt.ptr<f32>) {
    %ap = tt.splat %a : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>>
    %av = tt.load %ap : tensor<64x128x!tt.ptr<bf16>>
    %bp = tt.splat %b : !tt.ptr<bf16> -> tensor<128x16x!tt.ptr<bf16>>
    %bv = tt.load %bp : tensor<128x16x!tt.ptr<bf16>>
    %ip = tt.splat %indices : !tt.ptr<i32> -> tensor<64x1x!tt.ptr<i32>>
    %iv = tt.load %ip : tensor<64x1x!tt.ptr<i32>>
    %ie = tle.gpu.set_layout %iv {target_encoding = #gather} : tensor<64x1xi32> -> tensor<64x1xi32>
    %zero = arith.constant dense<0.0> : tensor<64x16xf32>
    // CHECK: %[[DOT:.*]] = tt.dot {{.*}} -> tensor<64x16xf32, #[[$GATHER]]>
    %dot = tt.dot %av, %bv, %zero : tensor<64x128xbf16> * tensor<128x16xbf16> -> tensor<64x16xf32>
    // CHECK-NOT: unrealized_conversion_cast
    // CHECK: %[[VALUE:.*]] = tt.gather %[[DOT]][{{.*}}] {{.*}} -> tensor<64x1xf32, #[[$GATHER]]>
    %value = tt.gather %dot[%ie] {axis = 1 : i32} : (tensor<64x16xf32>, tensor<64x1xi32>) -> tensor<64x1xf32>
    %op = tt.splat %output : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>>
    // CHECK: tt.store {{.*}}, %[[VALUE]]
    tt.store %op, %value : tensor<64x1x!tt.ptr<f32>>
    tt.return
  }
}

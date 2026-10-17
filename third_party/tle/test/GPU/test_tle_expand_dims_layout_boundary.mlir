// RUN: triton-opt %s -convert-triton-to-tritongpu='target=cuda:90 num-warps=4' | FileCheck %s

#matrix = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [1, 0]}>
#row = #ttg.slice<{dim = 1, parent = #matrix}>

module {
  // An explicit source layout is valid even when the expansion uses another
  // axis. Convert that use to the required slice instead of propagating an
  // incompatible slice through expand_dims.
  // CHECK-LABEL: tt.func @expand_in_another_layout_domain
  tt.func @expand_in_another_layout_domain(%output: !tt.ptr<i32>) {
    %index = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %hint = tle.gpu.set_layout %index {target_encoding = #row} : tensor<4xi32> -> tensor<4xi32>
    // CHECK: %[[SOURCE:.*]] = ttg.convert_layout %{{.*}} : tensor<4xi32, #ttg.slice<{dim = 1, parent = {{.*}}}>> -> tensor<4xi32, #ttg.slice<{dim = 0, parent = {{.*}}}>>
    // CHECK: %[[EXPANDED:.*]] = tt.expand_dims %[[SOURCE]] {axis = 0 : i32{{.*}}}
    %expanded = tt.expand_dims %hint {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %other = tle.gpu.set_layout %expanded {target_encoding = #matrix} : tensor<1x4xi32> -> tensor<1x4xi32>
    %out = tt.splat %output : !tt.ptr<i32> -> tensor<1x4x!tt.ptr<i32>>
    // CHECK: tt.store %{{.*}}, %[[EXPANDED]]
    tt.store %out, %other : tensor<1x4x!tt.ptr<i32>>
    tt.return
  }

  // A shared computed predicate is not a range/constant that can simply be
  // rematerialized. Its two expansion uses need independent slice contracts.
  // CHECK-LABEL: tt.func @shared_predicate_matrix
  tt.func @shared_predicate_matrix(%limit: i32, %output: !tt.ptr<i1>) {
    %index = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %bound = tt.splat %limit : i32 -> tensor<4xi32>
    %active = arith.cmpi slt, %index, %bound : tensor<4xi32>
    // CHECK-DAG: tt.expand_dims %{{.*}} {axis = 1 : i32{{.*}}} : tensor<4xi1, #ttg.slice<{dim = 1, parent = {{.*}}}>>
    // CHECK-DAG: tt.expand_dims %{{.*}} {axis = 0 : i32{{.*}}} : tensor<4xi1, #ttg.slice<{dim = 0, parent = {{.*}}}>>
    %rows = tt.expand_dims %active {axis = 1 : i32} : tensor<4xi1> -> tensor<4x1xi1>
    %cols = tt.expand_dims %active {axis = 0 : i32} : tensor<4xi1> -> tensor<1x4xi1>
    %rows_b = tt.broadcast %rows : tensor<4x1xi1> -> tensor<4x4xi1>
    %cols_b = tt.broadcast %cols : tensor<1x4xi1> -> tensor<4x4xi1>
    %pair = arith.andi %rows_b, %cols_b : tensor<4x4xi1>
    %hint = tle.gpu.set_layout %pair {target_encoding = #matrix} : tensor<4x4xi1> -> tensor<4x4xi1>
    %out = tt.splat %output : !tt.ptr<i1> -> tensor<4x4x!tt.ptr<i1>>
    tt.store %out, %hint : tensor<4x4x!tt.ptr<i1>>
    tt.return
  }
}

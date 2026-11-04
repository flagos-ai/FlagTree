// RUN: triton-opt %s -convert-triton-to-tritongpu='target=cuda:90 num-warps=4 enable-source-remat=false' | FileCheck %s

// The transpose and pointer consumers constrain independent edges of one
// computed index. Encoded tensor types alone do not make addptr legal.
#source = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#transpose = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
module {
  // CHECK-LABEL: tt.func @shared_index_addptr
  tt.func @shared_index_addptr(%input: !tt.ptr<i32>, %out: !tt.ptr<i32>, %offset: i32) {
    %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %row = tt.expand_dims %range {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %matrix = tt.broadcast %row : tensor<1x128xi32> -> tensor<2x128xi32>
    %base = tt.splat %offset : i32 -> tensor<2x128xi32>
    %index = arith.addi %matrix, %base : tensor<2x128xi32>
    %pointer = tt.splat %input : !tt.ptr<i32> -> tensor<2x128x!tt.ptr<i32>>
    %address = tt.addptr %pointer, %index : tensor<2x128x!tt.ptr<i32>>, tensor<2x128xi32>
    %hinted = tle.gpu.set_layout %address {target_encoding = #source} : tensor<2x128x!tt.ptr<i32>> -> tensor<2x128x!tt.ptr<i32>>
    %value = tt.load %hinted : tensor<2x128x!tt.ptr<i32>>
    %trans = tt.trans %index {order = array<i32: 1, 0>} : tensor<2x128xi32> -> tensor<128x2xi32>
    %other = tle.gpu.set_layout %trans {target_encoding = #transpose} : tensor<128x2xi32> -> tensor<128x2xi32>
    %op = tt.splat %out : !tt.ptr<i32> -> tensor<128x2x!tt.ptr<i32>>
    // CHECK: ttg.convert_layout
    // CHECK: tt.addptr
    // CHECK-NOT: unrealized_conversion_cast
    // CHECK: tt.store
    tt.store %op, %other : tensor<128x2x!tt.ptr<i32>>
    // Keep the pointer edge observable too.
    %outp = tt.splat %out : !tt.ptr<i32> -> tensor<2x128x!tt.ptr<i32>>
    tt.store %outp, %value : tensor<2x128x!tt.ptr<i32>>
    tt.return
  }
}

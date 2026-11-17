// RUN: triton-opt --split-input-file --verify-each %s -graph-optimize='rule-mask=8' -canonicalize -cse -o %t
// RUN: FileCheck %s < %t
// RUN: triton-opt --split-input-file --verify-each %s -graph-optimize='rule-mask=8 compile-mode=simt_only' -canonicalize -cse -o %t.simt
// RUN: triton-opt --split-input-file --verify-each %t -graph-optimize='rule-mask=8' -canonicalize -cse -o %t.twice
// RUN: diff %t %t.twice
// RUN: triton-opt --split-input-file --verify-each %s -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.disabled
// RUN: triton-opt --split-input-file %s -canonicalize -cse -o %t.control
// RUN: diff %t.control %t.disabled
// RUN: diff %t.control %t.simt

// CHECK: module attributes {{.*}}hacc.coalesce_factor = 8 : i32
// CHECK-LABEL: tt.func @reshape_transpose(
// CHECK: tt.load {{.*}} : tensor<8x16x16x!tt.ptr<f16>>
// CHECK: scf.for
// CHECK: %[[RHS:.*]] = tt.load {{.*}} : tensor<16x16x!tt.ptr<f16>>
// CHECK: tt.dot %{{.*}}, %[[RHS]], %{{.*}} : tensor<128x16xf16> * tensor<16x16xf16> -> tensor<128x16xf32>
// CHECK: tt.trans {{.*}}order = array<i32: 0, 2, 1>
// CHECK: tt.store {{.*}} : tensor<8x16x16x!tt.ptr<f32>>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hacc.grid_num_tiles = 64 : i32} {
  tt.func @reshape_transpose(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %C: !tt.ptr<f32>, %count: i32) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %stride = arith.constant 256 : i32
    %base = arith.muli %pid, %stride : i32
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %c = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %r2 = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %c2d = tt.expand_dims %c {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %sixteen = arith.constant dense<16> : tensor<16x1xi32>
    %rs = arith.muli %r2, %sixteen : tensor<16x1xi32>
    %rr = tt.broadcast %rs : tensor<16x1xi32> -> tensor<16x16xi32>
    %cc = tt.broadcast %c2d : tensor<1x16xi32> -> tensor<16x16xi32>
    %local = arith.addi %rr, %cc : tensor<16x16xi32>
    %bases = tt.splat %base : i32 -> tensor<16x16xi32>
    %offset = arith.addi %bases, %local : tensor<16x16xi32>
    %ap = tt.splat %A : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %pa = tt.addptr %ap, %offset : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %a = tt.load %pa : tensor<16x16x!tt.ptr<f16>>
    %br = tt.expand_dims %c {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %bs = arith.constant dense<16> : tensor<16x1xi32>
    %bro = arith.muli %br, %bs : tensor<16x1xi32>
    %brr = tt.broadcast %bro : tensor<16x1xi32> -> tensor<16x16xi32>
    %bcc = tt.broadcast %c2d : tensor<1x16xi32> -> tensor<16x16xi32>
    %boff = arith.addi %brr, %bcc : tensor<16x16xi32>
    %bp = tt.splat %B : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %pb = tt.addptr %bp, %boff : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %upper = arith.constant 2 : i32
    %zero0 = arith.constant dense<0.0> : tensor<16x16xf32>
    %loop = scf.for %iv = %c0 to %upper step %c1 iter_args(%acc0 = %zero0) -> (tensor<16x16xf32>) : i32 {
      %b = tt.load %pb : tensor<16x16x!tt.ptr<f16>>
      %dot0_0 = tt.dot %a, %b, %acc0 : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
      scf.yield %dot0_0 : tensor<16x16xf32>
    }

    %view = tt.reshape %loop : tensor<16x16xf32> -> tensor<4x64xf32>
    %transposed = tt.trans %view {order = array<i32: 1, 0>} : tensor<4x64xf32> -> tensor<64x4xf32>
    %output = tt.reshape %transposed : tensor<64x4xf32> -> tensor<16x16xf32>
    %cp = tt.splat %C : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %pc = tt.addptr %cp, %offset : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %pc, %output : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

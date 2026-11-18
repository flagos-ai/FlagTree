// RUN: triton-opt --split-input-file %s -graph-optimize='rule-mask=8' -canonicalize -cse --verify-each -o %t
// RUN: FileCheck %s < %t
// RUN: triton-opt --split-input-file %t -graph-optimize='rule-mask=8' -canonicalize -cse --verify-each -o %t.twice
// RUN: diff %t %t.twice
// RUN: triton-opt --split-input-file %s -graph-optimize='rule-mask=8 compile-mode=simt_only' -canonicalize -cse -o %t.simt
// RUN: triton-opt --split-input-file %s -canonicalize -cse -o %t.control
// RUN: diff %t.control %t.simt

// No static grid hint is required. The launcher supplies the original extent
// and shrinks the x-grid; per-row memory masks protect a partial final group.
// CHECK: hacc.program_grid_transforms = {{.*}}factor = 8 : i64
// CHECK-LABEL: tt.func @runtime_grid(
// CHECK-SAME: %[[EXTENT:arg[0-9]+]]: i32, %{{arg[0-9]+}}: i32)
// CHECK: %[[LIMIT:.*]] = tt.splat %[[EXTENT]] : i32 -> tensor<8xi32>
// CHECK: arith.cmpi slt, {{.*}}, %[[LIMIT]] : tensor<8xi32>
// CHECK: tt.load {{.*}}, %{{.*}}, %{{.*}} : tensor<8x16x16x!tt.ptr<f16>>
// CHECK: tt.dot {{.*}} : tensor<128x16xf16> * tensor<16x16xf16> -> tensor<128x16xf32>
// CHECK: tt.store {{.*}}, %{{.*}}, %{{.*}} : tensor<8x16x16x!tt.ptr<f32>>


module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func @runtime_grid(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %C: !tt.ptr<f32>, %count: i32) {
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

    %cp = tt.splat %C : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %pc = tt.addptr %cp, %offset : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %pc, %loop : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Dynamic tail rows must preserve the original memory mask and clamp the
// last row before evaluating the shared pid-dependent loop bound.
// CHECK-LABEL: tt.func @runtime_grid_masked_bounds(
// CHECK-SAME: %[[COLUMNS:arg[0-9]+]]: i32, %[[GRID:arg[0-9]+]]: i32, %{{arg[0-9]+}}: i32)
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[ONES:.*]] = arith.constant dense<1> : tensor<8xi32>
// CHECK: %[[END:.*]] = arith.subi %[[GRID]], %[[ONE]] : i32
// CHECK: %[[LAST:.*]] = arith.minsi %{{.*}}, %[[END]] : i32
// CHECK: %[[ROWS:.*]] = arith.addi {{.*}} : tensor<8xi32>
// CHECK: %[[GRID_TILE:.*]] = tt.splat %[[GRID]] : i32 -> tensor<8xi32>
// CHECK: %[[VALID:.*]] = arith.cmpi slt, %[[ROWS]], %[[GRID_TILE]] : tensor<8xi32>
// CHECK: %[[COUNT:.*]] = tt.splat %[[COLUMNS]] : i32 -> tensor<16x16xi32>
// CHECK: %[[ORIGINAL:.*]] = arith.cmpi slt, %{{.*}}, %[[COUNT]] : tensor<16x16xi32>
// CHECK: %[[VALID1:.*]] = tt.expand_dims %[[VALID]] {axis = 1 : i32}
// CHECK: %[[VALID2:.*]] = tt.expand_dims %[[VALID1]] {axis = 2 : i32}
// CHECK: %[[VALID_TILE:.*]] = tt.broadcast %[[VALID2]]
// CHECK: %[[ORIGINAL1:.*]] = tt.expand_dims %[[ORIGINAL]] {axis = 0 : i32}
// CHECK: %[[ORIGINAL_TILE:.*]] = tt.broadcast %[[ORIGINAL1]]
// CHECK: %[[MASK:.*]] = arith.andi %[[VALID_TILE]], %[[ORIGINAL_TILE]] : tensor<8x16x16xi1>
// CHECK: tt.load %{{.*}}, %[[MASK]], %{{.*}} : tensor<8x16x16x!tt.ptr<f16>>
// CHECK: %[[UPPER:.*]] = arith.addi %[[LAST]], %[[ONE]] : i32
// CHECK: %[[ROW_UPPER:.*]] = arith.addi %[[ROWS]], %[[ONES]] : tensor<8xi32>
// CHECK: %[[RESULT:.*]] = scf.for %[[IV:.*]] = %{{.*}} to %[[UPPER]] step %[[ONE]] iter_args(%[[ACC:.*]] = %{{.*}})
// CHECK: %[[IV_TILE:.*]] = tt.splat %[[IV]] : i32 -> tensor<8xi32>
// CHECK: %[[ACTIVE:.*]] = arith.cmpi slt, %[[IV_TILE]], %[[ROW_UPPER]] : tensor<8xi32>
// CHECK: %[[ACTIVE1:.*]] = tt.expand_dims %[[ACTIVE]] {axis = 1 : i32}
// CHECK: %[[ACTIVE2:.*]] = tt.expand_dims %[[ACTIVE1]] {axis = 2 : i32}
// CHECK: %[[ACTIVE_TILE:.*]] = tt.broadcast %[[ACTIVE2]]
// CHECK: %[[NEXT:.*]] = arith.select %[[ACTIVE_TILE]], %{{.*}}, %[[ACC]]
// CHECK: scf.yield %[[NEXT]]
// CHECK: tt.store %{{.*}}, %[[RESULT]], %[[MASK]] : tensor<8x16x16x!tt.ptr<f32>>

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func @runtime_grid_masked_bounds(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %C: !tt.ptr<f32>, %count: i32) {
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
    %count_tile = tt.splat %count : i32 -> tensor<16x16xi32>
    %mask = arith.cmpi slt, %cc, %count_tile : tensor<16x16xi32>
    %zero = arith.constant dense<0.0> : tensor<16x16xf16>
    %a = tt.load %pa, %mask, %zero : tensor<16x16x!tt.ptr<f16>>
    %br = tt.expand_dims %c {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %bs = arith.constant dense<16> : tensor<16x1xi32>
    %bro = arith.muli %br, %bs : tensor<16x1xi32>
    %brr = tt.broadcast %bro : tensor<16x1xi32> -> tensor<16x16xi32>
    %bcc = tt.broadcast %c2d : tensor<1x16xi32> -> tensor<16x16xi32>
    %boff = arith.addi %brr, %bcc : tensor<16x16xi32>
    %bp = tt.splat %B : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %pb = tt.addptr %bp, %boff : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %upper = arith.addi %pid, %c1 : i32
    %zero0 = arith.constant dense<0.0> : tensor<16x16xf32>
    %loop = scf.for %iv = %c0 to %upper step %c1 iter_args(%acc0 = %zero0) -> (tensor<16x16xf32>) : i32 {
      %b = tt.load %pb : tensor<16x16x!tt.ptr<f16>>
      %dot0_0 = tt.dot %a, %b, %acc0 : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
      scf.yield %dot0_0 : tensor<16x16xf32>
    }

    %cp = tt.splat %C : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %pc = tt.addptr %cp, %offset : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %pc, %loop, %mask : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

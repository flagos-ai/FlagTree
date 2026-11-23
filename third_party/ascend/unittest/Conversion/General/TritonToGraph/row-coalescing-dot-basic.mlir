// RUN: triton-opt --split-input-file --verify-each %s -graph-optimize='rule-mask=8' -canonicalize -cse -o %t
// RUN: FileCheck %s < %t
// RUN: triton-opt --split-input-file --verify-each %s -graph-optimize='rule-mask=8 compile-mode=simt_only' -canonicalize -cse -o %t.simt
// RUN: triton-opt --split-input-file --verify-each %t -graph-optimize='rule-mask=8' -canonicalize -cse -o %t.twice
// RUN: diff %t %t.twice
// RUN: triton-opt --split-input-file --verify-each %s -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.disabled
// RUN: triton-opt --split-input-file %s -canonicalize -cse -o %t.control
// RUN: diff %t.control %t.disabled
// RUN: diff %t.control %t.simt

// The same valid candidates must remain unchanged on non-Ascend 950 targets.
// RUN: sed 's/Ascend950PR_9579/Ascend910B2/g' %s > %t.a2.mlir
// RUN: triton-opt --split-input-file %t.a2.mlir -canonicalize -cse -o %t.a2.control
// RUN: triton-opt --split-input-file --verify-each %t.a2.mlir -graph-optimize='rule-mask=8' -canonicalize -cse -o %t.a2
// RUN: diff %t.a2.control %t.a2
// RUN: sed 's/Ascend950PR_9579/Ascend910_9391/g' %s > %t.a3.mlir
// RUN: triton-opt --split-input-file %t.a3.mlir -canonicalize -cse -o %t.a3.control
// RUN: triton-opt --split-input-file --verify-each %t.a3.mlir -graph-optimize='rule-mask=8' -canonicalize -cse -o %t.a3
// RUN: diff %t.a3.control %t.a3
// Missing target information must not enable the Ascend 950-only pattern.
// RUN: sed 's/hacc.target = #hacc.target<"Ascend950PR_9579">, //g' %s > %t.no-target.mlir
// RUN: triton-opt --split-input-file %t.no-target.mlir -canonicalize -cse -o %t.no-target.control
// RUN: triton-opt --split-input-file --verify-each %t.no-target.mlir -graph-optimize='rule-mask=8' -canonicalize -cse -o %t.no-target
// RUN: diff %t.no-target.control %t.no-target

// Merge independent dot rows while keeping the RHS shared; preserve per-row trip counts.
// CHECK: module attributes {{.*}}hacc.coalesce_factor = 8 : i32
// CHECK-LABEL: tt.func @one_dot_m16(
// CHECK: tt.load {{.*}} : tensor<8x16x16x!tt.ptr<f16>>
// CHECK: scf.for
// CHECK: %[[RHS:.*]] = tt.load {{.*}} : tensor<16x16x!tt.ptr<f16>>
// CHECK: tt.dot %{{.*}}, %[[RHS]], %{{.*}} : tensor<128x16xf16> * tensor<16x16xf16> -> tensor<128x16xf32>
// CHECK: arith.cmpi slt, {{.*}} : tensor<8xi32>
// CHECK: arith.select {{.*}} : tensor<8x16x16xi1>, tensor<8x16x16xf32>
// CHECK: tt.store {{.*}} : tensor<8x16x16x!tt.ptr<f32>>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hacc.grid_num_tiles = 64 : i32} {
  tt.func @one_dot_m16(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %C: !tt.ptr<f32>, %count: i32) {
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

// CHECK: module attributes {{.*}}hacc.coalesce_factor = 8 : i32
// CHECK-LABEL: tt.func @two_accumulators(
// CHECK: tt.load {{.*}} : tensor<8x16x16x!tt.ptr<f16>>
// CHECK: scf.for
// CHECK: %[[RHS:.*]] = tt.load {{.*}} : tensor<16x16x!tt.ptr<f16>>
// CHECK: tt.dot %{{.*}}, %[[RHS]], %{{.*}} : tensor<128x16xf16> * tensor<16x16xf16> -> tensor<128x16xf32>
// CHECK: arith.cmpi slt, {{.*}} : tensor<8xi32>
// CHECK: arith.select {{.*}} : tensor<8x16x16xi1>, tensor<8x16x16xf32>
// CHECK: tt.store {{.*}} : tensor<8x16x16x!tt.ptr<f32>>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hacc.grid_num_tiles = 64 : i32} {
  tt.func @two_accumulators(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %C: !tt.ptr<f32>, %count: i32) {
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
    %zero1 = arith.constant dense<1.0> : tensor<16x16xf32>
    %loop:2 = scf.for %iv = %c0 to %upper step %c1 iter_args(%acc0 = %zero0, %acc1 = %zero1) -> (tensor<16x16xf32>, tensor<16x16xf32>) : i32 {
      %b = tt.load %pb : tensor<16x16x!tt.ptr<f16>>
      %dot0_0 = tt.dot %a, %b, %acc0 : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
      %dot1_0 = tt.dot %a, %b, %acc1 : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
      scf.yield %dot0_0, %dot1_0 : tensor<16x16xf32>, tensor<16x16xf32>
    }
    %sum1 = arith.addf %loop#0, %loop#1 : tensor<16x16xf32>
    %cp = tt.splat %C : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %pc = tt.addptr %cp, %offset : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %pc, %sum1 : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// CHECK: module attributes {{.*}}hacc.coalesce_factor = 8 : i32
// CHECK-LABEL: tt.func @commuted_monotonic_bound(
// CHECK: tt.load {{.*}} : tensor<8x16x16x!tt.ptr<f16>>
// CHECK: scf.for
// CHECK: %[[RHS:.*]] = tt.load {{.*}} : tensor<16x16x!tt.ptr<f16>>
// CHECK: tt.dot %{{.*}}, %[[RHS]], %{{.*}} : tensor<128x16xf16> * tensor<16x16xf16> -> tensor<128x16xf32>
// CHECK: arith.cmpi slt, {{.*}} : tensor<8xi32>
// CHECK: arith.select {{.*}} : tensor<8x16x16xi1>, tensor<8x16x16xf32>
// CHECK: tt.store {{.*}} : tensor<8x16x16x!tt.ptr<f32>>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hacc.grid_num_tiles = 64 : i32} {
  tt.func @commuted_monotonic_bound(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %C: !tt.ptr<f32>, %count: i32) {
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
    %upper = arith.addi %c2, %pid : i32
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

// CHECK: module attributes {{.*}}hacc.coalesce_factor = 4 : i32
// CHECK-LABEL: tt.func @smaller_exact_grid_group(
// CHECK: tt.load {{.*}} : tensor<4x16x16x!tt.ptr<f16>>
// CHECK: scf.for
// CHECK: %[[RHS:.*]] = tt.load {{.*}} : tensor<16x16x!tt.ptr<f16>>
// CHECK: tt.dot %{{.*}}, %[[RHS]], %{{.*}} : tensor<64x16xf16> * tensor<16x16xf16> -> tensor<64x16xf32>
// CHECK: arith.cmpi slt, {{.*}} : tensor<4xi32>
// CHECK: arith.select {{.*}} : tensor<4x16x16xi1>, tensor<4x16x16xf32>
// CHECK: tt.store {{.*}} : tensor<4x16x16x!tt.ptr<f32>>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hacc.grid_num_tiles = 12 : i32} {
  tt.func @smaller_exact_grid_group(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %C: !tt.ptr<f32>, %count: i32) {
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

// CHECK: module attributes {{.*}}hacc.coalesce_factor = 8 : i32
// CHECK-LABEL: tt.func @bf16_dot(
// CHECK: tt.load {{.*}} : tensor<8x16x16x!tt.ptr<bf16>>
// CHECK: scf.for
// CHECK: %[[RHS:.*]] = tt.load {{.*}} : tensor<16x16x!tt.ptr<bf16>>
// CHECK: tt.dot %{{.*}}, %[[RHS]], %{{.*}} : tensor<128x16xbf16> * tensor<16x16xbf16> -> tensor<128x16xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hacc.grid_num_tiles = 64 : i32} {
  tt.func @bf16_dot(%A: !tt.ptr<bf16>, %B: !tt.ptr<bf16>, %C: !tt.ptr<f32>, %count: i32) {
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
    %ap = tt.splat %A : !tt.ptr<bf16> -> tensor<16x16x!tt.ptr<bf16>>
    %pa = tt.addptr %ap, %offset : tensor<16x16x!tt.ptr<bf16>>, tensor<16x16xi32>
    %a = tt.load %pa : tensor<16x16x!tt.ptr<bf16>>
    %br = tt.expand_dims %c {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %bs = arith.constant dense<16> : tensor<16x1xi32>
    %bro = arith.muli %br, %bs : tensor<16x1xi32>
    %brr = tt.broadcast %bro : tensor<16x1xi32> -> tensor<16x16xi32>
    %bcc = tt.broadcast %c2d : tensor<1x16xi32> -> tensor<16x16xi32>
    %boff = arith.addi %brr, %bcc : tensor<16x16xi32>
    %bp = tt.splat %B : !tt.ptr<bf16> -> tensor<16x16x!tt.ptr<bf16>>
    %pb = tt.addptr %bp, %boff : tensor<16x16x!tt.ptr<bf16>>, tensor<16x16xi32>
    %upper = arith.constant 2 : i32
    %zero0 = arith.constant dense<0.0> : tensor<16x16xf32>
    %loop = scf.for %iv = %c0 to %upper step %c1 iter_args(%acc0 = %zero0) -> (tensor<16x16xf32>) : i32 {
      %b = tt.load %pb : tensor<16x16x!tt.ptr<bf16>>
      %dot0_0 = tt.dot %a, %b, %acc0 : tensor<16x16xbf16> * tensor<16x16xbf16> -> tensor<16x16xf32>
      scf.yield %dot0_0 : tensor<16x16xf32>
    }

    %cp = tt.splat %C : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %pc = tt.addptr %cp, %offset : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %pc, %loop : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

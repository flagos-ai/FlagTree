// RUN: triton-opt %s --graph-optimize='rule-mask=256' --split-input-file \
// RUN: | FileCheck %s
// RUN: triton-opt %s --graph-optimize='rule-mask=0' --split-input-file \
// RUN: | FileCheck %s --check-prefix=DISABLED

// ConvertModuloToMask drops a modulo that only wraps tile addresses, so the
// addresses stay linear, and gives the loads a boundary mask so the lanes that
// used to wrap read in-bounds zeros instead.  rule-mask=256 isolates the rule so
// nothing below can be attributed to another graph rule.
//
// The rewrite no longer consults tt.divisibility: the boundary mask is injected
// unconditionally, which is what keeps the transfer inside the tensor even when
// nothing can be proven about the bound.

// -----
// The load has no mask of its own, so it gets the boundary mask plus a zero
// fill.  This is the fused_moe weight-tile shape.
// CHECK-LABEL: tt.func public @modulo_remsi_unmasked_load
// CHECK-NOT: arith.remsi
// CHECK: %[[BOUND:.*]] = arith.cmpi slt
// CHECK: %[[EXPAND:.*]] = tt.expand_dims %[[BOUND]] {axis = 0 : i32}
// CHECK: %[[MASK:.*]] = tt.broadcast %[[EXPAND]]
// CHECK: %[[FILL:.*]] = arith.constant dense<0.000000e+00> : tensor<32x16xf16>
// CHECK: tt.load %{{.*}}, %[[MASK]], %[[FILL]] : tensor<32x16x!tt.ptr<f16>>
// CHECK-NOT: arith.remsi
// DISABLED-LABEL: tt.func public @modulo_remsi_unmasked_load
// DISABLED: arith.remsi
tt.func public @modulo_remsi_unmasked_load(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                          %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                          %n: i32) {
  %c16 = arith.constant 16 : i32
  %pid = tt.get_program_id x : i32
  %blk = arith.muli %pid, %c16 : i32
  %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %offs = arith.addi %blk_splat, %range : tensor<16xi32>
  %n_splat = tt.splat %n : i32 -> tensor<16xi32>
  %rem = arith.remsi %offs, %n_splat : tensor<16xi32>

  %rem_expd = tt.expand_dims %rem {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %rem_bc = tt.broadcast %rem_expd : tensor<1x16xi32> -> tensor<32x16xi32>
  %row = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %row_expd = tt.expand_dims %row {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %row_stride = arith.constant dense<16> : tensor<32x1xi32>
  %row_scaled = arith.muli %row_expd, %row_stride : tensor<32x1xi32>
  %row_bc = tt.broadcast %row_scaled : tensor<32x1xi32> -> tensor<32x16xi32>
  %addr = arith.addi %row_bc, %rem_bc : tensor<32x16xi32>
  %base_splat = tt.splat %base : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>>
  %ptrs = tt.addptr %base_splat, %addr : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>
  %val = tt.load %ptrs : tensor<32x16x!tt.ptr<f16>>

  %guard = arith.cmpi slt, %offs, %n_splat : tensor<16xi32>
  %guard_expd = tt.expand_dims %guard {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
  %guard_bc = tt.broadcast %guard_expd : tensor<1x16xi1> -> tensor<32x16xi1>
  %col_expd = tt.expand_dims %offs {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %col_bc = tt.broadcast %col_expd : tensor<1x16xi32> -> tensor<32x16xi32>
  %out_addr = arith.addi %row_bc, %col_bc : tensor<32x16xi32>
  %dst_splat = tt.splat %dst : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>>
  %out_ptrs = tt.addptr %dst_splat, %out_addr : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>
  tt.store %out_ptrs, %val, %guard_bc : tensor<32x16x!tt.ptr<f16>>
  tt.return
}

// -----
// A load that already carries a mask keeps it: the boundary mask is combined
// with arith.andi and the existing fill is left untouched.
// CHECK-LABEL: tt.func public @modulo_remsi_masked_load
// CHECK-NOT: arith.remsi
// CHECK: %[[BOUND:.*]] = arith.cmpi slt
// CHECK: %[[EXPAND:.*]] = tt.expand_dims %[[BOUND]] {axis = 0 : i32}
// CHECK: %[[MASK:.*]] = tt.broadcast %[[EXPAND]]
// CHECK: %[[BOTH:.*]] = arith.andi %{{.*}}, %[[MASK]]
// CHECK: tt.load %{{.*}}, %[[BOTH]], %{{.*}} : tensor<32x16x!tt.ptr<f16>>
// CHECK-NOT: arith.remsi
// DISABLED-LABEL: tt.func public @modulo_remsi_masked_load
// DISABLED: arith.remsi
// DISABLED-NOT: arith.andi
tt.func public @modulo_remsi_masked_load(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                        %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                        %n: i32, %k: i32) {
  %c16 = arith.constant 16 : i32
  %pid = tt.get_program_id x : i32
  %blk = arith.muli %pid, %c16 : i32
  %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %offs = arith.addi %blk_splat, %range : tensor<16xi32>
  %n_splat = tt.splat %n : i32 -> tensor<16xi32>
  %rem = arith.remsi %offs, %n_splat : tensor<16xi32>

  %rem_expd = tt.expand_dims %rem {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %rem_bc = tt.broadcast %rem_expd : tensor<1x16xi32> -> tensor<32x16xi32>
  %row = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %row_expd = tt.expand_dims %row {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %row_stride = arith.constant dense<16> : tensor<32x1xi32>
  %row_scaled = arith.muli %row_expd, %row_stride : tensor<32x1xi32>
  %row_bc = tt.broadcast %row_scaled : tensor<32x1xi32> -> tensor<32x16xi32>
  %addr = arith.addi %row_bc, %rem_bc : tensor<32x16xi32>
  %base_splat = tt.splat %base : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>>
  %ptrs = tt.addptr %base_splat, %addr : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>

  // The row axis has its own boundary mask, unrelated to the modulo.
  %k_splat = tt.splat %k : i32 -> tensor<32x1xi32>
  %row_guard = arith.cmpi slt, %row_expd, %k_splat : tensor<32x1xi32>
  %row_guard_bc = tt.broadcast %row_guard : tensor<32x1xi1> -> tensor<32x16xi1>
  %fill = arith.constant dense<1.000000e+00> : tensor<32x16xf16>
  %val = tt.load %ptrs, %row_guard_bc, %fill : tensor<32x16x!tt.ptr<f16>>

  %guard = arith.cmpi slt, %offs, %n_splat : tensor<16xi32>
  %guard_expd = tt.expand_dims %guard {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
  %guard_bc = tt.broadcast %guard_expd : tensor<1x16xi1> -> tensor<32x16xi1>
  %col_expd = tt.expand_dims %offs {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %col_bc = tt.broadcast %col_expd : tensor<1x16xi32> -> tensor<32x16xi32>
  %out_addr = arith.addi %row_bc, %col_bc : tensor<32x16xi32>
  %dst_splat = tt.splat %dst : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>>
  %out_ptrs = tt.addptr %dst_splat, %out_addr : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>
  tt.store %out_ptrs, %val, %guard_bc : tensor<32x16x!tt.ptr<f16>>
  tt.return
}

// -----
// Canonicalization expands a modulo into divsi/muli/subi.  All three ops go
// away together, since the identity x == (x / d) * d + x % d makes the subi
// exactly the modulo.
// CHECK-LABEL: tt.func public @modulo_expanded_into_divsi_muli_subi
// CHECK-NOT: arith.divsi
// CHECK-NOT: arith.remsi
// CHECK: %[[BOUND:.*]] = arith.cmpi slt
// CHECK: tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<32x16x!tt.ptr<f16>>
// CHECK-NOT: arith.divsi
// DISABLED-LABEL: tt.func public @modulo_expanded_into_divsi_muli_subi
// DISABLED: arith.divsi
tt.func public @modulo_expanded_into_divsi_muli_subi(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                                    %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                                    %n: i32) {
  %c16 = arith.constant 16 : i32
  %pid = tt.get_program_id x : i32
  %blk = arith.muli %pid, %c16 : i32
  %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %offs = arith.addi %blk_splat, %range : tensor<16xi32>
  %n_splat = tt.splat %n : i32 -> tensor<16xi32>
  %quot = arith.divsi %offs, %n_splat : tensor<16xi32>
  %prod = arith.muli %quot, %n_splat : tensor<16xi32>
  %rem = arith.subi %offs, %prod : tensor<16xi32>

  %rem_expd = tt.expand_dims %rem {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %rem_bc = tt.broadcast %rem_expd : tensor<1x16xi32> -> tensor<32x16xi32>
  %row = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %row_expd = tt.expand_dims %row {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %row_stride = arith.constant dense<16> : tensor<32x1xi32>
  %row_scaled = arith.muli %row_expd, %row_stride : tensor<32x1xi32>
  %row_bc = tt.broadcast %row_scaled : tensor<32x1xi32> -> tensor<32x16xi32>
  %addr = arith.addi %row_bc, %rem_bc : tensor<32x16xi32>
  %base_splat = tt.splat %base : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>>
  %ptrs = tt.addptr %base_splat, %addr : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>
  %val = tt.load %ptrs : tensor<32x16x!tt.ptr<f16>>

  %guard = arith.cmpi slt, %offs, %n_splat : tensor<16xi32>
  %guard_expd = tt.expand_dims %guard {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
  %guard_bc = tt.broadcast %guard_expd : tensor<1x16xi1> -> tensor<32x16xi1>
  %col_expd = tt.expand_dims %offs {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %col_bc = tt.broadcast %col_expd : tensor<1x16xi32> -> tensor<32x16xi32>
  %out_addr = arith.addi %row_bc, %col_bc : tensor<32x16xi32>
  %dst_splat = tt.splat %dst : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>>
  %out_ptrs = tt.addptr %dst_splat, %out_addr : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>
  tt.store %out_ptrs, %val, %guard_bc : tensor<32x16x!tt.ptr<f16>>
  tt.return
}

// -----
// The store-mask guard is often written on the already expanded offset, so the
// comparison is found through tt.expand_dims as well.
// CHECK-LABEL: tt.func public @modulo_guard_through_expand_dims
// CHECK-NOT: arith.remsi
// CHECK: tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<32x16x!tt.ptr<f16>>
tt.func public @modulo_guard_through_expand_dims(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                                %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                                %n: i32) {
  %c16 = arith.constant 16 : i32
  %pid = tt.get_program_id x : i32
  %blk = arith.muli %pid, %c16 : i32
  %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %offs = arith.addi %blk_splat, %range : tensor<16xi32>
  %n_splat = tt.splat %n : i32 -> tensor<16xi32>
  %rem = arith.remsi %offs, %n_splat : tensor<16xi32>

  %rem_expd = tt.expand_dims %rem {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %rem_bc = tt.broadcast %rem_expd : tensor<1x16xi32> -> tensor<32x16xi32>
  %row = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %row_expd = tt.expand_dims %row {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %row_stride = arith.constant dense<16> : tensor<32x1xi32>
  %row_scaled = arith.muli %row_expd, %row_stride : tensor<32x1xi32>
  %row_bc = tt.broadcast %row_scaled : tensor<32x1xi32> -> tensor<32x16xi32>
  %addr = arith.addi %row_bc, %rem_bc : tensor<32x16xi32>
  %base_splat = tt.splat %base : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>>
  %ptrs = tt.addptr %base_splat, %addr : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>
  %val = tt.load %ptrs : tensor<32x16x!tt.ptr<f16>>

  %col_expd = tt.expand_dims %offs {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %n_splat_2d = tt.splat %n : i32 -> tensor<1x16xi32>
  %guard = arith.cmpi slt, %col_expd, %n_splat_2d : tensor<1x16xi32>
  %guard_bc = tt.broadcast %guard : tensor<1x16xi1> -> tensor<32x16xi1>
  %col_bc = tt.broadcast %col_expd : tensor<1x16xi32> -> tensor<32x16xi32>
  %out_addr = arith.addi %row_bc, %col_bc : tensor<32x16xi32>
  %dst_splat = tt.splat %dst : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>>
  %out_ptrs = tt.addptr %dst_splat, %out_addr : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>
  tt.store %out_ptrs, %val, %guard_bc : tensor<32x16x!tt.ptr<f16>>
  tt.return
}

// -----
// The wrapped index can address rows instead of columns.  The mask is expanded
// on the same axis, so it varies along the same dimension of the load.
// CHECK-LABEL: tt.func public @modulo_on_row_axis
// CHECK-NOT: arith.remsi
// CHECK: %[[BOUND:.*]] = arith.cmpi slt
// CHECK: %[[EXPAND:.*]] = tt.expand_dims %[[BOUND]] {axis = 1 : i32}
// CHECK: %[[MASK:.*]] = tt.broadcast %[[EXPAND]]
// CHECK: tt.load %{{.*}}, %[[MASK]], %{{.*}} : tensor<16x32x!tt.ptr<f16>>
tt.func public @modulo_on_row_axis(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                   %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                   %m: i32) {
  %c16 = arith.constant 16 : i32
  %pid = tt.get_program_id x : i32
  %blk = arith.muli %pid, %c16 : i32
  %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %offs = arith.addi %blk_splat, %range : tensor<16xi32>
  %m_splat = tt.splat %m : i32 -> tensor<16xi32>
  %rem = arith.remsi %offs, %m_splat : tensor<16xi32>

  %rem_expd = tt.expand_dims %rem {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %row_stride = arith.constant dense<32> : tensor<16x1xi32>
  %rem_scaled = arith.muli %rem_expd, %row_stride : tensor<16x1xi32>
  %rem_bc = tt.broadcast %rem_scaled : tensor<16x1xi32> -> tensor<16x32xi32>
  %col = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %col_expd = tt.expand_dims %col {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
  %col_bc = tt.broadcast %col_expd : tensor<1x32xi32> -> tensor<16x32xi32>
  %addr = arith.addi %rem_bc, %col_bc : tensor<16x32xi32>
  %base_splat = tt.splat %base : !tt.ptr<f16> -> tensor<16x32x!tt.ptr<f16>>
  %ptrs = tt.addptr %base_splat, %addr : tensor<16x32x!tt.ptr<f16>>, tensor<16x32xi32>
  %val = tt.load %ptrs : tensor<16x32x!tt.ptr<f16>>

  %guard = arith.cmpi slt, %offs, %m_splat : tensor<16xi32>
  %guard_expd = tt.expand_dims %guard {axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
  %guard_bc = tt.broadcast %guard_expd : tensor<16x1xi1> -> tensor<16x32xi1>
  %out_row = tt.expand_dims %offs {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %out_scaled = arith.muli %out_row, %row_stride : tensor<16x1xi32>
  %out_bc = tt.broadcast %out_scaled : tensor<16x1xi32> -> tensor<16x32xi32>
  %out_addr = arith.addi %out_bc, %col_bc : tensor<16x32xi32>
  %dst_splat = tt.splat %dst : !tt.ptr<f16> -> tensor<16x32x!tt.ptr<f16>>
  %out_ptrs = tt.addptr %dst_splat, %out_addr : tensor<16x32x!tt.ptr<f16>>, tensor<16x32xi32>
  tt.store %out_ptrs, %val, %guard_bc : tensor<16x32x!tt.ptr<f16>>
  tt.return
}

// -----
// The address is normally carried around the reduction loop, so the walk has to
// follow scf.for iter_args to reach the load it feeds.
// CHECK-LABEL: tt.func public @modulo_load_inside_loop
// CHECK-NOT: arith.remsi
// CHECK: %[[BOUND:.*]] = arith.cmpi slt
// CHECK: scf.for
// CHECK: %[[EXPAND:.*]] = tt.expand_dims %[[BOUND]] {axis = 0 : i32}
// CHECK: %[[MASK:.*]] = tt.broadcast %[[EXPAND]]
// CHECK: tt.load %{{.*}}, %[[MASK]], %{{.*}} : tensor<32x16x!tt.ptr<f16>>
tt.func public @modulo_load_inside_loop(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                       %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                       %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %c16 = arith.constant 16 : i32
  %pid = tt.get_program_id x : i32
  %blk = arith.muli %pid, %c16 : i32
  %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %offs = arith.addi %blk_splat, %range : tensor<16xi32>
  %n_splat = tt.splat %n : i32 -> tensor<16xi32>
  %rem = arith.remsi %offs, %n_splat : tensor<16xi32>

  %rem_expd = tt.expand_dims %rem {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %rem_bc = tt.broadcast %rem_expd : tensor<1x16xi32> -> tensor<32x16xi32>
  %row = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %row_expd = tt.expand_dims %row {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %row_stride = arith.constant dense<16> : tensor<32x1xi32>
  %row_scaled = arith.muli %row_expd, %row_stride : tensor<32x1xi32>
  %row_bc = tt.broadcast %row_scaled : tensor<32x1xi32> -> tensor<32x16xi32>
  %addr = arith.addi %row_bc, %rem_bc : tensor<32x16xi32>
  %base_splat = tt.splat %base : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>>
  %ptrs = tt.addptr %base_splat, %addr : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>
  %step = arith.constant dense<512> : tensor<32x16xi32>
  %zeros = arith.constant dense<0.000000e+00> : tensor<32x16xf16>

  %loop:2 = scf.for %iv = %c0 to %c4 step %c1 iter_args(%acc = %zeros, %cur = %ptrs) -> (tensor<32x16xf16>, tensor<32x16x!tt.ptr<f16>>) : i32 {
    %tile = tt.load %cur : tensor<32x16x!tt.ptr<f16>>
    %next_acc = arith.addf %acc, %tile : tensor<32x16xf16>
    %next_ptrs = tt.addptr %cur, %step : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>
    scf.yield %next_acc, %next_ptrs : tensor<32x16xf16>, tensor<32x16x!tt.ptr<f16>>
  }

  %guard = arith.cmpi slt, %offs, %n_splat : tensor<16xi32>
  %guard_expd = tt.expand_dims %guard {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
  %guard_bc = tt.broadcast %guard_expd : tensor<1x16xi1> -> tensor<32x16xi1>
  %col_expd = tt.expand_dims %offs {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %col_bc = tt.broadcast %col_expd : tensor<1x16xi32> -> tensor<32x16xi32>
  %out_addr = arith.addi %row_bc, %col_bc : tensor<32x16xi32>
  %dst_splat = tt.splat %dst : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>>
  %out_ptrs = tt.addptr %dst_splat, %out_addr : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>
  tt.store %out_ptrs, %loop#0, %guard_bc : tensor<32x16x!tt.ptr<f16>>
  tt.return
}

// -----
// Negative: a pointer carried by an outer loop and then by an inner loop is
// deliberately not linearized.  The direct-memory lowerer must preserve the
// inner result as the outer loop state before this case can become eligible.
// CHECK-LABEL: tt.func public @modulo_load_inside_nested_loop
// CHECK: arith.remsi
tt.func public @modulo_load_inside_nested_loop(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                              %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                              %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %c16 = arith.constant 16 : i32
  %pid = tt.get_program_id x : i32
  %blk = arith.muli %pid, %c16 : i32
  %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %offs = arith.addi %blk_splat, %range : tensor<16xi32>
  %n_splat = tt.splat %n : i32 -> tensor<16xi32>
  %rem = arith.remsi %offs, %n_splat : tensor<16xi32>

  %rem_expd = tt.expand_dims %rem {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %rem_bc = tt.broadcast %rem_expd : tensor<1x16xi32> -> tensor<32x16xi32>
  %row = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %row_expd = tt.expand_dims %row {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %row_stride = arith.constant dense<16> : tensor<32x1xi32>
  %row_scaled = arith.muli %row_expd, %row_stride : tensor<32x1xi32>
  %row_bc = tt.broadcast %row_scaled : tensor<32x1xi32> -> tensor<32x16xi32>
  %addr = arith.addi %row_bc, %rem_bc : tensor<32x16xi32>
  %base_splat = tt.splat %base : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>>
  %ptrs = tt.addptr %base_splat, %addr : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>
  %step = arith.constant dense<512> : tensor<32x16xi32>
  %zeros = arith.constant dense<0.000000e+00> : tensor<32x16xf16>

  %outer:2 = scf.for %outer_iv = %c0 to %c4 step %c1 iter_args(%outer_acc = %zeros, %outer_ptrs = %ptrs) -> (tensor<32x16xf16>, tensor<32x16x!tt.ptr<f16>>) : i32 {
    %inner:2 = scf.for %inner_iv = %c0 to %c4 step %c1 iter_args(%inner_acc = %outer_acc, %inner_ptrs = %outer_ptrs) -> (tensor<32x16xf16>, tensor<32x16x!tt.ptr<f16>>) : i32 {
      %tile = tt.load %inner_ptrs : tensor<32x16x!tt.ptr<f16>>
      %next_acc = arith.addf %inner_acc, %tile : tensor<32x16xf16>
      %next_ptrs = tt.addptr %inner_ptrs, %step : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>
      scf.yield %next_acc, %next_ptrs : tensor<32x16xf16>, tensor<32x16x!tt.ptr<f16>>
    }
    scf.yield %inner#0, %inner#1 : tensor<32x16xf16>, tensor<32x16x!tt.ptr<f16>>
  }

  %guard = arith.cmpi slt, %offs, %n_splat : tensor<16xi32>
  %guard_expd = tt.expand_dims %guard {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
  %guard_bc = tt.broadcast %guard_expd : tensor<1x16xi1> -> tensor<32x16xi1>
  %col_expd = tt.expand_dims %offs {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %col_bc = tt.broadcast %col_expd : tensor<1x16xi32> -> tensor<32x16xi32>
  %out_addr = arith.addi %row_bc, %col_bc : tensor<32x16xi32>
  %dst_splat = tt.splat %dst : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>>
  %out_ptrs = tt.addptr %dst_splat, %out_addr : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>
  tt.store %out_ptrs, %outer#0, %guard_bc : tensor<32x16x!tt.ptr<f16>>
  tt.return
}

// -----
// Negative: a compile-time constant divisor belongs to TritonToStructured,
// whose visitOperandRem keeps the wrap and re-expresses it as a strided access.
// That is exactly equivalent, so it is always preferable to discarding the wrap.
// CHECK-LABEL: tt.func public @modulo_skip_constant_divisor
// CHECK: arith.remsi
tt.func public @modulo_skip_constant_divisor(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                            %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
  %c16 = arith.constant 16 : i32
  %c1024 = arith.constant 1024 : i32
  %pid = tt.get_program_id x : i32
  %blk = arith.muli %pid, %c16 : i32
  %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %offs = arith.addi %blk_splat, %range : tensor<16xi32>
  %bound = tt.splat %c1024 : i32 -> tensor<16xi32>
  %rem = arith.remsi %offs, %bound : tensor<16xi32>

  %rem_expd = tt.expand_dims %rem {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %rem_bc = tt.broadcast %rem_expd : tensor<1x16xi32> -> tensor<32x16xi32>
  %row = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %row_expd = tt.expand_dims %row {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %row_stride = arith.constant dense<16> : tensor<32x1xi32>
  %row_scaled = arith.muli %row_expd, %row_stride : tensor<32x1xi32>
  %row_bc = tt.broadcast %row_scaled : tensor<32x1xi32> -> tensor<32x16xi32>
  %addr = arith.addi %row_bc, %rem_bc : tensor<32x16xi32>
  %base_splat = tt.splat %base : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>>
  %ptrs = tt.addptr %base_splat, %addr : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>
  %val = tt.load %ptrs : tensor<32x16x!tt.ptr<f16>>

  %guard = arith.cmpi slt, %offs, %bound : tensor<16xi32>
  %guard_expd = tt.expand_dims %guard {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
  %guard_bc = tt.broadcast %guard_expd : tensor<1x16xi1> -> tensor<32x16xi1>
  %col_expd = tt.expand_dims %offs {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %col_bc = tt.broadcast %col_expd : tensor<1x16xi32> -> tensor<32x16xi32>
  %out_addr = arith.addi %row_bc, %col_bc : tensor<32x16xi32>
  %dst_splat = tt.splat %dst : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>>
  %out_ptrs = tt.addptr %dst_splat, %out_addr : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>
  tt.store %out_ptrs, %val, %guard_bc : tensor<32x16x!tt.ptr<f16>>
  tt.return
}

// -----
// Negative: nothing proves that the kernel already discards the wrapped lanes,
// so the wrap has to stay.
// CHECK-LABEL: tt.func public @modulo_skip_without_store_mask_guard
// CHECK: arith.remsi
tt.func public @modulo_skip_without_store_mask_guard(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                                    %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                                    %n: i32) {
  %c16 = arith.constant 16 : i32
  %pid = tt.get_program_id x : i32
  %blk = arith.muli %pid, %c16 : i32
  %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %offs = arith.addi %blk_splat, %range : tensor<16xi32>
  %n_splat = tt.splat %n : i32 -> tensor<16xi32>
  %rem = arith.remsi %offs, %n_splat : tensor<16xi32>

  %rem_expd = tt.expand_dims %rem {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %rem_bc = tt.broadcast %rem_expd : tensor<1x16xi32> -> tensor<32x16xi32>
  %row = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %row_expd = tt.expand_dims %row {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %row_stride = arith.constant dense<16> : tensor<32x1xi32>
  %row_scaled = arith.muli %row_expd, %row_stride : tensor<32x1xi32>
  %row_bc = tt.broadcast %row_scaled : tensor<32x1xi32> -> tensor<32x16xi32>
  %addr = arith.addi %row_bc, %rem_bc : tensor<32x16xi32>
  %base_splat = tt.splat %base : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>>
  %ptrs = tt.addptr %base_splat, %addr : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>
  %val = tt.load %ptrs : tensor<32x16x!tt.ptr<f16>>

  %col_expd = tt.expand_dims %offs {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %col_bc = tt.broadcast %col_expd : tensor<1x16xi32> -> tensor<32x16xi32>
  %out_addr = arith.addi %row_bc, %col_bc : tensor<32x16xi32>
  %dst_splat = tt.splat %dst : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>>
  %out_ptrs = tt.addptr %dst_splat, %out_addr : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>
  tt.store %out_ptrs, %val : tensor<32x16x!tt.ptr<f16>>
  tt.return
}

// -----
// Negative: the guard compares against a different runtime scalar than the
// divisor, so it says nothing about the wrapped lanes.
// CHECK-LABEL: tt.func public @modulo_skip_mismatched_bound
// CHECK: arith.remsi
tt.func public @modulo_skip_mismatched_bound(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                            %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                            %n: i32, %other: i32) {
  %c16 = arith.constant 16 : i32
  %pid = tt.get_program_id x : i32
  %blk = arith.muli %pid, %c16 : i32
  %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %offs = arith.addi %blk_splat, %range : tensor<16xi32>
  %n_splat = tt.splat %n : i32 -> tensor<16xi32>
  %other_splat = tt.splat %other : i32 -> tensor<16xi32>
  %rem = arith.remsi %offs, %n_splat : tensor<16xi32>

  %rem_expd = tt.expand_dims %rem {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %rem_bc = tt.broadcast %rem_expd : tensor<1x16xi32> -> tensor<32x16xi32>
  %row = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %row_expd = tt.expand_dims %row {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %row_stride = arith.constant dense<16> : tensor<32x1xi32>
  %row_scaled = arith.muli %row_expd, %row_stride : tensor<32x1xi32>
  %row_bc = tt.broadcast %row_scaled : tensor<32x1xi32> -> tensor<32x16xi32>
  %addr = arith.addi %row_bc, %rem_bc : tensor<32x16xi32>
  %base_splat = tt.splat %base : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>>
  %ptrs = tt.addptr %base_splat, %addr : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>
  %val = tt.load %ptrs : tensor<32x16x!tt.ptr<f16>>

  %guard = arith.cmpi slt, %offs, %other_splat : tensor<16xi32>
  %guard_expd = tt.expand_dims %guard {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
  %guard_bc = tt.broadcast %guard_expd : tensor<1x16xi1> -> tensor<32x16xi1>
  %col_expd = tt.expand_dims %offs {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %col_bc = tt.broadcast %col_expd : tensor<1x16xi32> -> tensor<32x16xi32>
  %out_addr = arith.addi %row_bc, %col_bc : tensor<32x16xi32>
  %dst_splat = tt.splat %dst : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>>
  %out_ptrs = tt.addptr %dst_splat, %out_addr : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>
  tt.store %out_ptrs, %val, %guard_bc : tensor<32x16x!tt.ptr<f16>>
  tt.return
}

// -----
// Negative: the comparison exists but only selects data, so it does not prove
// that the wrapped lanes stay out of memory.
// CHECK-LABEL: tt.func public @modulo_skip_guard_not_on_a_store
// CHECK: arith.remsi
tt.func public @modulo_skip_guard_not_on_a_store(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                                %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                                %n: i32) {
  %c16 = arith.constant 16 : i32
  %pid = tt.get_program_id x : i32
  %blk = arith.muli %pid, %c16 : i32
  %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %offs = arith.addi %blk_splat, %range : tensor<16xi32>
  %n_splat = tt.splat %n : i32 -> tensor<16xi32>
  %rem = arith.remsi %offs, %n_splat : tensor<16xi32>

  %rem_expd = tt.expand_dims %rem {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %rem_bc = tt.broadcast %rem_expd : tensor<1x16xi32> -> tensor<32x16xi32>
  %row = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %row_expd = tt.expand_dims %row {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %row_stride = arith.constant dense<16> : tensor<32x1xi32>
  %row_scaled = arith.muli %row_expd, %row_stride : tensor<32x1xi32>
  %row_bc = tt.broadcast %row_scaled : tensor<32x1xi32> -> tensor<32x16xi32>
  %addr = arith.addi %row_bc, %rem_bc : tensor<32x16xi32>
  %base_splat = tt.splat %base : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>>
  %ptrs = tt.addptr %base_splat, %addr : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>
  %val = tt.load %ptrs : tensor<32x16x!tt.ptr<f16>>

  // The comparison only picks between two tiles; it never reaches a store mask.
  %guard = arith.cmpi slt, %offs, %n_splat : tensor<16xi32>
  %guard_expd = tt.expand_dims %guard {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
  %guard_bc = tt.broadcast %guard_expd : tensor<1x16xi1> -> tensor<32x16xi1>
  %zeros = arith.constant dense<0.000000e+00> : tensor<32x16xf16>
  %picked = arith.select %guard_bc, %val, %zeros : tensor<32x16xi1>, tensor<32x16xf16>
  %col_expd = tt.expand_dims %offs {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %col_bc = tt.broadcast %col_expd : tensor<1x16xi32> -> tensor<32x16xi32>
  %out_addr = arith.addi %row_bc, %col_bc : tensor<32x16xi32>
  %dst_splat = tt.splat %dst : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>>
  %out_ptrs = tt.addptr %dst_splat, %out_addr : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>
  tt.store %out_ptrs, %picked : tensor<32x16x!tt.ptr<f16>>
  tt.return
}

// -----
// Negative: the wrapped index also addresses the store.  Dropping the wrap
// would move where results are written, not just which data is read.
// CHECK-LABEL: tt.func public @modulo_skip_result_addresses_store
// CHECK: arith.remsi
tt.func public @modulo_skip_result_addresses_store(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                                  %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                                  %n: i32) {
  %c16 = arith.constant 16 : i32
  %pid = tt.get_program_id x : i32
  %blk = arith.muli %pid, %c16 : i32
  %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %offs = arith.addi %blk_splat, %range : tensor<16xi32>
  %n_splat = tt.splat %n : i32 -> tensor<16xi32>
  %rem = arith.remsi %offs, %n_splat : tensor<16xi32>

  %rem_expd = tt.expand_dims %rem {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %rem_bc = tt.broadcast %rem_expd : tensor<1x16xi32> -> tensor<32x16xi32>
  %row = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %row_expd = tt.expand_dims %row {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %row_stride = arith.constant dense<16> : tensor<32x1xi32>
  %row_scaled = arith.muli %row_expd, %row_stride : tensor<32x1xi32>
  %row_bc = tt.broadcast %row_scaled : tensor<32x1xi32> -> tensor<32x16xi32>
  %addr = arith.addi %row_bc, %rem_bc : tensor<32x16xi32>
  %base_splat = tt.splat %base : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>>
  %ptrs = tt.addptr %base_splat, %addr : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>
  %val = tt.load %ptrs : tensor<32x16x!tt.ptr<f16>>

  %guard = arith.cmpi slt, %offs, %n_splat : tensor<16xi32>
  %guard_expd = tt.expand_dims %guard {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
  %guard_bc = tt.broadcast %guard_expd : tensor<1x16xi1> -> tensor<32x16xi1>
  %dst_splat = tt.splat %dst : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>>
  %out_ptrs = tt.addptr %dst_splat, %addr : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>
  tt.store %out_ptrs, %val, %guard_bc : tensor<32x16x!tt.ptr<f16>>
  tt.return
}

// -----
// Negative: the wrapped index is also the data a load falls back to, so the
// widened index would become observable in the loaded values.
// CHECK-LABEL: tt.func public @modulo_skip_result_used_as_load_fill
// CHECK: arith.remsi
tt.func public @modulo_skip_result_used_as_load_fill(%base: !tt.ptr<i32> {tt.divisibility = 16 : i32},
                                                    %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32},
                                                    %n: i32) {
  %c16 = arith.constant 16 : i32
  %pid = tt.get_program_id x : i32
  %blk = arith.muli %pid, %c16 : i32
  %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %offs = arith.addi %blk_splat, %range : tensor<16xi32>
  %n_splat = tt.splat %n : i32 -> tensor<16xi32>
  %rem = arith.remsi %offs, %n_splat : tensor<16xi32>

  %rem_expd = tt.expand_dims %rem {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %rem_bc = tt.broadcast %rem_expd : tensor<1x16xi32> -> tensor<32x16xi32>
  %row = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %row_expd = tt.expand_dims %row {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %row_stride = arith.constant dense<16> : tensor<32x1xi32>
  %row_scaled = arith.muli %row_expd, %row_stride : tensor<32x1xi32>
  %row_bc = tt.broadcast %row_scaled : tensor<32x1xi32> -> tensor<32x16xi32>
  %addr = arith.addi %row_bc, %rem_bc : tensor<32x16xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<32x16x!tt.ptr<i32>>
  %ptrs = tt.addptr %base_splat, %addr : tensor<32x16x!tt.ptr<i32>>, tensor<32x16xi32>

  %guard = arith.cmpi slt, %offs, %n_splat : tensor<16xi32>
  %guard_expd = tt.expand_dims %guard {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
  %guard_bc = tt.broadcast %guard_expd : tensor<1x16xi1> -> tensor<32x16xi1>
  // The wrapped index is the fill value here, not only an address.
  %val = tt.load %ptrs, %guard_bc, %rem_bc : tensor<32x16x!tt.ptr<i32>>
  %col_expd = tt.expand_dims %offs {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %col_bc = tt.broadcast %col_expd : tensor<1x16xi32> -> tensor<32x16xi32>
  %out_addr = arith.addi %row_bc, %col_bc : tensor<32x16xi32>
  %dst_splat = tt.splat %dst : !tt.ptr<i32> -> tensor<32x16x!tt.ptr<i32>>
  %out_ptrs = tt.addptr %dst_splat, %out_addr : tensor<32x16x!tt.ptr<i32>>, tensor<32x16xi32>
  tt.store %out_ptrs, %val, %guard_bc : tensor<32x16x!tt.ptr<i32>>
  tt.return
}

// -----
// Negative: the wrapped index leaves the address computation, so the widened
// value would be observable.
// CHECK-LABEL: tt.func public @modulo_skip_result_leaves_address_chain
// CHECK: arith.remsi
tt.func public @modulo_skip_result_leaves_address_chain(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                       %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                       %n: i32) {
  %c16 = arith.constant 16 : i32
  %pid = tt.get_program_id x : i32
  %blk = arith.muli %pid, %c16 : i32
  %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %offs = arith.addi %blk_splat, %range : tensor<16xi32>
  %n_splat = tt.splat %n : i32 -> tensor<16xi32>
  %rem = arith.remsi %offs, %n_splat : tensor<16xi32>

  %rem_expd = tt.expand_dims %rem {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %rem_bc = tt.broadcast %rem_expd : tensor<1x16xi32> -> tensor<32x16xi32>
  %row = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %row_expd = tt.expand_dims %row {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %row_stride = arith.constant dense<16> : tensor<32x1xi32>
  %row_scaled = arith.muli %row_expd, %row_stride : tensor<32x1xi32>
  %row_bc = tt.broadcast %row_scaled : tensor<32x1xi32> -> tensor<32x16xi32>
  %addr = arith.addi %row_bc, %rem_bc : tensor<32x16xi32>
  %base_splat = tt.splat %base : !tt.ptr<f32> -> tensor<32x16x!tt.ptr<f32>>
  %ptrs = tt.addptr %base_splat, %addr : tensor<32x16x!tt.ptr<f32>>, tensor<32x16xi32>
  %val = tt.load %ptrs : tensor<32x16x!tt.ptr<f32>>
  // The index becomes data here.
  %as_float = arith.sitofp %rem_bc : tensor<32x16xi32> to tensor<32x16xf32>
  %scaled = arith.mulf %val, %as_float : tensor<32x16xf32>

  %guard = arith.cmpi slt, %offs, %n_splat : tensor<16xi32>
  %guard_expd = tt.expand_dims %guard {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
  %guard_bc = tt.broadcast %guard_expd : tensor<1x16xi1> -> tensor<32x16xi1>
  %col_expd = tt.expand_dims %offs {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %col_bc = tt.broadcast %col_expd : tensor<1x16xi32> -> tensor<32x16xi32>
  %out_addr = arith.addi %row_bc, %col_bc : tensor<32x16xi32>
  %dst_splat = tt.splat %dst : !tt.ptr<f32> -> tensor<32x16x!tt.ptr<f32>>
  %out_ptrs = tt.addptr %dst_splat, %out_addr : tensor<32x16x!tt.ptr<f32>>, tensor<32x16xi32>
  tt.store %out_ptrs, %scaled, %guard_bc : tensor<32x16x!tt.ptr<f32>>
  tt.return
}

// -----
// Negative: the index is expanded on two different axes, so it addresses two
// different dimensions and no single mask orientation would be correct.
// CHECK-LABEL: tt.func public @modulo_skip_conflicting_expand_axes
// CHECK: arith.remsi
tt.func public @modulo_skip_conflicting_expand_axes(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                                   %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                                   %n: i32) {
  %c16 = arith.constant 16 : i32
  %pid = tt.get_program_id x : i32
  %blk = arith.muli %pid, %c16 : i32
  %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %offs = arith.addi %blk_splat, %range : tensor<16xi32>
  %n_splat = tt.splat %n : i32 -> tensor<16xi32>
  %rem = arith.remsi %offs, %n_splat : tensor<16xi32>

  %col_index = tt.expand_dims %rem {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %col_bc = tt.broadcast %col_index : tensor<1x16xi32> -> tensor<16x16xi32>
  %row_index = tt.expand_dims %rem {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %row_stride = arith.constant dense<16> : tensor<16x1xi32>
  %row_scaled = arith.muli %row_index, %row_stride : tensor<16x1xi32>
  %row_bc = tt.broadcast %row_scaled : tensor<16x1xi32> -> tensor<16x16xi32>
  %addr = arith.addi %row_bc, %col_bc : tensor<16x16xi32>
  %base_splat = tt.splat %base : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
  %ptrs = tt.addptr %base_splat, %addr : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
  %val = tt.load %ptrs : tensor<16x16x!tt.ptr<f16>>

  %guard = arith.cmpi slt, %offs, %n_splat : tensor<16xi32>
  %guard_expd = tt.expand_dims %guard {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
  %guard_bc = tt.broadcast %guard_expd : tensor<1x16xi1> -> tensor<16x16xi1>
  %out_col = tt.expand_dims %offs {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %out_bc = tt.broadcast %out_col : tensor<1x16xi32> -> tensor<16x16xi32>
  %dst_splat = tt.splat %dst : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
  %out_ptrs = tt.addptr %dst_splat, %out_bc : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
  tt.store %out_ptrs, %val, %guard_bc : tensor<16x16x!tt.ptr<f16>>
  tt.return
}

// -----
// Negative: a transpose moves the index to another dimension, so the axis its
// expand_dims recorded would no longer describe where it varies, and a mask
// built on that axis would zero the wrong lanes.
// CHECK-LABEL: tt.func public @modulo_skip_transposed_address
// CHECK: arith.remsi
tt.func public @modulo_skip_transposed_address(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                              %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                              %n: i32) {
  %c16 = arith.constant 16 : i32
  %pid = tt.get_program_id x : i32
  %blk = arith.muli %pid, %c16 : i32
  %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %offs = arith.addi %blk_splat, %range : tensor<16xi32>
  %n_splat = tt.splat %n : i32 -> tensor<16xi32>
  %rem = arith.remsi %offs, %n_splat : tensor<16xi32>

  %rem_expd = tt.expand_dims %rem {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %rem_bc = tt.broadcast %rem_expd : tensor<16x1xi32> -> tensor<16x8xi32>
  %transposed = tt.trans %rem_bc {order = array<i32: 1, 0>} : tensor<16x8xi32> -> tensor<8x16xi32>
  %base_splat = tt.splat %base : !tt.ptr<f16> -> tensor<8x16x!tt.ptr<f16>>
  %ptrs = tt.addptr %base_splat, %transposed : tensor<8x16x!tt.ptr<f16>>, tensor<8x16xi32>
  %val = tt.load %ptrs : tensor<8x16x!tt.ptr<f16>>

  %guard = arith.cmpi slt, %offs, %n_splat : tensor<16xi32>
  %guard_expd = tt.expand_dims %guard {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
  %guard_bc = tt.broadcast %guard_expd : tensor<1x16xi1> -> tensor<8x16xi1>
  %out_col = tt.expand_dims %offs {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %out_bc = tt.broadcast %out_col : tensor<1x16xi32> -> tensor<8x16xi32>
  %dst_splat = tt.splat %dst : !tt.ptr<f16> -> tensor<8x16x!tt.ptr<f16>>
  %out_ptrs = tt.addptr %dst_splat, %out_bc : tensor<8x16x!tt.ptr<f16>>, tensor<8x16xi32>
  tt.store %out_ptrs, %val, %guard_bc : tensor<8x16x!tt.ptr<f16>>
  tt.return
}

// -----
// Negative: a rank-one load has no second dimension for the boundary mask to
// leave alone, and it is already a contiguous transfer.
// CHECK-LABEL: tt.func public @modulo_skip_rank_one_load
// CHECK: arith.remsi
tt.func public @modulo_skip_rank_one_load(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                         %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                         %n: i32) {
  %c16 = arith.constant 16 : i32
  %pid = tt.get_program_id x : i32
  %blk = arith.muli %pid, %c16 : i32
  %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %offs = arith.addi %blk_splat, %range : tensor<16xi32>
  %n_splat = tt.splat %n : i32 -> tensor<16xi32>
  %rem = arith.remsi %offs, %n_splat : tensor<16xi32>

  %rem_expd = tt.expand_dims %rem {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %base_splat = tt.splat %base : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %ptrs = tt.addptr %base_splat, %rem : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  %val = tt.load %ptrs : tensor<16x!tt.ptr<f32>>

  %guard = arith.cmpi slt, %offs, %n_splat : tensor<16xi32>
  %dst_splat = tt.splat %dst : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %out_ptrs = tt.addptr %dst_splat, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  tt.store %out_ptrs, %val, %guard : tensor<16x!tt.ptr<f32>>
  tt.return
}

// -----
// Negative: the wrapped value is not a tile offset at all, so there is no
// linear access to recover.
// CHECK-LABEL: tt.func public @modulo_skip_without_make_range
// CHECK: arith.remsi
tt.func public @modulo_skip_without_make_range(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                              %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                              %n: i32) {
  %c16 = arith.constant 16 : i32
  %pid = tt.get_program_id x : i32
  %blk = arith.muli %pid, %c16 : i32
  %offs = tt.splat %blk : i32 -> tensor<16xi32>
  %n_splat = tt.splat %n : i32 -> tensor<16xi32>
  %rem = arith.remsi %offs, %n_splat : tensor<16xi32>

  %rem_expd = tt.expand_dims %rem {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %rem_bc = tt.broadcast %rem_expd : tensor<1x16xi32> -> tensor<32x16xi32>
  %row = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %row_expd = tt.expand_dims %row {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %row_stride = arith.constant dense<16> : tensor<32x1xi32>
  %row_scaled = arith.muli %row_expd, %row_stride : tensor<32x1xi32>
  %row_bc = tt.broadcast %row_scaled : tensor<32x1xi32> -> tensor<32x16xi32>
  %addr = arith.addi %row_bc, %rem_bc : tensor<32x16xi32>
  %base_splat = tt.splat %base : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>>
  %ptrs = tt.addptr %base_splat, %addr : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>
  %val = tt.load %ptrs : tensor<32x16x!tt.ptr<f16>>

  %guard = arith.cmpi slt, %offs, %n_splat : tensor<16xi32>
  %guard_expd = tt.expand_dims %guard {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
  %guard_bc = tt.broadcast %guard_expd : tensor<1x16xi1> -> tensor<32x16xi1>
  %dst_splat = tt.splat %dst : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>>
  %out_ptrs = tt.addptr %dst_splat, %row_bc : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>
  tt.store %out_ptrs, %val, %guard_bc : tensor<32x16x!tt.ptr<f16>>
  tt.return
}

// -----
// Negative: the divisor varies per lane, so one boundary comparison cannot
// describe the wrap.
// CHECK-LABEL: tt.func public @modulo_skip_non_splat_divisor
// CHECK: arith.remsi
tt.func public @modulo_skip_non_splat_divisor(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                             %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                             %n: i32) {
  %c16 = arith.constant 16 : i32
  %pid = tt.get_program_id x : i32
  %blk = arith.muli %pid, %c16 : i32
  %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %offs = arith.addi %blk_splat, %range : tensor<16xi32>
  %n_splat = tt.splat %n : i32 -> tensor<16xi32>
  %divisors = arith.addi %n_splat, %range : tensor<16xi32>
  %rem = arith.remsi %offs, %divisors : tensor<16xi32>

  %rem_expd = tt.expand_dims %rem {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %rem_bc = tt.broadcast %rem_expd : tensor<1x16xi32> -> tensor<32x16xi32>
  %row = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %row_expd = tt.expand_dims %row {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %row_stride = arith.constant dense<16> : tensor<32x1xi32>
  %row_scaled = arith.muli %row_expd, %row_stride : tensor<32x1xi32>
  %row_bc = tt.broadcast %row_scaled : tensor<32x1xi32> -> tensor<32x16xi32>
  %addr = arith.addi %row_bc, %rem_bc : tensor<32x16xi32>
  %base_splat = tt.splat %base : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>>
  %ptrs = tt.addptr %base_splat, %addr : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>
  %val = tt.load %ptrs : tensor<32x16x!tt.ptr<f16>>

  %guard = arith.cmpi slt, %offs, %n_splat : tensor<16xi32>
  %guard_expd = tt.expand_dims %guard {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
  %guard_bc = tt.broadcast %guard_expd : tensor<1x16xi1> -> tensor<32x16xi1>
  %col_expd = tt.expand_dims %offs {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %col_bc = tt.broadcast %col_expd : tensor<1x16xi32> -> tensor<32x16xi32>
  %out_addr = arith.addi %row_bc, %col_bc : tensor<32x16xi32>
  %dst_splat = tt.splat %dst : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>>
  %out_ptrs = tt.addptr %dst_splat, %out_addr : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>
  tt.store %out_ptrs, %val, %guard_bc : tensor<32x16x!tt.ptr<f16>>
  tt.return
}

// -----
// Negative: a two-dimensional wrap is not a tile offset in the form this rule
// reasons about.
// CHECK-LABEL: tt.func public @modulo_skip_two_dimensional_wrap
// CHECK: arith.remsi
tt.func public @modulo_skip_two_dimensional_wrap(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                                %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                                %n: i32) {
  %c16 = arith.constant 16 : i32
  %pid = tt.get_program_id x : i32
  %blk = arith.muli %pid, %c16 : i32
  %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %offs = arith.addi %blk_splat, %range : tensor<16xi32>
  %offs_expd = tt.expand_dims %offs {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %offs_bc = tt.broadcast %offs_expd : tensor<1x16xi32> -> tensor<32x16xi32>
  %n_splat_2d = tt.splat %n : i32 -> tensor<32x16xi32>
  %rem = arith.remsi %offs_bc, %n_splat_2d : tensor<32x16xi32>

  %base_splat = tt.splat %base : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>>
  %ptrs = tt.addptr %base_splat, %rem : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>
  %val = tt.load %ptrs : tensor<32x16x!tt.ptr<f16>>

  %n_splat = tt.splat %n : i32 -> tensor<16xi32>
  %guard = arith.cmpi slt, %offs, %n_splat : tensor<16xi32>
  %guard_expd = tt.expand_dims %guard {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
  %guard_bc = tt.broadcast %guard_expd : tensor<1x16xi1> -> tensor<32x16xi1>
  %dst_splat = tt.splat %dst : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>>
  %out_ptrs = tt.addptr %dst_splat, %offs_bc : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>
  tt.store %out_ptrs, %val, %guard_bc : tensor<32x16x!tt.ptr<f16>>
  tt.return
}

// -----
// Negative: the wrap is on 64-bit offsets, which is not the tile-offset form
// this rule reasons about.
// CHECK-LABEL: tt.func public @modulo_skip_wide_offsets
// CHECK: arith.remsi
tt.func public @modulo_skip_wide_offsets(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                        %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                        %n: i32) {
  %c16 = arith.constant 16 : i32
  %pid = tt.get_program_id x : i32
  %blk = arith.muli %pid, %c16 : i32
  %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %offs = arith.addi %blk_splat, %range : tensor<16xi32>
  %offs_wide = arith.extsi %offs : tensor<16xi32> to tensor<16xi64>
  %n_wide = arith.extsi %n : i32 to i64
  %n_splat_wide = tt.splat %n_wide : i64 -> tensor<16xi64>
  %rem = arith.remsi %offs_wide, %n_splat_wide : tensor<16xi64>

  %rem_expd = tt.expand_dims %rem {axis = 0 : i32} : tensor<16xi64> -> tensor<1x16xi64>
  %rem_bc = tt.broadcast %rem_expd : tensor<1x16xi64> -> tensor<32x16xi64>
  %base_splat = tt.splat %base : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>>
  %ptrs = tt.addptr %base_splat, %rem_bc : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi64>
  %val = tt.load %ptrs : tensor<32x16x!tt.ptr<f16>>

  %n_splat = tt.splat %n : i32 -> tensor<16xi32>
  %guard = arith.cmpi slt, %offs, %n_splat : tensor<16xi32>
  %guard_expd = tt.expand_dims %guard {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
  %guard_bc = tt.broadcast %guard_expd : tensor<1x16xi1> -> tensor<32x16xi1>
  %col_expd = tt.expand_dims %offs {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %col_bc = tt.broadcast %col_expd : tensor<1x16xi32> -> tensor<32x16xi32>
  %dst_splat = tt.splat %dst : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>>
  %out_ptrs = tt.addptr %dst_splat, %col_bc : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>
  tt.store %out_ptrs, %val, %guard_bc : tensor<32x16x!tt.ptr<f16>>
  tt.return
}

// -----
// Negative: a subtraction that is not the divsi/muli/subi expansion of a wrap
// must not be mistaken for one.
// CHECK-LABEL: tt.func public @modulo_skip_plain_subtraction
// CHECK: arith.subi
tt.func public @modulo_skip_plain_subtraction(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                             %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                             %n: i32) {
  %c16 = arith.constant 16 : i32
  %pid = tt.get_program_id x : i32
  %blk = arith.muli %pid, %c16 : i32
  %blk_splat = tt.splat %blk : i32 -> tensor<16xi32>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %offs = arith.addi %blk_splat, %range : tensor<16xi32>
  %n_splat = tt.splat %n : i32 -> tensor<16xi32>
  // A quotient of a different value makes this subtraction unrelated to a wrap.
  %quot = arith.divsi %range, %n_splat : tensor<16xi32>
  %prod = arith.muli %quot, %n_splat : tensor<16xi32>
  %shifted = arith.subi %offs, %prod : tensor<16xi32>

  %expd = tt.expand_dims %shifted {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %bc = tt.broadcast %expd : tensor<1x16xi32> -> tensor<32x16xi32>
  %base_splat = tt.splat %base : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>>
  %ptrs = tt.addptr %base_splat, %bc : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>
  %val = tt.load %ptrs : tensor<32x16x!tt.ptr<f16>>

  %guard = arith.cmpi slt, %offs, %n_splat : tensor<16xi32>
  %guard_expd = tt.expand_dims %guard {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
  %guard_bc = tt.broadcast %guard_expd : tensor<1x16xi1> -> tensor<32x16xi1>
  %col_expd = tt.expand_dims %offs {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %col_bc = tt.broadcast %col_expd : tensor<1x16xi32> -> tensor<32x16xi32>
  %dst_splat = tt.splat %dst : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>>
  %out_ptrs = tt.addptr %dst_splat, %col_bc : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>
  tt.store %out_ptrs, %val, %guard_bc : tensor<32x16x!tt.ptr<f16>>
  tt.return
}

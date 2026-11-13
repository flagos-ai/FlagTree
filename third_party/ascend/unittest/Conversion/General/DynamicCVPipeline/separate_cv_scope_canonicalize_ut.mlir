// RUN: triton-opt %s  --separate-cv-scope --preserve-control-attrs-canonicalize | FileCheck %s

//===----------------------------------------------------------------------===//
// SeparateCVScope + canonicalize focused tests
//
// These cases keep values alive by storing into sink memrefs passed as function
// arguments, instead of local alloc/store chains that canonicalize can delete as
// dead. Control-flow tests use dynamic conditions or dynamic loop bounds, and
// the owning-side results are consumed after the control-flow op.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @basic_scope_split_and_cleanup(
// CHECK: scope.scope : () -> () {
// CHECK: arith.addi
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: scope.scope : () -> () {
// CHECK: arith.muli
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @basic_scope_split_and_cleanup(%vec_in: i32, %cube_in: i32, %outv: memref<1xi32>, %outc: memref<1xi32>) {
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %c2v = arith.constant {ssbuffer.core_type = "VECTOR"} 2 : i32
    %c3v = arith.constant {ssbuffer.core_type = "VECTOR"} 3 : i32
    %c2c = arith.constant {ssbuffer.core_type = "CUBE"} 2 : i32
    %0 = arith.addi %vec_in, %c2v {ssbuffer.core_type = "VECTOR"} : i32
    %1 = arith.addi %0, %c3v {ssbuffer.core_type = "VECTOR"} : i32
    %2 = arith.muli %cube_in, %c2c {ssbuffer.core_type = "CUBE"} : i32
    memref.store %1, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xi32>
    memref.store %2, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi32>
    func.return
  }
}

// -----

// CHECK-LABEL: func.func @container_if_without_core_type(
// CHECK: scope.scope : () -> () {
// CHECK: scf.if
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: scope.scope : () -> () {
// CHECK: scf.if
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @container_if_without_core_type(%cond: i1, %vec_in: i32, %cube_in: i32, %outv: memref<1xi32>, %outc: memref<1xi32>) {
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %c2v = arith.constant {ssbuffer.core_type = "VECTOR"} 2 : i32
    %c2c = arith.constant {ssbuffer.core_type = "CUBE"} 2 : i32
    scf.if %cond {
      %0 = arith.addi %vec_in, %c2v {ssbuffer.core_type = "VECTOR"} : i32
      %1 = arith.muli %cube_in, %c2c {ssbuffer.core_type = "CUBE"} : i32
      memref.store %0, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xi32>
      memref.store %1, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi32>
    } else {
      %0 = arith.subi %vec_in, %c2v {ssbuffer.core_type = "VECTOR"} : i32
      %1 = arith.addi %cube_in, %c2c {ssbuffer.core_type = "CUBE"} : i32
      memref.store %0, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xi32>
      memref.store %1, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi32>
    }
    func.return
  }
}

// -----

// CHECK-LABEL: func.func @mixed_if_yield_neutralize_scalar(
// CHECK: scope.scope : () -> () {
// CHECK: scf.if
// CHECK: scf.yield
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: scope.scope : () -> () {
// CHECK: scf.if
// CHECK: scf.yield
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @mixed_if_yield_neutralize_scalar(%cond: i1, %vec_in: i32, %cube_in: i32, %outv: memref<1xi32>, %outc: memref<1xi32>) {
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %c1v = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : i32
    %c2c = arith.constant {ssbuffer.core_type = "CUBE"} 2 : i32
    %0:2 = scf.if %cond -> (i32, i32) {
      %1 = arith.addi %vec_in, %c1v {ssbuffer.core_type = "VECTOR"} : i32
      %2 = arith.addi %cube_in, %c2c {ssbuffer.core_type = "CUBE"} : i32
      scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %1, %2 : i32, i32
    } else {
      %1 = arith.subi %vec_in, %c1v {ssbuffer.core_type = "VECTOR"} : i32
      %2 = arith.subi %cube_in, %c2c {ssbuffer.core_type = "CUBE"} : i32
      scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %1, %2 : i32, i32
    } {ssbuffer.core_type = "VECTOR, CUBE"}
    %1 = arith.addi %0#0, %c1v {ssbuffer.core_type = "VECTOR"} : i32
    %2 = arith.addi %0#1, %c2c {ssbuffer.core_type = "CUBE"} : i32
    memref.store %1, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xi32>
    memref.store %2, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi32>
    func.return
  }
}

// -----

// CHECK-LABEL: func.func @mixed_for_iter_args_index_neutralize(
// CHECK: scope.scope : () -> () {
// CHECK: scf.for
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: scope.scope : () -> () {
// CHECK: scf.for
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @mixed_for_iter_args_index_neutralize(%lb: i32, %ub: i32, %vec_init: i32, %cube_init: index, %outv: memref<1xi32>, %outc: memref<1xindex>) {
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %stepv = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : i32
    %stepc = arith.constant {ssbuffer.core_type = "CUBE"} 2 : index
    %0:2 = scf.for %i = %lb to %ub step %stepv iter_args(%a = %vec_init, %b = %cube_init) -> (i32, index) : i32 {
      %1 = arith.addi %a, %stepv {ssbuffer.core_type = "VECTOR"} : i32
      %2 = arith.addi %b, %stepc {ssbuffer.core_type = "CUBE"} : index
      scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %1, %2 : i32, index
    } {ssbuffer.core_type = "VECTOR, CUBE"}
    %1 = arith.addi %0#0, %stepv {ssbuffer.core_type = "VECTOR"} : i32
    %2 = arith.addi %0#1, %stepc {ssbuffer.core_type = "CUBE"} : index
    memref.store %1, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xi32>
    memref.store %2, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xindex>
    func.return
  }
}

// -----

// CHECK-LABEL: func.func @for_forwarding_use_can_be_ignored(
// CHECK: scope.scope : () -> () {
// CHECK: scf.for
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: scope.scope : () -> () {
// CHECK: scf.for
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @for_forwarding_use_can_be_ignored(%lb: i32, %ub: i32, %vec_init: i32, %cube_init: i32, %outv: memref<1xi32>, %outc: memref<1xi32>) {
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %c1v = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : i32
    %c1c = arith.constant {ssbuffer.core_type = "CUBE"} 1 : i32
    %0:2 = scf.for %i = %lb to %ub step %c1v iter_args(%a = %vec_init, %b = %cube_init) -> (i32, i32) : i32 {
      %1 = arith.addi %a, %c1v {ssbuffer.core_type = "VECTOR"} : i32
      %2 = arith.addi %b, %c1c {ssbuffer.core_type = "CUBE"} : i32
      scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %1, %2 : i32, i32
    } {ssbuffer.core_type = "VECTOR, CUBE"}
    %1:2 = scf.for %j = %lb to %ub step %c1v iter_args(%x = %0#0, %y = %0#1) -> (i32, i32) : i32 {
      scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %x, %y : i32, i32
    } {ssbuffer.core_type = "VECTOR, CUBE"}
    %2 = arith.addi %1#0, %c1v {ssbuffer.core_type = "VECTOR"} : i32
    %3 = arith.addi %1#1, %c1c {ssbuffer.core_type = "CUBE"} : i32
    memref.store %2, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xi32>
    memref.store %3, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi32>
    func.return
  }
}

// -----

// CHECK-LABEL: func.func @for_cross_slot_yield_use_does_not_preserve_source(
// CHECK: scope.scope : () -> () {
// CHECK: scf.for
// CHECK: arith.addf
// CHECK: arith.addf
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: scope.scope : () -> () {
// CHECK: scf.for
// CHECK-NOT: arith.addf
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @for_cross_slot_yield_use_does_not_preserve_source(%lb: i32, %ub: i32, %vec_src: f32, %vec_dst: f32, %cube_init: i32, %outv: memref<1xf32>, %outc: memref<1xi32>) {
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %vf1 = arith.constant {ssbuffer.core_type = "VECTOR"} 1.000000e+00 : f32
    %c1v = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : i32
    %c1c = arith.constant {ssbuffer.core_type = "CUBE"} 1 : i32
    %0:3 = scf.for %i = %lb to %ub step %c1v iter_args(%a = %vec_src, %b = %vec_dst, %c = %cube_init) -> (f32, f32, i32) : i32 {
      %1 = arith.addf %a, %vf1 {ssbuffer.core_type = "VECTOR"} : f32
      %2 = arith.addf %1, %b {ssbuffer.core_type = "VECTOR"} : f32
      %3 = arith.addi %c, %c1c {ssbuffer.core_type = "CUBE"} : i32
      scf.yield {ssbuffer.core_type = "VECTOR, VECTOR, CUBE"} %1, %2, %3 : f32, f32, i32
    } {ssbuffer.core_type = "VECTOR, VECTOR, CUBE"}
    %1 = arith.addf %0#1, %vf1 {ssbuffer.core_type = "VECTOR"} : f32
    %2 = arith.addi %0#2, %c1c {ssbuffer.core_type = "CUBE"} : i32
    memref.store %1, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xf32>
    memref.store %2, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi32>
    func.return
  }
}

// -----

// CHECK-LABEL: func.func @while_dependency_preserved_for_cube(
// CHECK: scope.scope : () -> () {
// CHECK: scf.if
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: scope.scope : () -> () {
// CHECK: scf.if
// CHECK: scf.while
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @while_dependency_preserved_for_cube(%cond: i1, %vec_init: i64, %cube_init: i64, %limit: i64, %outv: memref<1xi64>, %outc: memref<1xi64>) {
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %c1v = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : i64
    %c2c = arith.constant {ssbuffer.core_type = "CUBE"} 2 : i64
    %0 = scf.if %cond -> (i64) {
      %1 = arith.addi %vec_init, %c1v {ssbuffer.core_type = "VECTOR"} : i64
      scf.yield {ssbuffer.core_type = "VECTOR"} %1 : i64
    } else {
      %1 = arith.subi %limit, %c1v {ssbuffer.core_type = "VECTOR"} : i64
      scf.yield {ssbuffer.core_type = "VECTOR"} %1 : i64
    } {ssbuffer.core_type = "VECTOR"}
    %1:3 = scf.while (%a = %vec_init, %b = %cube_init, %c = %0) : (i64, i64, i64) -> (i64, i64, i64) {
      %2 = arith.addi %a, %c {ssbuffer.core_type = "VECTOR"} : i64
      %3 = arith.cmpi slt, %2, %limit {ssbuffer.core_type = "VECTOR"} : i64
      scf.condition(%3) %2, %b, %c : i64, i64, i64
    } do {
    ^bb0(%a_iter: i64, %b_iter: i64, %c_iter: i64):
      %2 = arith.addi %b_iter, %c2c {ssbuffer.core_type = "CUBE"} : i64
      scf.yield {ssbuffer.core_type = "CUBE, CUBE, VECTOR"} %a_iter, %2, %c_iter : i64, i64, i64
    } attributes {ssbuffer.core_type = "CUBE, CUBE, VECTOR"}
    %2 = arith.addi %1#1, %c2c {ssbuffer.core_type = "CUBE"} : i64
    %3 = arith.addi %1#2, %c1v {ssbuffer.core_type = "VECTOR"} : i64
    memref.store %3, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xi64>
    memref.store %2, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi64>
    func.return
  }
}

// -----

// CHECK-LABEL: func.func @while_inner_if_terminator_use_blocks_dead_shell_erase(
// CHECK: scope.scope : () -> () {
// CHECK: scf.while
// CHECK: scf.if
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: scope.scope : () -> () {
// CHECK: scf.while
// CHECK: scf.if
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @while_inner_if_terminator_use_blocks_dead_shell_erase(%if_cond: i1, %vec_init: i32, %cube_init: i32, %limit: i32, %outv: memref<1xi32>, %outc: memref<1xi32>) {
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %c1v = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : i32
    %c2c = arith.constant {ssbuffer.core_type = "CUBE"} 2 : i32
    %0:2 = scf.while (%v = %vec_init, %c = %cube_init) : (i32, i32) -> (i32, i32) {
      %keep = arith.cmpi slt, %v, %limit {ssbuffer.core_type = "VECTOR"} : i32
      scf.condition(%keep) %v, %c : i32, i32
    } do {
    ^bb0(%v_iter: i32, %c_iter: i32):
      %next_v = scf.if %if_cond -> (i32) {
        %inc_v = arith.addi %v_iter, %c1v {ssbuffer.core_type = "VECTOR"} : i32
        scf.yield {ssbuffer.core_type = "VECTOR"} %inc_v : i32
      } else {
        scf.yield {ssbuffer.core_type = "VECTOR"} %v_iter : i32
      } {ssbuffer.core_type = "VECTOR"}
      %next_c = arith.addi %c_iter, %c2c {ssbuffer.core_type = "CUBE"} : i32
      scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %next_v, %next_c : i32, i32
    } attributes {ssbuffer.core_type = "VECTOR, CUBE"}
    %1 = arith.addi %0#0, %c1v {ssbuffer.core_type = "VECTOR"} : i32
    %2 = arith.addi %0#1, %c2c {ssbuffer.core_type = "CUBE"} : i32
    memref.store %1, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xi32>
    memref.store %2, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi32>
    func.return
  }
}

// -----

// CHECK-LABEL: func.func @while_forwarding_use_can_be_ignored(
// CHECK: scope.scope : () -> () {
// CHECK: scf.while
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: scope.scope : () -> () {
// CHECK: scf.while
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @while_forwarding_use_can_be_ignored(%vec_init: i64, %cube_init: i64, %limit: i64, %outv: memref<1xi64>, %outc: memref<1xi64>) {
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %c1v = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : i64
    %c2c = arith.constant {ssbuffer.core_type = "CUBE"} 2 : i64
    %0:2 = scf.while (%v = %vec_init, %c = %cube_init) : (i64, i64) -> (i64, i64) {
      %keep = arith.cmpi slt, %v, %limit {ssbuffer.core_type = "VECTOR"} : i64
      scf.condition(%keep) %v, %c : i64, i64
    } do {
    ^bb0(%v_iter: i64, %c_iter: i64):
      %nv = arith.addi %v_iter, %c1v {ssbuffer.core_type = "VECTOR"} : i64
      %nc = arith.addi %c_iter, %c2c {ssbuffer.core_type = "CUBE"} : i64
      scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %nv, %nc : i64, i64
    } attributes {ssbuffer.core_type = "VECTOR, CUBE"}
    %1:2 = scf.while (%x = %0#0, %y = %0#1) : (i64, i64) -> (i64, i64) {
      %keep = arith.cmpi slt, %x, %limit {ssbuffer.core_type = "VECTOR"} : i64
      scf.condition(%keep) %x, %y : i64, i64
    } do {
    ^bb0(%x_iter: i64, %y_iter: i64):
      scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %x_iter, %y_iter : i64, i64
    } attributes {ssbuffer.core_type = "VECTOR, CUBE"}
    %2 = arith.addi %1#0, %c1v {ssbuffer.core_type = "VECTOR"} : i64
    %3 = arith.addi %1#1, %c2c {ssbuffer.core_type = "CUBE"} : i64
    memref.store %2, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xi64>
    memref.store %3, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi64>
    func.return
  }
}

// -----

// CHECK-LABEL: func.func @for_same_slot_loop_carry_preserved(
// CHECK: scope.scope : () -> () {
// CHECK: scf.for
// CHECK: scf.if
// CHECK: scf.while
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: scope.scope : () -> () {
// CHECK: %[[FOR:.*]]:2 = scf.for
// CHECK: %[[IF:.*]]:2 = scf.if
// CHECK: %[[WHILE:.*]]:2 = scf.while
// CHECK: arith.extsi
// CHECK: scf.yield %[[WHILE]]#0, %[[WHILE]]#1 : i32, i64
// CHECK: scf.yield %[[IF]]#0, %[[IF]]#1 : i32, i64
// CHECK: memref.store %[[FOR]]#1
// CHECK: scope.return
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @for_same_slot_loop_carry_preserved(%lb: i32, %ub: i32, %cond: i1, %vec_init: i32, %cube_init: i64, %outv: memref<1xi32>, %outc: memref<1xi64>) {
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %c1v = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : i32
    %c2v = arith.constant {ssbuffer.core_type = "VECTOR"} 2 : i32
    %c1c = arith.constant {ssbuffer.core_type = "CUBE"} 1 : i64
    %0:2 = scf.for %i = %lb to %ub step %c1v iter_args(%state = %vec_init, %cube = %cube_init) -> (i32, i64) : i32 {
      %1:2 = scf.if %cond -> (i32, i64) {
        %2 = arith.addi %state, %c2v {ssbuffer.core_type = "VECTOR"} : i32
        %3:2 = scf.while (%iter = %state, %acc = %cube) : (i32, i64) -> (i32, i64) {
          %4 = arith.cmpi slt, %iter, %2 {ssbuffer.core_type = "VECTOR"} : i32
          scf.condition(%4) %iter, %acc : i32, i64
        } do {
        ^bb0(%iter_arg: i32, %acc_arg: i64):
          %4 = arith.addi %iter_arg, %c1v {ssbuffer.core_type = "VECTOR"} : i32
          %5 = arith.extsi %iter_arg {ssbuffer.core_type = "CUBE"} : i32 to i64
          %6 = arith.addi %acc_arg, %5 {ssbuffer.core_type = "CUBE"} : i64
          scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %4, %6 : i32, i64
        } attributes {ssbuffer.core_type = "VECTOR, CUBE"}
        scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %3#0, %3#1 : i32, i64
      } else {
        scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %state, %cube : i32, i64
      } {ssbuffer.core_type = "VECTOR, CUBE"}
      scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %1#0, %1#1 : i32, i64
    } {ssbuffer.core_type = "VECTOR, CUBE"}
    memref.store %0#0, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xi32>
    memref.store %0#1, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi64>
    func.return
  }
}

// -----

// CHECK-LABEL: func.func @memref_slot_neutralize_uses_alloc(
// CHECK: scope.scope : () -> () {
// CHECK: scf.if
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: scope.scope : () -> () {
// CHECK: memref.load
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @memref_slot_neutralize_uses_alloc(%cond: i1, %src: memref<4xi32>, %vec_seed: i32, %outv: memref<1xi32>, %outc: memref<1xi32>) {
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %cube_idx = arith.constant {ssbuffer.core_type = "CUBE"} 2 : index
    %c1v = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : i32
    %0:2 = scf.if %cond -> (i32, memref<4xi32>) {
      %1 = arith.addi %vec_seed, %c1v {ssbuffer.core_type = "VECTOR"} : i32
      scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %1, %src : i32, memref<4xi32>
    } else {
      %1 = arith.subi %vec_seed, %c1v {ssbuffer.core_type = "VECTOR"} : i32
      scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %1, %src : i32, memref<4xi32>
    } {ssbuffer.core_type = "VECTOR, CUBE"}
    %1 = arith.addi %0#0, %c1v {ssbuffer.core_type = "VECTOR"} : i32
    %2 = memref.load %0#1[%cube_idx] {ssbuffer.core_type = "CUBE"} : memref<4xi32>
    memref.store %1, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xi32>
    memref.store %2, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi32>
    func.return
  }
}

// -----

// CHECK-LABEL: func.func @static_tensor_slot_neutralize_uses_dense_zero(
// CHECK: scope.scope : () -> () {
// CHECK: arith.addf
// CHECK: bufferization.materialize_in_destination
// CHECK: scope.return
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: scope.scope : () -> () {
// CHECK: scf.if
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @static_tensor_slot_neutralize_uses_dense_zero(%cond: i1, %src: tensor<4xf32>, %cube_seed: i32, %outv: memref<4xf32>, %outc: memref<1xi32>) {
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %c1v = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : i32
    %c2c = arith.constant {ssbuffer.core_type = "CUBE"} 2 : i32
    %0:2 = scf.if %cond -> (tensor<4xf32>, i32) {
      %1 = arith.addi %cube_seed, %c2c {ssbuffer.core_type = "CUBE"} : i32
      scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %src, %1 : tensor<4xf32>, i32
    } else {
      %1 = arith.subi %cube_seed, %c1v {ssbuffer.core_type = "CUBE"} : i32
      scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %src, %1 : tensor<4xf32>, i32
    } {ssbuffer.core_type = "VECTOR, CUBE"}
    %1 = arith.addf %0#0, %src {ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
    %2 = arith.addi %0#1, %c2c {ssbuffer.core_type = "CUBE"} : i32
    bufferization.materialize_in_destination %1 in writable %outv {ssbuffer.core_type = "VECTOR"} : (tensor<4xf32>, memref<4xf32>) -> ()
    memref.store %2, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi32>
    func.return
  }
}

// -----

// CHECK-LABEL: func.func @cross_scope_live_user_keeps_producer(
// CHECK: scope.scope : () -> () {
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: scope.scope : () -> () {
// CHECK: arith.muli
// CHECK-NOT: arith.addi
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @cross_scope_live_user_keeps_producer(%cube_seed: i32, %vec_seed: i32, %outv: memref<1xi32>, %outc: memref<1xi32>) {
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %c2c = arith.constant {ssbuffer.core_type = "CUBE"} 2 : i32
    %0 = arith.muli %cube_seed, %c2c {ssbuffer.core_type = "CUBE"} : i32
    %1 = arith.addi %0, %vec_seed {ssbuffer.core_type = "VECTOR"} : i32
    memref.store %1, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xi32>
    memref.store %0, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi32>
    func.return
  }
}

// -----

// CHECK-LABEL: func.func @short_core_type_list_falls_back_to_first(
// CHECK: scope.scope : () -> () {
// CHECK-NOT: scf.if
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: scope.scope : () -> () {
// CHECK: scf.if
// CHECK: scf.yield
// CHECK: memref.store
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @short_core_type_list_falls_back_to_first(%cond: i1, %vec_seed: i32, %cube_seed0: i32, %cube_seed1: i32, %outv: memref<1xi32>, %outc0: memref<1xi32>, %outc1: memref<1xi32>) {
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc0 = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %idxc1 = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %c1v = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : i32
    %c2c = arith.constant {ssbuffer.core_type = "CUBE"} 2 : i32
    %0:2 = scf.if %cond -> (i32, i32) {
      %1 = arith.addi %cube_seed0, %c2c {ssbuffer.core_type = "CUBE"} : i32
      %2 = arith.subi %cube_seed1, %c2c {ssbuffer.core_type = "CUBE"} : i32
      scf.yield {ssbuffer.core_type = "CUBE"} %1, %2 : i32, i32
    } else {
      %1 = arith.subi %cube_seed0, %c2c {ssbuffer.core_type = "CUBE"} : i32
      %2 = arith.addi %cube_seed1, %c2c {ssbuffer.core_type = "CUBE"} : i32
      scf.yield {ssbuffer.core_type = "CUBE"} %1, %2 : i32, i32
    } {ssbuffer.core_type = "CUBE"}
    %1 = arith.addi %vec_seed, %c1v {ssbuffer.core_type = "VECTOR"} : i32
    %2 = arith.addi %0#0, %c2c {ssbuffer.core_type = "CUBE"} : i32
    %3 = arith.addi %0#1, %c2c {ssbuffer.core_type = "CUBE"} : i32
    memref.store %1, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xi32>
    memref.store %2, %outc0[%idxc0] {ssbuffer.core_type = "CUBE"} : memref<1xi32>
    memref.store %3, %outc1[%idxc1] {ssbuffer.core_type = "CUBE"} : memref<1xi32>
    func.return
  }
}

// -----

// CHECK-LABEL: func.func @vector_if_bound_with_cube_loop_side_effect(
// A VECTOR-only scf.if computes a loop upper bound. The for loop itself is
// VECTOR-typed (all results are VECTOR), but its body contains a CUBE
// side-effect (memref.store). Without the fix, the CUBE scope neutralizes
// the scf.if yields to 0 because findScopeRelevantUser only follows the for's
// SSA results (all VECTOR, no CUBE users), collapsing the bound to 0 and
// causing canonicalize to erase the CUBE store entirely.
// VECTOR scope: scf.if folds to arith.select (no side effects in branches),
// for loop accumulates with VECTOR result, memref.store to outv
// CHECK:      scope.scope : () -> () {
// CHECK-NEXT:   %[[UB_V:.*]] = arith.select
// CHECK-NEXT:   %[[RES_V:.*]] = scf.for {{.*}} to %[[UB_V]]
// CHECK-NEXT:     %{{.*}} = arith.addi
// CHECK-NEXT:     scf.yield
// CHECK-NEXT:   }
// CHECK-NEXT:   memref.store %[[RES_V]]
// CHECK-NEXT:   scope.return
// CHECK-NEXT: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CUBE scope: scf.if also folds to arith.select (preserved, NOT neutralized
// to 0); for loop runs the correct number of times with CUBE memref.store
// CHECK-NEXT: scope.scope : () -> () {
// CHECK-NEXT:   %[[UB_C:.*]] = arith.select
// CHECK-NEXT:   scf.for {{.*}} to %[[UB_C]]
// CHECK-NEXT:     memref.store
// CHECK-NEXT:   }
// CHECK-NEXT:   scope.return
// CHECK-NEXT: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @vector_if_bound_with_cube_loop_side_effect(
      %cond: i1, %ub_a: i32, %ub_b: i32, %vinit: i32, %cube_val: i32,
      %outv: memref<1xi32>, %outc: memref<1xi32>) {
    %c0 = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : i32
    %c1v = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : i32
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %ub = scf.if %cond -> (i32) {
      scf.yield {ssbuffer.core_type = "VECTOR"} %ub_a : i32
    } else {
      scf.yield {ssbuffer.core_type = "VECTOR"} %ub_b : i32
    } {ssbuffer.core_type = "VECTOR"}
    %res = scf.for %i = %c0 to %ub step %c1v
            iter_args(%acc = %vinit) -> (i32) : i32 {
      %nv = arith.addi %acc, %c1v {ssbuffer.core_type = "VECTOR"} : i32
      memref.store %cube_val, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi32>
      scf.yield {ssbuffer.core_type = "VECTOR"} %nv : i32
    } {ssbuffer.core_type = "VECTOR"}
    memref.store %res, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xi32>
    func.return
  }
}

// -----

// CHECK-LABEL: func.func @vector_if_condition_gates_cube_if_branch(
// A VECTOR-only inner scf.if produces an i1 used as the condition of an outer
// resultful scf.if. The inner if's only SSA result is the i1 (VECTOR), so it
// has no CUBE user; without the gating fix the CUBE scope neutralizes the inner
// yields, the condition folds to a constant, and canonicalize drops the gated
// CUBE memref.store. The fix keeps the inner if live in the CUBE scope because
// the outer if body still holds CUBE content.
// VECTOR scope: the gated branch holds no VECTOR side effect, so canonicalize
// drops the whole if and only the trailing VECTOR store remains.
// CHECK:      scope.scope : () -> () {
// CHECK-NEXT:   memref.store
// CHECK-NEXT:   scope.return
// CHECK-NEXT: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CUBE scope: the inner if folds to arith.select (no side effects in its
// branches) and feeds the condition of the preserved outer scf.if guarding
// the CUBE store; the condition is NOT neutralized to a constant.
// CHECK-NEXT: scope.scope : () -> () {
// CHECK-NEXT:   %[[CONDC:.*]] = arith.select
// CHECK-NEXT:   scf.if %[[CONDC]] {
// CHECK-NEXT:     memref.store
// CHECK-NEXT:   }
// CHECK-NEXT:   scope.return
// CHECK-NEXT: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @vector_if_condition_gates_cube_if_branch(
      %cond: i1, %va: i1, %vb: i1, %cube_val: i32,
      %outv: memref<1xi32>, %outc: memref<1xi32>) {
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %vseed = arith.constant {ssbuffer.core_type = "VECTOR"} 7 : i32
    %gate = scf.if %cond -> (i1) {
      scf.yield {ssbuffer.core_type = "VECTOR"} %va : i1
    } else {
      scf.yield {ssbuffer.core_type = "VECTOR"} %vb : i1
    } {ssbuffer.core_type = "VECTOR"}
    %r = scf.if %gate -> (i32) {
      memref.store %cube_val, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi32>
      scf.yield {ssbuffer.core_type = "VECTOR"} %vseed : i32
    } else {
      scf.yield {ssbuffer.core_type = "VECTOR"} %vseed : i32
    } {ssbuffer.core_type = "VECTOR"}
    memref.store %r, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xi32>
    func.return
  }
}

// -----

// CHECK-LABEL: func.func @vector_if_condition_gates_cube_while_body(
// A VECTOR-only inner scf.if produces an i1 used as the scf.while condition.
// The while results are all VECTOR (no CUBE user of the gate), but the while
// body holds a CUBE memref.store. Without the gating fix the CUBE scope
// neutralizes the inner if (condition -> false), the while never runs, and
// canonicalize erases the CUBE store. The fix keeps the inner if live because
// the while body still holds CUBE content.
// CHECK:      scope.scope : () -> () {
// CHECK-NEXT:   scf.while
// CHECK-NEXT:     arith.cmpi
// CHECK-NEXT:     arith.select
// CHECK-NEXT:     scf.condition
// CHECK-NEXT:   } do {
// CHECK-NEXT:   ^bb0
// CHECK-NEXT:     arith.addi
// CHECK-NEXT:     scf.yield
// CHECK-NEXT:   }
// CHECK-NEXT:   memref.store
// CHECK-NEXT:   scope.return
// CHECK-NEXT: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CUBE scope: the inner if folds to arith.select feeding scf.condition; the
// while loop and its CUBE store body are preserved (condition not constant).
// CHECK-NEXT: scope.scope : () -> () {
// CHECK-NEXT:   scf.while
// CHECK-NEXT:     arith.cmpi
// CHECK-NEXT:     %[[SELC:.*]] = arith.select
// CHECK-NEXT:     scf.condition(%[[SELC]])
// CHECK-NEXT:   } do {
// CHECK-NEXT:   ^bb0
// CHECK-NEXT:     memref.store
// CHECK-NEXT:     arith.addi
// CHECK-NEXT:     scf.yield
// CHECK-NEXT:   }
// CHECK-NEXT:   scope.return
// CHECK-NEXT: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @vector_if_condition_gates_cube_while_body(
      %va: i1, %vb: i1, %vec_init: i32, %cube_val: i32,
      %outv: memref<1xi32>, %outc: memref<1xi32>) {
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %c1v = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : i32
    %limit = arith.constant {ssbuffer.core_type = "VECTOR"} 4 : i32
    %0 = scf.while (%v = %vec_init) : (i32) -> (i32) {
      %lt = arith.cmpi slt, %v, %limit {ssbuffer.core_type = "VECTOR"} : i32
      %gate = scf.if %lt -> (i1) {
        scf.yield {ssbuffer.core_type = "VECTOR"} %va : i1
      } else {
        scf.yield {ssbuffer.core_type = "VECTOR"} %vb : i1
      } {ssbuffer.core_type = "VECTOR"}
      scf.condition(%gate) %v : i32
    } do {
    ^bb0(%v_iter: i32):
      memref.store %cube_val, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi32>
      %nv = arith.addi %v_iter, %c1v {ssbuffer.core_type = "VECTOR"} : i32
      scf.yield {ssbuffer.core_type = "VECTOR"} %nv : i32
    } attributes {ssbuffer.core_type = "VECTOR"}
    memref.store %0, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xi32>
    func.return
  }
}

// -----

// Regression: isProducedByForeignScope gate must only apply to loop ops
// (scf.for/scf.while), never to scf.if. The isLoopOp guard prevents the
// gate from firing on scf.if. A VECTOR-only scf.if feeding a mixed
// CUBE/VECTOR scf.while must stay live in the CUBE scope.
// CHECK-LABEL: func.func @if_into_while_preserved_in_cube(
// VECTOR scope: if preserved with real branch ops, then mixed while + store.
// CHECK:      scope.scope : () -> () {
// CHECK-NEXT:   %[[IF:.*]] = scf.if
// CHECK-NEXT:     arith.addi
// CHECK-NEXT:     scf.yield
// CHECK-NEXT:   } else {
// CHECK-NEXT:     arith.subi
// CHECK-NEXT:     scf.yield
// CHECK-NEXT:   }
// CHECK-NEXT:   %[[WHILE_SMALL:.*]] = scf.while
// CHECK-NEXT:     arith.addi
// CHECK-NEXT:     arith.cmpi
// CHECK-NEXT:     scf.condition
// CHECK-NEXT:   } do {
// CHECK-NEXT:   ^bb0({{.*}}):
// CHECK-NEXT:     arith.addi
// CHECK-NEXT:     scf.yield
// CHECK-NEXT:   }
// CHECK-NEXT:   memref.store
// CHECK-NEXT:   scope.return
// CHECK-NEXT: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CUBE scope: if preserved, mixed while preserved with CUBE arith.addi in body.
// CHECK-NEXT: scope.scope : () -> () {
// CHECK-NEXT:   %[[IFC:.*]] = scf.if
// CHECK-NEXT:     arith.addi
// CHECK-NEXT:     scf.yield
// CHECK-NEXT:   } else {
// CHECK-NEXT:     arith.subi
// CHECK-NEXT:     scf.yield
// CHECK-NEXT:   }
// CHECK-NEXT:   %[[WHILEC:.*]]:{{[0-9]+}} = scf.while
// CHECK-NEXT:     arith.addi
// CHECK-NEXT:     arith.cmpi
// CHECK-NEXT:     scf.condition
// CHECK-NEXT:   } do {
// CHECK-NEXT:   ^bb0({{.*}}):
// CHECK-NEXT:     arith.addi
// CHECK-NEXT:     arith.addi
// CHECK-NEXT:     scf.yield
// CHECK-NEXT:   }
// CHECK-NEXT:   arith.addi
// CHECK-NEXT:   memref.store
// CHECK-NEXT:   scope.return
// CHECK-NEXT: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @if_into_while_preserved_in_cube(%cond: i1, %vinit: i64, %cinit: i64, %limit: i64, %outv: memref<1xi64>, %outc: memref<1xi64>) {
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %c1v = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : i64
    %c2c = arith.constant {ssbuffer.core_type = "CUBE"} 2 : i64
    %c1c = arith.constant {ssbuffer.core_type = "CUBE"} 1 : i64
    %0 = scf.if %cond -> (i64) {
      %1 = arith.addi %vinit, %c1v {ssbuffer.core_type = "VECTOR"} : i64
      scf.yield {ssbuffer.core_type = "VECTOR"} %1 : i64
    } else {
      %1 = arith.subi %limit, %c1v {ssbuffer.core_type = "VECTOR"} : i64
      scf.yield {ssbuffer.core_type = "VECTOR"} %1 : i64
    } {ssbuffer.core_type = "VECTOR"}
    %1:2 = scf.while (%a = %0, %b = %cinit) : (i64, i64) -> (i64, i64) {
      %2 = arith.addi %a, %c1v {ssbuffer.core_type = "VECTOR"} : i64
      %3 = arith.cmpi slt, %2, %limit {ssbuffer.core_type = "VECTOR"} : i64
      scf.condition(%3) %2, %b : i64, i64
    } do {
    ^bb0(%a_iter: i64, %b_iter: i64):
      %2 = arith.addi %a_iter, %c1v {ssbuffer.core_type = "VECTOR"} : i64
      %3 = arith.addi %b_iter, %c2c {ssbuffer.core_type = "CUBE"} : i64
      scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %2, %3 : i64, i64
    } attributes {ssbuffer.core_type = "VECTOR, CUBE"}
    %2 = arith.addi %1#1, %c1c {ssbuffer.core_type = "CUBE"} : i64
    memref.store %1#0, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xi64>
    memref.store %2, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi64>
    func.return
  }
}

// -----

// Regression: in a 2-layer nested scf.for where the inner iter_arg flows
// directly into a CUBE accumulator (arith.mulf), needsLoopCarryPreserve on the
// inner for cannot find VECTOR users because same-slot yield blocks SSA
// propagation. The producer-scope gate (guarded by isLoopOp on scf.for) detects
// the foreign CUBE producer and skips findLiveUser so the slot is neutralized
// in VECTOR scope, erasing the CUBE op from the VECTOR clone while keeping it
// in the CUBE clone. An independent CUBE consumer of the outer loop result
// ensures the CUBE scope is not optimized away.
// CHECK-LABEL: func.func @nested_for_cube_acc_not_trapped_in_vector(
// VECTOR scope: outer+inner for survive, arith.addf{VECTOR} present
//               but arith.mulf{CUBE} absent, scalar CUBE consumer absent.
// CHECK:      scope.scope : () -> () {
// CHECK-NEXT:   [[OUTER:%.+]]:2 = scf.for
// CHECK:        [[INNER:%.+]]:2 = scf.for
// CHECK:          arith.addf
// CHECK-NOT:      arith.mulf
// CHECK:          scf.yield
// CHECK:        }
// CHECK:        scf.yield
// CHECK:        arith.addf
// CHECK:        tensor.extract
// CHECK:        memref.store
// CHECK-NEXT:   scope.return
// CHECK-NEXT: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CUBE scope: outer+inner for survive, arith.mulf{CUBE} present, and
//             the independent CUBE arith.addf consumer is present.
// CHECK-NEXT: scope.scope : () -> () {
// CHECK-NEXT:   [[OUTERC:%.+]]:2 = scf.for
// CHECK:        [[INNERC:%.+]]:2 = scf.for
// CHECK:          arith.mulf
// CHECK:          scf.yield
// CHECK:        }
// CHECK:        scf.yield
// CHECK:        arith.addf
// CHECK:        tensor.extract
// CHECK:        memref.store
// CHECK-NEXT:   scope.return
// CHECK-NEXT: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @nested_for_cube_acc_not_trapped_in_vector(
      %lb: index, %ub: index, %step: index,
      %acc_init: tensor<4xf32>, %vec_init: tensor<4xf32>,
      %ck: tensor<4xf32>, %vk: tensor<4xf32>, %cc_init: tensor<4xf32>,
      %outv: memref<1xf32>, %outc: memref<1xf32>) {
    %idx = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %1:2 = scf.for %a = %lb to %ub step %step
        iter_args(%oa = %acc_init, %ov = %vec_init) -> (tensor<4xf32>, tensor<4xf32>) : index {
      %2:2 = scf.for %b = %lb to %ub step %step
          iter_args(%ia = %oa, %iv = %ov) -> (tensor<4xf32>, tensor<4xf32>) : index {
        %3 = arith.mulf %ia, %ck {ssbuffer.core_type = "CUBE"} : tensor<4xf32>
        %4 = arith.addf %iv, %vk {ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
        scf.yield {ssbuffer.core_type = "CUBE, VECTOR"} %3, %4 : tensor<4xf32>, tensor<4xf32>
      } {ssbuffer.core_type = "CUBE, VECTOR"}
      scf.yield {ssbuffer.core_type = "CUBE, VECTOR"} %2#0, %2#1 : tensor<4xf32>, tensor<4xf32>
    } {ssbuffer.core_type = "CUBE, VECTOR"}
    %sv = arith.addf %1#0, %vk {ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
    %ev = tensor.extract %sv[%idx] {ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
    memref.store %ev, %outv[%idx] {ssbuffer.core_type = "VECTOR"} : memref<1xf32>
    %sc = arith.addf %1#0, %cc_init {ssbuffer.core_type = "CUBE"} : tensor<4xf32>
    %ec = tensor.extract %sc[%idxc] {ssbuffer.core_type = "CUBE"} : tensor<4xf32>
    memref.store %ec, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xf32>
    func.return
  }
}

// -----

// A VECTOR for can initialize a VECTOR while whose scalar results are consumed
// directly by CUBE operations. The CUBE clone must retain both complete state
// transitions; keeping either loop shell with neutral yields changes the live
// while results.
// CHECK-LABEL: func.func @vector_while_results_drive_cube(
// CHECK:      scope.scope : () -> () {
// CHECK:        %[[INIT:.*]]:2 = scf.for
// CHECK:          arith.index_cast
// CHECK-NEXT:     arith.addi
// CHECK-NEXT:     arith.addi
// CHECK-NEXT:     scf.yield
// CHECK:        %[[WHILE:.*]]:2 = scf.while
// CHECK:          arith.cmpi
// CHECK-NEXT:     scf.condition
// CHECK:        } do {
// CHECK:          arith.addi
// CHECK-NEXT:     arith.addi
// CHECK-NEXT:     scf.yield
// CHECK:        arith.muli %[[WHILE]]#0, %[[WHILE]]#1
// CHECK-NEXT:   memref.store
// CHECK-NEXT:   scope.return
// CHECK-NEXT: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @vector_while_results_drive_cube(
      %index_init: i64, %offset_init: i64, %limit: i64,
      %out: memref<1xi64>) {
    %idx = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %c0 = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %c2 = arith.constant {ssbuffer.core_type = "VECTOR"} 2 : index
    %c1 = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : index
    %step = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : i64
    %0:2 = scf.for %i = %c0 to %c2 step %c1
        iter_args(%index = %index_init, %offset = %offset_init)
        -> (i64, i64) {
      %i64 = arith.index_cast %i {ssbuffer.core_type = "VECTOR"}
          : index to i64
      %next_index = arith.addi %index, %i64
          {ssbuffer.core_type = "VECTOR"} : i64
      %next_offset = arith.addi %offset, %next_index
          {ssbuffer.core_type = "VECTOR"} : i64
      scf.yield {ssbuffer.core_type = "VECTOR, VECTOR"}
          %next_index, %next_offset : i64, i64
    } {ssbuffer.core_type = "VECTOR, VECTOR"}
    %1:2 = scf.while (%index = %0#0, %offset = %0#1)
        : (i64, i64) -> (i64, i64) {
      %continue = arith.cmpi slt, %index, %limit
          {ssbuffer.core_type = "VECTOR"} : i64
      scf.condition(%continue) %index, %offset : i64, i64
    } do {
    ^bb0(%index: i64, %offset: i64):
      %next_index = arith.addi %index, %step
          {ssbuffer.core_type = "VECTOR"} : i64
      %next_offset = arith.addi %offset, %next_index
          {ssbuffer.core_type = "VECTOR"} : i64
      scf.yield {ssbuffer.core_type = "VECTOR, VECTOR"}
          %next_index, %next_offset : i64, i64
    } attributes {ssbuffer.core_type = "VECTOR, VECTOR"}
    %product = arith.muli %1#0, %1#1 {ssbuffer.core_type = "CUBE"} : i64
    memref.store %product, %out[%idx] {ssbuffer.core_type = "CUBE"}
        : memref<1xi64>
    func.return
  }
}

// RUN: triton-opt %s --triton-to-unstructure -verify-each | FileCheck %s

// A pointer selected by an scf.if is an opaque complete address. When that
// value is yielded to an enclosing scf.while, every edge of the while boundary
// must use the complete-address representation instead of mixing i64 region
// arguments with a pointer backedge.
tt.func public @scalar_pointer_while_if_backedge(
    %initial: !tt.ptr<i32>, %alternate: !tt.ptr<i32>) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %result:2 = scf.while (%i = %c0, %ptr = %initial)
      : (i32, !tt.ptr<i32>) -> (i32, !tt.ptr<i32>) {
    %continue = arith.cmpi slt, %i, %c1 : i32
    scf.condition(%continue) %i, %ptr : i32, !tt.ptr<i32>
  } do {
  ^bb0(%i: i32, %ptr: !tt.ptr<i32>):
    %next_i = arith.addi %i, %c1 : i32
    %take_alternate = arith.cmpi eq, %i, %c0 : i32
    %next:2 = scf.if %take_alternate -> (!tt.ptr<i32>, i32) {
      scf.yield %alternate, %next_i : !tt.ptr<i32>, i32
    } else {
      scf.yield %ptr, %next_i : !tt.ptr<i32>, i32
    }
    scf.yield %next#1, %next#0 : i32, !tt.ptr<i32>
  }
  %loaded = tt.load %result#1 : !tt.ptr<i32>
  tt.return %loaded : i32
}

// CHECK-LABEL: tt.func public @scalar_pointer_while_if_backedge
// CHECK:       scf.while
// CHECK-SAME:  (i32, i64) -> (i32, i64)
// CHECK:       scf.condition
// CHECK-SAME:  i32, i64
// CHECK:       arith.select {{.*}} : !tt.ptr<i32>
// CHECK:       tt.ptr_to_int
// CHECK:       scf.yield
// CHECK-SAME:  i32, i64

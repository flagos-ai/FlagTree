// RUN: triton-opt %s --tle-dslregion-inline | FileCheck %s

module {
  llvm.func @_sink(!llvm.ptr)

  tt.func @k(%arg0: !tt.ptr<i32>) {
    %0 = "tle.dsl_region"(%arg0) ({
    ^bb0(%in: !tt.ptr<i32>):
      %p = "tle.extract_ptr"(%in) : (!tt.ptr<i32>) -> !llvm.ptr
      "tle.yield"(%p) : (!llvm.ptr) -> ()
    }) : (!tt.ptr<i32>) -> (!llvm.ptr)
    llvm.call @_sink(%0) : (!llvm.ptr) -> ()
    tt.return
  }
}

// CHECK-LABEL: tt.func @k(
// CHECK-NOT: tle.dsl_region
// CHECK: %[[P:.*]] = tle.extract_ptr
// CHECK: llvm.call @_sink(%[[P]]) : (!llvm.ptr) -> ()

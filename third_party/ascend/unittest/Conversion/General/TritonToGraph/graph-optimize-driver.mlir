// RUN: triton-opt --split-input-file --verify-each %s -graph-optimize='rule-mask=0 max-rewrites-per-function=1' -o - | FileCheck %s
// RUN: triton-opt --split-input-file --verify-each %s -graph-optimize -graph-optimize -o - | FileCheck %s

// CHECK-LABEL: tt.func @graph_optimize_first()
tt.func @graph_optimize_first() {
  // CHECK-NEXT: tt.return
  tt.return
  // CHECK-NEXT: }
}

// -----

// CHECK-LABEL: tt.func @graph_optimize_second(
tt.func @graph_optimize_second(%value: i32) -> i32 {
  // CHECK-NEXT: tt.return %{{.*}} : i32
  tt.return %value : i32
  // CHECK-NEXT: }
}

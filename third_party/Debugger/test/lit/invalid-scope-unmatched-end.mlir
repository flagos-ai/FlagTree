// RUN: not triton-opt %s --flagtree-resolve-debug-scope 2>&1 | FileCheck %s

module {
  func.func @unmatched_end() {
    "flagtree_debug.collect_end"() : () -> ()
    func.return
  }
}

// CHECK: debug collect_end without matching collect_begin

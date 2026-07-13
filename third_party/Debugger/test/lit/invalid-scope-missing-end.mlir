// RUN: not triton-opt %s --flagtree-resolve-debug-scope 2>&1 | FileCheck %s

module {
  func.func @missing_end() {
    "flagtree_debug.collect_begin"() {level = 1 : i32} : () -> ()
    func.return
  }
}

// CHECK: debug collect_begin without matching collect_end

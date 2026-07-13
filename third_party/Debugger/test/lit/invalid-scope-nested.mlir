// RUN: not triton-opt %s --flagtree-resolve-debug-scope 2>&1 | FileCheck %s

module {
  func.func @nested() {
    "flagtree_debug.collect_begin"() {level = 1 : i32} : () -> ()
    "flagtree_debug.collect_begin"() {level = 1 : i32} : () -> ()
    "flagtree_debug.collect_end"() : () -> ()
    "flagtree_debug.collect_end"() : () -> ()
    func.return
  }
}

// CHECK: illegal nested debug collect region

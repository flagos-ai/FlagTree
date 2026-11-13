// RUN: triton-opt --analyze-name %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @normal_kernel(
  func.func @normal_kernel() {
    // CHECK: return
    return
  }
}

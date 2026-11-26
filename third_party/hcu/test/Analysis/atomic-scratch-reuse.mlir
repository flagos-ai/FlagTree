// Exercise the HCU production filter and AsyncWait preprocessing on the shared cases.
// RUN: triton-opt %S/../../../../test/Analysis/atomic-scratch-reuse.mlir --test-tritonhcugpu-membar -verify-each | FileCheck %S/../../../../test/Analysis/atomic-scratch-reuse.mlir

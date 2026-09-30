// RUN: triton-opt %s -allow-unregistered-dialect -tritongpu-optimize-partition-warps | FileCheck %s

module attributes {ttg.target = "cuda:100", "ttg.num-warps" = 8 : i32} {


// Reuse partitions retain the physical warps that own their captured values,
// even when register pressure would otherwise allow fewer warps.
// CHECK-LABEL: @reuse_default_warps
tt.func @reuse_default_warps(%arg0: i32) {
  // CHECK: ttg.warp_specialize({{.*}}) attributes {reuseDefaultWarps = true}
  ttg.warp_specialize(%arg0) attributes {reuseDefaultWarps = true}
  default {
    ttg.warp_yield
  }
  // CHECK: partition0({{.*}}) num_warps(4)
  partition0(%arg1: i32) num_warps(4) {
    %0 = arith.addi %arg1, %arg1 : i32
    ttg.warp_return
  }
  // CHECK: partition1({{.*}}) num_warps(4)
  partition1(%arg1: i32) num_warps(4) {
    %0 = arith.subi %arg1, %arg1 : i32
    ttg.warp_return
  } : (i32) -> ()
  tt.return
}

}

// RUN: triton-opt %s -tritongpu-remove-layout-conversions | FileCheck %s
// RUN: triton-opt %s -tritongpu-remove-layout-conversions="enable-rlc-enhance=true enable-backward-propagation=false enable-store-layout-rematerialization=false" | FileCheck %s
// RUN: triton-opt %s -tritongpu-remove-layout-conversions="enable-rlc-enhance=true" | FileCheck %s

// Phase 2 producer closure must check the scan's target encoding, not merely
// include all its inputs/results in the proposal. The elementwise wrapper is
// intentional: a convert directly after scan is rejected before closure
// collection. A make_range input can be collected without a function argument
// stopping the closure. Returning the converted value fixes the target layout.

// CHECK-DAG: #[[$SRC:blocked[0-9]*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK-DAG: #[[$DST:blocked[0-9]*]] = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK-DAG: #[[$PARENT:blocked[0-9]*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
// CHECK-DAG: #[[$MMA:mma[0-9]*]] = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#slice = #ttg.slice<{dim = 0, parent = #parent}>
#row_slice = #ttg.slice<{dim = 1, parent = #parent}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#mma_slice = #ttg.slice<{dim = 0, parent = #mma}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:80"} {
  // A slice of a blocked encoding is not itself a blocked encoding.
  // CHECK-LABEL: @scan_reject_blocked_slice
  // CHECK: tt.make_range {{.*}} : tensor<128xi32, #[[$SRC]]>
  // CHECK: "tt.scan"
  // CHECK: : (tensor<128xi32, #[[$SRC]]>) -> tensor<128xi32, #[[$SRC]]>
  // CHECK: %[[WRAPPED:.+]] = arith.addi {{.*}} : tensor<128xi32, #[[$SRC]]>
  // CHECK: %[[CONVERTED:.+]] = ttg.convert_layout %[[WRAPPED]] : tensor<128xi32, #[[$SRC]]> -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #[[$PARENT]]}>>
  // CHECK: tt.return %[[CONVERTED]]
  tt.func public @scan_reject_blocked_slice() -> tensor<128xi32, #slice> {
    %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #blocked>
    %scan = "tt.scan"(%range) ({
    ^bb0(%a: i32, %b: i32):
      %sum = arith.addi %a, %b : i32
      tt.scan.return %sum : i32
    }) {axis = 0 : i32, reverse = false} : (tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    %wrapped = arith.addi %scan, %scan : tensor<128xi32, #blocked>
    %converted = ttg.convert_layout %wrapped : tensor<128xi32, #blocked> -> tensor<128xi32, #slice>
    tt.return %converted : tensor<128xi32, #slice>
  }

  // Producer-side solving admits MMA slices, but scan lowering does not.
  // CHECK-LABEL: @scan_reject_mma_slice
  // CHECK: tt.make_range {{.*}} : tensor<128xi32, #[[$SRC]]>
  // CHECK: "tt.scan"
  // CHECK: : (tensor<128xi32, #[[$SRC]]>) -> tensor<128xi32, #[[$SRC]]>
  // CHECK: %[[WRAPPED:.+]] = arith.addi {{.*}} : tensor<128xi32, #[[$SRC]]>
  // CHECK: %[[CONVERTED:.+]] = ttg.convert_layout %[[WRAPPED]] : tensor<128xi32, #[[$SRC]]> -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #[[$MMA]]}>>
  // CHECK: tt.return %[[CONVERTED]]
  tt.func public @scan_reject_mma_slice() -> tensor<128xi32, #mma_slice> {
    %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #blocked>
    %scan = "tt.scan"(%range) ({
    ^bb0(%a: i32, %b: i32):
      %sum = arith.addi %a, %b : i32
      tt.scan.return %sum : i32
    }) {axis = 0 : i32, reverse = false} : (tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    %wrapped = arith.addi %scan, %scan : tensor<128xi32, #blocked>
    %converted = ttg.convert_layout %wrapped : tensor<128xi32, #blocked> -> tensor<128xi32, #mma_slice>
    tt.return %converted : tensor<128xi32, #mma_slice>
  }

  // The legality check must not disable rematerialization in a blocked target.
  // CHECK-LABEL: @scan_accept_blocked
  // CHECK-NOT: ttg.convert_layout
  // CHECK: tt.make_range {{.*}} : tensor<128xi32, #[[$DST]]>
  // CHECK-NOT: ttg.convert_layout
  // CHECK: "tt.scan"
  // CHECK: : (tensor<128xi32, #[[$DST]]>) -> tensor<128xi32, #[[$DST]]>
  // CHECK-NOT: ttg.convert_layout
  // CHECK: %[[WRAPPED:.+]] = arith.addi {{.*}} : tensor<128xi32, #[[$DST]]>
  // CHECK-NOT: ttg.convert_layout
  // CHECK: tt.return %[[WRAPPED]]
  tt.func public @scan_accept_blocked() -> tensor<128xi32, #blocked1> {
    %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #blocked>
    %scan = "tt.scan"(%range) ({
    ^bb0(%a: i32, %b: i32):
      %sum = arith.addi %a, %b : i32
      tt.scan.return %sum : i32
    }) {axis = 0 : i32, reverse = false} : (tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    %wrapped = arith.addi %scan, %scan : tensor<128xi32, #blocked>
    %converted = ttg.convert_layout %wrapped : tensor<128xi32, #blocked> -> tensor<128xi32, #blocked1>
    tt.return %converted : tensor<128xi32, #blocked1>
  }

  // Regression topology from https://github.com/flagos-ai/FlagTree/issues/1440:
  // two independent scans share a runtime load; one supplies row addresses,
  // the other supplies values, and both meet at one 16x4 store. The issue
  // reported sm90; this pass-level fixture uses the file's cuda:80 target.
  // Both convert sources are single-use wrappers, allowing producer closure
  // collection to reach scan. The fixed pass must reject the slice target;
  // it must not form/merge those proposals and retag the scans to that slice.
  // Keep the scan operations separate: do not run CSE before this pass.
  // This is a compilation/layout test, not a GPU value test: the original
  // kernel can write different values to overlapping output addresses.
  // CHECK-LABEL: @shared_load_two_scans_store
  // CHECK: %[[INPUT:.+]] = tt.load {{.*}} : tensor<16x!tt.ptr<i32>, #{{blocked[0-9]*}}>
  // CHECK: %[[ROW_SCAN:.+]] = "tt.scan"(%[[INPUT]])
  // CHECK: : (tensor<16xi32, #{{blocked[0-9]*}}>) -> tensor<16xi32, #{{blocked[0-9]*}}>
  // CHECK: %[[VALUE_SCAN:.+]] = "tt.scan"(%[[INPUT]])
  // CHECK: : (tensor<16xi32, #{{blocked[0-9]*}}>) -> tensor<16xi32, #{{blocked[0-9]*}}>
  // CHECK: arith.subi %[[VALUE_SCAN]], %[[INPUT]]
  // CHECK: tt.store %{{.*}}, %{{.*}} : tensor<16x4x!tt.ptr<i32>, #{{blocked[0-9]*}}>
  // CHECK: tt.return
  tt.func public @shared_load_two_scans_store(%x: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %out: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
    %range = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #blocked>
    %xb = tt.splat %x : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>, #blocked>
    %xp = tt.addptr %xb, %range : tensor<16x!tt.ptr<i32>, #blocked>, tensor<16xi32, #blocked>
    %v = tt.load %xp : tensor<16x!tt.ptr<i32>, #blocked>
    %a = "tt.scan"(%v) ({
    ^bb0(%lhs: i32, %rhs: i32):
      %sum = arith.addi %lhs, %rhs : i32
      tt.scan.return %sum : i32
    }) {axis = 0 : i32, reverse = false} : (tensor<16xi32, #blocked>) -> tensor<16xi32, #blocked>
    %s = "tt.scan"(%v) ({
    ^bb0(%lhs: i32, %rhs: i32):
      %sum = arith.addi %lhs, %rhs : i32
      tt.scan.return %sum : i32
    }) {axis = 0 : i32, reverse = false} : (tensor<16xi32, #blocked>) -> tensor<16xi32, #blocked>
    %b = arith.subi %s, %v : tensor<16xi32, #blocked>
    %ob = tt.splat %out : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>, #blocked>
    %pa = tt.addptr %ob, %a : tensor<16x!tt.ptr<i32>, #blocked>, tensor<16xi32, #blocked>
    %pc = ttg.convert_layout %pa : tensor<16x!tt.ptr<i32>, #blocked> -> tensor<16x!tt.ptr<i32>, #row_slice>
    %bc = ttg.convert_layout %b : tensor<16xi32, #blocked> -> tensor<16xi32, #row_slice>
    %pe = tt.expand_dims %pc {axis = 1 : i32} : tensor<16x!tt.ptr<i32>, #row_slice> -> tensor<16x1x!tt.ptr<i32>, #parent>
    %be = tt.expand_dims %bc {axis = 1 : i32} : tensor<16xi32, #row_slice> -> tensor<16x1xi32, #parent>
    %pb = tt.broadcast %pe : tensor<16x1x!tt.ptr<i32>, #parent> -> tensor<16x4x!tt.ptr<i32>, #parent>
    %bb = tt.broadcast %be : tensor<16x1xi32, #parent> -> tensor<16x4xi32, #parent>
    %cols = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32, #slice>
    %ce = tt.expand_dims %cols {axis = 0 : i32} : tensor<4xi32, #slice> -> tensor<1x4xi32, #parent>
    %cb = tt.broadcast %ce : tensor<1x4xi32, #parent> -> tensor<16x4xi32, #parent>
    %ptr = tt.addptr %pb, %cb : tensor<16x4x!tt.ptr<i32>, #parent>, tensor<16x4xi32, #parent>
    %val = arith.addi %bb, %cb : tensor<16x4xi32, #parent>
    tt.store %ptr, %val : tensor<16x4x!tt.ptr<i32>, #parent>
    tt.return
  }
}

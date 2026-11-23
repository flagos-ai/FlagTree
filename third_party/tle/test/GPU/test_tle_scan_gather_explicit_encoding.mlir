// Copyright 2025- FlagOS Contributors
// SPDX-License-Identifier: MIT
// RUN: triton-opt %s -convert-triton-to-tritongpu='target=cuda:90 num-warps=8 enable-source-remat=false' | FileCheck %s

// A dynamic gather propagates its explicit index layout to a scan result.
// The scan must retain its input layout and convert each result separately.
#gather = #ttg.linear<{register = [[0, 8]], lane = [[0, 0], [0, 0], [0, 1], [0, 2], [0, 4]], warp = [[0, 0], [0, 0], [0, 0]], block = []}>
// CHECK: #[[$GATHER:.*]] = #ttg.linear<{{.*}}>

module {
  // CHECK-LABEL: tt.func @scan_to_dynamic_gather
  tt.func @scan_to_dynamic_gather(%input: !tt.ptr<f32>, %indices: !tt.ptr<i32>, %output: !tt.ptr<f32>) {
    %ip = tt.splat %input : !tt.ptr<f32> -> tensor<64x16x!tt.ptr<f32>>
    %values = tt.load %ip : tensor<64x16x!tt.ptr<f32>>
    %jp = tt.splat %indices : !tt.ptr<i32> -> tensor<1x16x!tt.ptr<i32>>
    %index = tt.load %jp : tensor<1x16x!tt.ptr<i32>>
    %explicit = tle.gpu.set_layout %index {target_encoding = #gather} : tensor<1x16xi32> -> tensor<1x16xi32>
    // CHECK: %[[SCAN:.*]] = "tt.scan"({{.*}})
    %prefix = "tt.scan"(%values) ({
    ^bb0(%left: f32, %right: f32):
      %maximum = arith.maximumf %left, %right : f32
      "tt.scan.return"(%maximum) : (f32) -> ()
    }) {axis = 0 : i32, reverse = false} : (tensor<64x16xf32>) -> tensor<64x16xf32>
    // CHECK: %[[CONVERTED:.*]] = ttg.convert_layout %[[SCAN]] {{.*}} : {{.*}} -> tensor<64x16xf32, #[[$GATHER]]>
    // CHECK-NOT: unrealized_conversion_cast
    // CHECK: %[[RESULT:.*]] = tt.gather %[[CONVERTED]][{{.*}}] {{.*}} -> tensor<1x16xf32, #[[$GATHER]]>
    %selected = tt.gather %prefix[%explicit] {axis = 0 : i32} : (tensor<64x16xf32>, tensor<1x16xi32>) -> tensor<1x16xf32>
    %op = tt.splat %output : !tt.ptr<f32> -> tensor<1x16x!tt.ptr<f32>>
    tt.store %op, %selected : tensor<1x16x!tt.ptr<f32>>
    tt.return
  }

  // CHECK-LABEL: tt.func @variadic_reverse_scan_to_gather
  tt.func @variadic_reverse_scan_to_gather(%input: !tt.ptr<f32>, %integer_input: !tt.ptr<i32>, %indices: !tt.ptr<i32>, %output: !tt.ptr<f32>, %integer_output: !tt.ptr<i32>) {
    %ip = tt.splat %input : !tt.ptr<f32> -> tensor<64x16x!tt.ptr<f32>>
    %values = tt.load %ip : tensor<64x16x!tt.ptr<f32>>
    %np = tt.splat %integer_input : !tt.ptr<i32> -> tensor<64x16x!tt.ptr<i32>>
    %numbers = tt.load %np : tensor<64x16x!tt.ptr<i32>>
    %jp = tt.splat %indices : !tt.ptr<i32> -> tensor<1x16x!tt.ptr<i32>>
    %index = tt.load %jp : tensor<1x16x!tt.ptr<i32>>
    %explicit = tle.gpu.set_layout %index {target_encoding = #gather} : tensor<1x16xi32> -> tensor<1x16xi32>
    // CHECK: %[[PAIR:.*]]:2 = "tt.scan"({{.*}})
    %pair:2 = "tt.scan"(%values, %numbers) ({
    ^bb0(%left: f32, %left_number: i32, %right: f32, %right_number: i32):
      %maximum = arith.maximumf %left, %right : f32
      %integer_maximum = arith.maxsi %left_number, %right_number : i32
      "tt.scan.return"(%maximum, %integer_maximum) : (f32, i32) -> ()
    }) {axis = 0 : i32, reverse = true} : (tensor<64x16xf32>, tensor<64x16xi32>) -> (tensor<64x16xf32>, tensor<64x16xi32>)
    // CHECK: %[[FLOAT:.*]] = ttg.convert_layout %[[PAIR]]#0 {{.*}} : {{.*}} -> tensor<64x16xf32, #[[$GATHER]]>
    // CHECK: %[[INTEGER:.*]] = ttg.convert_layout %[[PAIR]]#1 {{.*}} : {{.*}} -> tensor<64x16xi32, #[[$GATHER]]>
    // CHECK-NOT: unrealized_conversion_cast
    // CHECK: tt.gather %[[FLOAT]]
    // CHECK: tt.gather %[[INTEGER]]
    %selected = tt.gather %pair#0[%explicit] {axis = 0 : i32} : (tensor<64x16xf32>, tensor<1x16xi32>) -> tensor<1x16xf32>
    %selected_integer = tt.gather %pair#1[%explicit] {axis = 0 : i32} : (tensor<64x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %op = tt.splat %output : !tt.ptr<f32> -> tensor<1x16x!tt.ptr<f32>>
    tt.store %op, %selected : tensor<1x16x!tt.ptr<f32>>
    %no = tt.splat %integer_output : !tt.ptr<i32> -> tensor<1x16x!tt.ptr<i32>>
    tt.store %no, %selected_integer : tensor<1x16x!tt.ptr<i32>>
    tt.return
  }

  // CHECK-LABEL: tt.func @ordinary_scan_keeps_input_encoding
  tt.func @ordinary_scan_keeps_input_encoding(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>) {
    %ip = tt.splat %input : !tt.ptr<f32> -> tensor<64x16x!tt.ptr<f32>>
    %values = tt.load %ip : tensor<64x16x!tt.ptr<f32>>
    // CHECK: %[[ORDINARY:.*]] = "tt.scan"({{.*}})
    %prefix = "tt.scan"(%values) ({
    ^bb0(%left: f32, %right: f32):
      %maximum = arith.maximumf %left, %right : f32
      "tt.scan.return"(%maximum) : (f32) -> ()
    }) {axis = 0 : i32, reverse = false} : (tensor<64x16xf32>) -> tensor<64x16xf32>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.store {{.*}}, %[[ORDINARY]]
    %op = tt.splat %output : !tt.ptr<f32> -> tensor<64x16x!tt.ptr<f32>>
    tt.store %op, %prefix : tensor<64x16x!tt.ptr<f32>>
    tt.return
  }
}

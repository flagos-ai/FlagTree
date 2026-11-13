// RUN: triton-opt %s --verify-each '-graph-optimize=rule-mask=1' -o - | FileCheck %s --check-prefix=GRAPH
// RUN: triton-opt %s --verify-each '-graph-optimize=rule-mask=1' -triton-to-linalg -o /dev/null
// RUN: triton-opt %s --verify-each '-graph-optimize=rule-mask=1' -triton-to-linalg -o - | FileCheck %s --check-prefix=LOWER

// Keep a lowering regression on the unified, type-changing layout rewrite.
// The original access is tensor<2x3>; the graph pass must emit the equivalent
// tensor<3x2> access before the normal Triton-to-Linalg lowering runs.
// GRAPH-LABEL: tt.func @lower_unified_layout(
// GRAPH: tt.assert {{.*}}, "int32 overflow detected for operation mul"{{.*}} : tensor<3x1xi1>
// GRAPH: tt.load {{.*}} : tensor<3x2x!tt.ptr<f32>>
// GRAPH: math.tan {{.*}} : tensor<3x2xf32>
// GRAPH: tt.store {{.*}} : tensor<3x2x!tt.ptr<f32>>
tt.func @lower_unified_layout(%source: !tt.ptr<f32>, %destination: !tt.ptr<f32>) {
  %axis0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %axis1 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %axis0_expand = tt.expand_dims %axis0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %axis1_expand = tt.expand_dims %axis1 {axis = 0 : i32} : tensor<3xi32> -> tensor<1x3xi32>
  %two = arith.constant 2 : i32
  %axis1_scale = tt.splat %two : i32 -> tensor<1x3xi32>
  %axis1_scaled = arith.muli %axis1_expand, %axis1_scale : tensor<1x3xi32>
  %axis1_scaled_i64 = arith.extsi %axis1_scaled : tensor<1x3xi32> to tensor<1x3xi64>
  %axis1_max = arith.constant dense<2147483647> : tensor<1x3xi64>
  %axis1_min = arith.constant dense<-2147483648> : tensor<1x3xi64>
  %axis1_le = arith.cmpi sle, %axis1_scaled_i64, %axis1_max : tensor<1x3xi64>
  %axis1_ge = arith.cmpi sge, %axis1_scaled_i64, %axis1_min : tensor<1x3xi64>
  %axis1_ok = arith.andi %axis1_le, %axis1_ge : tensor<1x3xi1>
  tt.assert %axis1_ok, "int32 overflow detected for operation mul" {tt.auto_overflow_assert} : tensor<1x3xi1>
  %axis0_full = tt.broadcast %axis0_expand : tensor<2x1xi32> -> tensor<2x3xi32>
  %axis1_full = tt.broadcast %axis1_scaled : tensor<1x3xi32> -> tensor<2x3xi32>
  %offset = arith.addi %axis0_full, %axis1_full : tensor<2x3xi32>
  %source_base = tt.splat %source : !tt.ptr<f32> -> tensor<2x3x!tt.ptr<f32>>
  %destination_base = tt.splat %destination : !tt.ptr<f32> -> tensor<2x3x!tt.ptr<f32>>
  %source_ptr = tt.addptr %source_base, %offset : tensor<2x3x!tt.ptr<f32>>, tensor<2x3xi32>
  %destination_ptr = tt.addptr %destination_base, %offset : tensor<2x3x!tt.ptr<f32>>, tensor<2x3xi32>
  %loaded = tt.load %source_ptr : tensor<2x3x!tt.ptr<f32>>
  %result = math.tan %loaded : tensor<2x3xf32>
  tt.store %destination_ptr, %result : tensor<2x3x!tt.ptr<f32>>
  tt.return
}

// The matching message alone must not make a user-authored device_assert
// disappear during lowering.  Only the private frontend marker identifies an
// automatic overflow assertion.
// LOWER: func.func private @{{.*}}(i1) attributes {msg = "int32 overflow detected for operation mul"}
// LOWER-LABEL: func.func @lower_user_same_message_assert(
// LOWER: call @{{.*}}({{.*}}) : (i1) -> ()
// LOWER-NOT: call @
// LOWER: return
tt.func @lower_user_same_message_assert() {
  %true = arith.constant true
  tt.assert %true, "int32 overflow detected for operation mul" {tt.auto_overflow_assert} : i1
  tt.assert %true, "int32 overflow detected for operation mul" : i1
  tt.return
}

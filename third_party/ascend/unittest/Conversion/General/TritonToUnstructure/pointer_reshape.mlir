// RUN: triton-opt %s --triton-to-unstructure | FileCheck %s --check-prefixes=CHECK,SCALAR
// RUN: triton-opt %s --triton-to-unstructure=compile-on-910-95=true | FileCheck %s --check-prefixes=CHECK,INDIRECT

// Rank reduction must retain the common base, the i64 offsets, the mask, and
// the nonzero "other" value. It must not leave a pointer reshape for T2L.
tt.func @masked_pointer_reshape(%base: !tt.ptr<f32>, %offsets: tensor<2x2x4xi64>, %mask: tensor<4x4xi1>) -> tensor<4x4xf32> {
  %other = arith.constant dense<-7.0> : tensor<4x4xf32>
  %bases = tt.splat %base : !tt.ptr<f32> -> tensor<2x2x4x!tt.ptr<f32>>
  %ptrs = tt.addptr %bases, %offsets : tensor<2x2x4x!tt.ptr<f32>>, tensor<2x2x4xi64>
  %flat = tt.reshape %ptrs : tensor<2x2x4x!tt.ptr<f32>> -> tensor<4x4x!tt.ptr<f32>>
  %loaded = tt.load %flat, %mask, %other : tensor<4x4x!tt.ptr<f32>>
  tt.return %loaded : tensor<4x4xf32>
}

// CHECK-LABEL: tt.func @masked_pointer_reshape
// CHECK-SAME:  %[[BASE:.*]]: !tt.ptr<f32>, %[[OFFSETS:.*]]: tensor<2x2x4xi64>, %[[MASK:.*]]: tensor<4x4xi1>
// CHECK-DAG:   %[[OTHER:.*]] = arith.constant dense<-7.000000e+00> : tensor<4x4xf32>
// CHECK-DAG:   %[[FLAT:.*]] = tt.reshape %[[OFFSETS]] : tensor<2x2x4xi64> -> tensor<4x4xi64>
// INDIRECT:    ascend.indirect_load %[[BASE]] : <f32>, %[[FLAT]] : tensor<4x4xi64>, %[[MASK]] : tensor<4x4xi1>, %[[OTHER]] : tensor<4x4xf32>
// SCALAR:      %[[LOADED:.*]] = scf.for
// SCALAR:      tensor.extract %[[FLAT]]
// SCALAR:      tt.load
// SCALAR:      arith.select %[[MASK]], %[[LOADED]], %[[OTHER]]
// CHECK:       tt.return

// Reshape the complete displacement after addptr analysis has performed the
// signed extension. The original narrow integer computation remains intact.
tt.func @store_pointer_reshape(%base: !tt.ptr<i32>, %offsets: tensor<2x4xi32>, %values: tensor<8xi32>, %mask: tensor<8xi1>) {
  %bases = tt.splat %base : !tt.ptr<i32> -> tensor<2x4x!tt.ptr<i32>>
  %ptrs = tt.addptr %bases, %offsets : tensor<2x4x!tt.ptr<i32>>, tensor<2x4xi32>
  %flat = tt.reshape %ptrs : tensor<2x4x!tt.ptr<i32>> -> tensor<8x!tt.ptr<i32>>
  tt.store %flat, %values, %mask : tensor<8x!tt.ptr<i32>>
  tt.return
}

// CHECK-LABEL: tt.func @store_pointer_reshape
// CHECK-SAME:  %[[BASE:.*]]: !tt.ptr<i32>, %[[OFFSETS:.*]]: tensor<2x4xi32>, %[[VALUES:.*]]: tensor<8xi32>, %[[MASK:.*]]: tensor<8xi1>
// CHECK:       %[[WIDE:.*]] = arith.extsi %[[OFFSETS]] : tensor<2x4xi32> to tensor<2x4xi64>
// CHECK:       %[[FLAT:.*]] = tt.reshape %[[WIDE]] : tensor<2x4xi64> -> tensor<8xi64>
// INDIRECT:    ascend.indirect_store %[[BASE]] : <i32>, %[[FLAT]] : tensor<8xi64>, %[[VALUES]] : tensor<8xi32>, %[[MASK]] : tensor<8xi1>
// SCALAR:      scf.for
// SCALAR:      tensor.extract %[[FLAT]]
// SCALAR:      tt.store
// CHECK:       tt.return

// Fix one legal lane order for both analyzed memory uses and remaining pointer
// users. The layout hint and the unreshaped pointer's other uses survive.
tt.func @reshape_attributes_and_other_use(%base: !tt.ptr<f32>, %offsets: tensor<8xi32>) -> (tensor<2x4xf32>, tensor<2x4x!tt.ptr<f32>>, tensor<8x!tt.ptr<f32>>) {
  %bases = tt.splat %base : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %bases, %offsets : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %reshaped = tt.reshape %ptrs allow_reorder efficient_layout : tensor<8x!tt.ptr<f32>> -> tensor<2x4x!tt.ptr<f32>>
  %loaded = tt.load %reshaped : tensor<2x4x!tt.ptr<f32>>
  tt.return %loaded, %reshaped, %ptrs : tensor<2x4xf32>, tensor<2x4x!tt.ptr<f32>>, tensor<8x!tt.ptr<f32>>
}

// CHECK-LABEL: tt.func @reshape_attributes_and_other_use
// CHECK-SAME:  %[[BASE:.*]]: !tt.ptr<f32>, %[[OFFSETS:.*]]: tensor<8xi32>
// CHECK:       %[[WIDE:.*]] = arith.extsi %[[OFFSETS]] : tensor<8xi32> to tensor<8xi64>
// CHECK:       %[[ORIGINAL:.*]] = tt.addptr {{.*}} : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
// CHECK:       %[[OFFSET:.*]] = tt.reshape %[[WIDE]] efficient_layout : tensor<8xi64> -> tensor<2x4xi64>
// CHECK:       %[[PTRS:.*]] = tt.reshape %[[ORIGINAL]] efficient_layout : tensor<8x!tt.ptr<f32>> -> tensor<2x4x!tt.ptr<f32>>
// INDIRECT:    ascend.indirect_load %[[BASE]] : <f32>, %[[OFFSET]] : tensor<2x4xi64>
// SCALAR:      scf.for
// CHECK:       tt.return {{.*}}, %[[PTRS]], %[[ORIGINAL]]

// A common base can be behind multiple addptr and reshape operations. Widen
// each addptr offset before accumulating it, then reshape the complete sum.
tt.func @nested_addptr_and_reshape(%base: !tt.ptr<f32>, %first: tensor<2x4xi32>, %second: tensor<2x4xi64>) -> tensor<8xf32> {
  %bases = tt.splat %base : !tt.ptr<f32> -> tensor<2x4x!tt.ptr<f32>>
  %p0 = tt.addptr %bases, %first : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
  %p1 = tt.addptr %p0, %second : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi64>
  %p2 = tt.reshape %p1 : tensor<2x4x!tt.ptr<f32>> -> tensor<4x2x!tt.ptr<f32>>
  %p3 = tt.reshape %p2 : tensor<4x2x!tt.ptr<f32>> -> tensor<8x!tt.ptr<f32>>
  %loaded = tt.load %p3 : tensor<8x!tt.ptr<f32>>
  tt.return %loaded : tensor<8xf32>
}

// CHECK-LABEL: tt.func @nested_addptr_and_reshape
// CHECK-SAME:  %[[BASE:.*]]: !tt.ptr<f32>, %[[FIRST:.*]]: tensor<2x4xi32>, %[[SECOND:.*]]: tensor<2x4xi64>
// CHECK:       %[[WIDE:.*]] = arith.extsi %[[FIRST]] : tensor<2x4xi32> to tensor<2x4xi64>
// CHECK:       %[[SUM:.*]] = arith.addi %[[WIDE]], %[[SECOND]] : tensor<2x4xi64>
// CHECK:       %[[FLAT:.*]] = tt.reshape %[[SUM]] : tensor<2x4xi64> -> tensor<8xi64>
// INDIRECT:    ascend.indirect_load %[[BASE]] : <f32>, %[[FLAT]] : tensor<8xi64>
// SCALAR:      scf.for
// SCALAR:      tensor.extract %[[FLAT]]
// CHECK:       tt.return

// A broadcast changes which lanes share offsets. Keep the complete broadcast
// displacement, without copying the source's axis classification through reshape.
tt.func @broadcast_pointer_reshape(%base: !tt.ptr<f32>, %offsets: tensor<2x1xi64>) -> tensor<8xf32> {
  %bases = tt.splat %base : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
  %ptrs = tt.addptr %bases, %offsets : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi64>
  %broadcast = tt.broadcast %ptrs : tensor<2x1x!tt.ptr<f32>> -> tensor<2x4x!tt.ptr<f32>>
  %flat = tt.reshape %broadcast : tensor<2x4x!tt.ptr<f32>> -> tensor<8x!tt.ptr<f32>>
  %loaded = tt.load %flat : tensor<8x!tt.ptr<f32>>
  tt.return %loaded : tensor<8xf32>
}

// CHECK-LABEL: tt.func @broadcast_pointer_reshape
// CHECK-SAME:  %[[BASE:.*]]: !tt.ptr<f32>, %[[OFFSETS:.*]]: tensor<2x1xi64>
// CHECK:       %[[BROADCAST:.*]] = tt.broadcast %[[OFFSETS]] : tensor<2x1xi64> -> tensor<2x4xi64>
// CHECK:       %[[FLAT:.*]] = tt.reshape %[[BROADCAST]] : tensor<2x4xi64> -> tensor<8xi64>
// INDIRECT:    ascend.indirect_load %[[BASE]] : <f32>, %[[FLAT]] : tensor<8xi64>
// SCALAR:      scf.for
// CHECK:       tt.return

// setUnstructured must not leave scalarLike=true, which would bypass the
// conservative access path before conversion examines any individual axis.
tt.func @uniform_pointer_reshape(%base: !tt.ptr<f32>) -> tensor<8xf32> {
  %bases = tt.splat %base : !tt.ptr<f32> -> tensor<2x4x!tt.ptr<f32>>
  %flat = tt.reshape %bases : tensor<2x4x!tt.ptr<f32>> -> tensor<8x!tt.ptr<f32>>
  %loaded = tt.load %flat : tensor<8x!tt.ptr<f32>>
  tt.return %loaded : tensor<8xf32>
}

// CHECK-LABEL: tt.func @uniform_pointer_reshape
// INDIRECT:    ascend.indirect_load
// SCALAR:      scf.for
// CHECK:       tt.return

// A lane-wise base is not a common scalar base. Extract actual reshaped
// pointers lane by lane, rather than choosing one allocation for indirect_load.
tt.func @keep_lane_wise_base(%bases: tensor<2x4x!tt.ptr<f32>>, %offsets: tensor<2x4xi32>) -> tensor<8xf32> {
  %ptrs = tt.addptr %bases, %offsets : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
  %flat = tt.reshape %ptrs : tensor<2x4x!tt.ptr<f32>> -> tensor<8x!tt.ptr<f32>>
  %loaded = tt.load %flat : tensor<8x!tt.ptr<f32>>
  tt.return %loaded : tensor<8xf32>
}

// CHECK-LABEL: tt.func @keep_lane_wise_base
// CHECK:       %[[PTRS:.*]] = tt.addptr
// CHECK:       %[[FLAT:.*]] = tt.reshape %[[PTRS]] : tensor<2x4x!tt.ptr<f32>> -> tensor<8x!tt.ptr<f32>>
// CHECK:       scf.for
// CHECK:       tensor.extract %[[FLAT]]
// CHECK:       tt.load
// CHECK-NOT:   ascend.indirect_load
// CHECK:       tt.return

// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s

// CHECK:           #map = affine_map<(d0) -> (d0)>
// CHECK:           #map1 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK:           #map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

tt.func public @assert_tensor_1d() {
  %0 = tensor.empty() : tensor<4xi1>
  tt.assert %0, "message" : tensor<4xi1>
  tt.return
}

// CHECK-LABEL:   func.func @assert_tensor_1d
// CHECK-NOT:       tt.assert
// CHECK:           linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} ins(%0 : tensor<4xi1>) { 
// CHECK:           ^bb0(%in: i1):
// CHECK:             cf.assert %in, "Assertion `message` failed"
// CHECK:             linalg.yield
// CHECK:           }
// CHECK-NOT:       tt.assert

tt.func public @assert_tensor_2d() {
  %0 = tensor.empty() : tensor<4x4xi1>
  tt.assert %0, "message" : tensor<4x4xi1>
  tt.return
}

// CHECK-LABEL:   func.func @assert_tensor_2d
// CHECK-NOT:       tt.assert
// CHECK:           linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<4x4xi1>) { 
// CHECK:           ^bb0(%in: i1):
// CHECK:             cf.assert %in, "Assertion `message` failed"
// CHECK:             linalg.yield
// CHECK:           }
// CHECK-NOT:       tt.assert

tt.func public @assert_tensor_3d() {
  %0 = tensor.empty() : tensor<4x4x4xi1>
  tt.assert %0, "message" : tensor<4x4x4xi1>
  tt.return
}

// CHECK-LABEL:   func.func @assert_tensor_3d
// CHECK-NOT:       tt.assert
// CHECK:           linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<4x4x4xi1>) { 
// CHECK:           ^bb0(%in: i1):
// CHECK:             cf.assert %in, "Assertion `message` failed"
// CHECK:             linalg.yield
// CHECK:           }
// CHECK-NOT:       tt.assert

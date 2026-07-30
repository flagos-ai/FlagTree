// RUN: triton-shared-opt --triton-ptr-to-memref %s | FileCheck %s

#map = affine_map<(d0) -> (d0)>
module {
  func.func public @add_kernel_01234(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
    %0 = builtin.unrealized_conversion_cast %arg2 : !tt.ptr<f32> to memref<*xf32>
    %1 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to memref<*xf32>
    %2 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to memref<*xf32>
    %c1024_i32 = arith.constant 1024 : i32
    %3 = tt.get_program_id x : i32
    %4 = arith.muli %3, %c1024_i32 : i32
    %5 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %6 = tt.splat %4 : i32 -> tensor<1024xi32>
    %7 = arith.addi %6, %5 : tensor<1024xi32>
    %8 = tt.splat %arg3 : i32 -> tensor<1024xi32>
    %9 = arith.cmpi slt, %7, %8 : tensor<1024xi32>
    %cast = memref.cast %2 : memref<*xf32> to memref<?xf32>
    %10 = bufferization.to_tensor %cast restrict : memref<?xf32> to tensor<?xf32>
    %11 = tensor.empty() : tensor<1024xf32>
    %12 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%7, %9 : tensor<1024xi32>, tensor<1024xi1>) outs(%11 : tensor<1024xf32>) {
    ^bb0(%in: i32, %in_2: i1, %out: f32):
      %17 = scf.if %in_2 -> (f32) {
        %18 = arith.index_cast %in : i32 to index
        %extracted = tensor.extract %10[%18] : tensor<?xf32>
        scf.yield %extracted : f32
      } else {
        %cst = arith.constant 0.000000e+00 : f32
        scf.yield %cst : f32
      }
      linalg.yield %17 : f32
    } -> tensor<1024xf32>
    %cast_0 = memref.cast %1 : memref<*xf32> to memref<?xf32>
    %13 = bufferization.to_tensor %cast_0 restrict : memref<?xf32> to tensor<?xf32>
    %14 = tensor.empty() : tensor<1024xf32>
    %15 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%7, %9 : tensor<1024xi32>, tensor<1024xi1>) outs(%14 : tensor<1024xf32>) {
    ^bb0(%in: i32, %in_2: i1, %out: f32):
      %17 = scf.if %in_2 -> (f32) {
        %18 = arith.index_cast %in : i32 to index
        %extracted = tensor.extract %13[%18] : tensor<?xf32>
        scf.yield %extracted : f32
      } else {
        %cst = arith.constant 0.000000e+00 : f32
        scf.yield %cst : f32
      }
      linalg.yield %17 : f32
    } -> tensor<1024xf32>
    %16 = arith.addf %12, %15 : tensor<1024xf32>
    %cast_1 = memref.cast %0 : memref<*xf32> to memref<?xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    affine.for %arg4 = 0 to 1024 {
      %extracted = tensor.extract %9[%arg4] : tensor<1024xi1>
      scf.if %extracted {
        %extracted_2 = tensor.extract %16[%arg4] : tensor<1024xf32>
        %extracted_3 = tensor.extract %7[%arg4] : tensor<1024xi32>
        %17 = arith.index_cast %extracted_3 : i32 to index
        memref.store %extracted_2, %cast_1[%17] : memref<?xf32>
      }
    }
    return
  }
}

// CHECK:   func.func public @add_kernel_01234(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: memref<*xf32>, %arg3: i32)
// CHECK-NOT: builtin.unrealized_conversion_cast %arg2 : memref<*xf32> to !tt.ptr<f32>
// CHECK-NOT: builtin.unrealized_conversion_cast %arg1 : memref<*xf32> to !tt.ptr<f32>
// CHECK-NOT: builtin.unrealized_conversion_cast %arg0 : memref<*xf32> to !tt.ptr<f32>

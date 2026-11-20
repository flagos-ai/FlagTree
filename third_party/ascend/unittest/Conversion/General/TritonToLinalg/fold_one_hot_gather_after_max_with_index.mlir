// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s

module {
  func.func @fold_one_hot_gather_after_max_with_index(%logits: tensor<2x4xf32>, %indices: tensor<2x4xi32>, %mask: tensor<2x4xi1>) -> tensor<2x1xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<2x4xf32>
    %masked = arith.select %mask, %logits, %cst : tensor<2x4xi1>, tensor<2x4xf32>
    %max:2 = "tt.reduce"(%masked, %indices) <{axis = 1 : i32}> ({
    ^bb0(%in: f32, %in_index: i32, %init: f32, %init_index: i32):
      %is_greater = arith.cmpf ogt, %in, %init : f32
      %new_value = arith.select %is_greater, %in, %init : f32
      %new_index = arith.select %is_greater, %in_index, %init_index : i32
      tt.reduce.return %new_value, %new_index : f32, i32
    }) : (tensor<2x4xf32>, tensor<2x4xi32>) -> (tensor<2xf32>, tensor<2xi32>)
    %expanded_index = tt.expand_dims %max#1 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %broadcasted_index = tt.broadcast %expanded_index : tensor<2x1xi32> -> tensor<2x4xi32>
    %one_hot = arith.cmpi eq, %broadcasted_index, %indices : tensor<2x4xi32>
    %one_hot_f32 = arith.uitofp %one_hot : tensor<2x4xi1> to tensor<2x4xf32>
    %selected = arith.mulf %logits, %one_hot_f32 : tensor<2x4xf32>
    %sum = "tt.reduce"(%selected) <{axis = 1 : i32}> ({
    ^bb0(%in: f32, %init: f32):
      %add = arith.addf %in, %init : f32
      tt.reduce.return %add : f32
    }) : (tensor<2x4xf32>) -> tensor<2xf32>
    %result = tensor.expand_shape %sum [[0, 1]] output_shape [2, 1] : tensor<2xf32> into tensor<2x1xf32>
    return %result : tensor<2x1xf32>
  }
}

// CHECK-LABEL: func.func @fold_one_hot_gather_after_max_with_index
// CHECK: %[[MAX:.*]]:2 = linalg.reduce
// CHECK-SAME: reduce_mode = "max_with_index"
// CHECK-NOT: arith.uitofp
// CHECK-NOT: arith.mulf
// CHECK-NOT: arith.cmpi eq
// CHECK-NOT: linalg.reduce ins(%{{.*}} : tensor<2x4xf32>) outs(%{{.*}} : tensor<2xf32>) dimensions = [1]
// CHECK: tensor.expand_shape %[[MAX]]#0 {{\[\[}}0, 1{{\]\]}} output_shape [2, 1]

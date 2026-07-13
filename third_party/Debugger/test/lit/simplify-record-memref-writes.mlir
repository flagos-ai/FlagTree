module {
  func.func @kernel(%debug: memref<?xi32> {flagtree.debug.hidden_arg = "__debug_ctrl_ptr"},
                    %out: memref<?xi32>) {
    %c0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    %c7_i32 = arith.constant 7 : i32
    %c9_i32 = arith.constant 9 : i32

    %empty = "tensor.empty"() : () -> tensor<1xi32>
    %inserted = "tensor.insert"(%c7_i32, %empty, %c0)
      : (i32, tensor<1xi32>, index) -> tensor<1xi32>
    %debug_dst = memref.reinterpret_cast %debug to offset: [5], sizes: [1], strides: [1]
      : memref<?xi32> to memref<1xi32, strided<[1], offset: 5>>
    "bufferization.materialize_in_destination"(%inserted, %debug_dst)
      : (tensor<1xi32>, memref<1xi32, strided<[1], offset: 5>>) -> ()

    %filled = "linalg.fill"(%c9_i32, %empty)
      : (i32, tensor<1xi32>) -> tensor<1xi32>
    %out_dst = memref.reinterpret_cast %out to offset: [%c5], sizes: [1], strides: [1]
      : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    "bufferization.materialize_in_destination"(%filled, %out_dst)
      : (tensor<1xi32>, memref<1xi32, strided<[1], offset: ?>>) -> ()

    return
  }
}

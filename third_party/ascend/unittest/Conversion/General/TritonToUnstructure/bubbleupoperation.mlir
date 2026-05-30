// RUN: triton-opt %s --bubble-up-operation | FileCheck %s

// CHECK-LABEL: tt.func @test_subi_extract_bubbleup
tt.func @test_subi_extract_bubbleup(%a: tensor<128xi32>, %b: tensor<128xi32>, %i: index, %c: i32) -> i32 {
    %0 = arith.subi %a, %b : tensor<128xi32>
    %1 = tensor.extract %0[%i] : tensor<128xi32>
    %2 = arith.muli %1, %c : i32
    tt.return %2 : i32
}


// CHECK-LABEL: tt.func @test_maxsi_extract_bubbleup
tt.func @test_maxsi_extract_bubbleup(%a: tensor<128xi32>, %b: tensor<128xi32>, %i: index, %c: i32) -> i32 {
    %0 = arith.maxsi %a, %b : tensor<128xi32>
    %1 = tensor.extract %0[%i] : tensor<128xi32>
    %2 = arith.muli %1, %c : i32
    tt.return %2 : i32
}


// CHECK-LABEL: tt.func @test_minsi_extract_bubbleup
tt.func @test_minsi_extract_bubbleup(%a: tensor<128xi32>, %b: tensor<128xi32>, %i: index, %c: i32) -> i32 {
    %0 = arith.minsi %a, %b : tensor<128xi32>
    %1 = tensor.extract %0[%i] : tensor<128xi32>
    %2 = arith.muli %1, %c : i32
    tt.return %2 : i32
}


// CHECK-LABEL: tt.func @test_extf_extract_bubbleup
tt.func @test_extf_extract_bubbleup(%a: tensor<128xf16>, %i: index, %c: f32) -> f32 {
    %0 = arith.extf %a : tensor<128xf16> to tensor<128xf32>
    %1 = tensor.extract %0[%i] : tensor<128xf32>
    %2 = arith.mulf %1, %c : f32
    tt.return %2 : f32
}


// CHECK-LABEL: tt.func @test_minnumf_extract_bubbleup
tt.func @test_minnumf_extract_bubbleup(%a: tensor<128xf32>, %b: tensor<128xf32>, %i: index, %c: f32) -> f32 {
    %0 = arith.minnumf %a, %b : tensor<128xf32>
    %1 = tensor.extract %0[%i] : tensor<128xf32>
    %2 = arith.mulf %1, %c : f32
    tt.return %2 : f32
}


// CHECK-LABEL: tt.func @test_maxnumf_extract_bubbleup
tt.func @test_maxnumf_extract_bubbleup(%a: tensor<128xf32>, %b: tensor<128xf32>, %i: index, %c: f32) -> f32 {
    %0 = arith.maxnumf %a, %b : tensor<128xf32>
    %1 = tensor.extract %0[%i] : tensor<128xf32>
    %2 = arith.mulf %1, %c : f32
    tt.return %2 : f32
}


// CHECK-LABEL: tt.func @test_cmpf_extract_bubbleup
tt.func @test_cmpf_extract_bubbleup(%a: tensor<128xf32>, %b: tensor<128xf32>, %i: index) -> i1 {
    %0 = arith.cmpf olt, %a, %b : tensor<128xf32>
    %1 = tensor.extract %0[%i] : tensor<128xi1>
    tt.return %1 : i1
}


// CHECK-LABEL: tt.func @test_addptr_extract_bubbleup
tt.func @test_addptr_extract_bubbleup(%a: tensor<128x!tt.ptr<f32>>, %b: tensor<128xi32>, %i: index) -> !tt.ptr<f32> {
    %0 = tt.addptr %a, %b : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %1 = tensor.extract %0[%i] : tensor<128x!tt.ptr<f32>>
    tt.return %1 : !tt.ptr<f32>
}


// CHECK-LABEL: tt.func @test_ceil_extract_bubbleup
tt.func @test_ceil_extract_bubbleup(%a: tensor<128xf32>, %i: index, %c: f32) -> f32 {
    %0 = math.ceil %a : tensor<128xf32>
    %1 = tensor.extract %0[%i] : tensor<128xf32>
    %2 = arith.mulf %1, %c : f32
    tt.return %2 : f32
}


// CHECK-LABEL: tt.func @test_slice_extract_dropdim_bubbleup
tt.func @test_slice_extract_dropdim_bubbleup(%a: tensor<128x128x128xf32>, %i: index, %j: index) -> f32 {
    %0 = tensor.extract_slice %a[0, %i, 0][1, 1, 128][1, 1, 1] : tensor<128x128x128xf32> to tensor<128xf32>
    %1 = tensor.extract %0[%j] : tensor<128xf32>
    tt.return %1 : f32
}


// CHECK-LABEL: tt.func @test_expand_slice_bubbleup
tt.func @test_expand_slice_bubbleup(%a: tensor<128xf32>, %i: index, %c: f32) -> tensor<1x1xf32> {
    %0 = tt.expand_dims %a {axis = 0 : i32} : tensor<128xf32> -> tensor<1x128xf32>
    %1 = tensor.extract_slice %0[0, %i][1, 1][1, 1] : tensor<1x128xf32> to tensor<1x1xf32>
    tt.return %1 : tensor<1x1xf32>
}


// CHECK-LABEL: tt.func @test_expand_slice_dropdim_bubbleup
tt.func @test_expand_slice_dropdim_bubbleup(%a: tensor<128x128xf32>, %i: index, %c: f32) -> tensor<128x1xf32> {
    %0 = tt.expand_dims %a {axis = 2 : i32} : tensor<128x128xf32> -> tensor<128x128x1xf32>
    %1 = tensor.extract_slice %0[%i, 0, 0][1, 128, 1][1, 1, 1] : tensor<128x128x1xf32> to tensor<128x1xf32>
    tt.return %1 : tensor<128x1xf32>
}


// CHECK-LABEL: tt.func @test_splat_slice_bubbleup
tt.func @test_splat_slice_bubbleup(%a: f32, %i: index, %c: f32) -> tensor<1xf32> {
    %0 = tt.splat %a : f32 -> tensor<128xf32>
    %1 = tensor.extract_slice %0[%i][1][1] : tensor<128xf32> to tensor<1xf32>
    tt.return %1 : tensor<1xf32>
}


// CHECK-LABEL: tt.func @test_makerange_slice_bubbleup
tt.func @test_makerange_slice_bubbleup(%i: index, %c: f32) -> tensor<1xi32> {
    %0 = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %1 = tensor.extract_slice %0[%i][1][1] : tensor<128xi32> to tensor<1xi32>
    tt.return %1 : tensor<1xi32>
}


// CHECK-LABEL: tt.func @test_slice_all_bubbleup
tt.func @test_slice_all_bubbleup(%i: index, %c: f32) -> tensor<128xi32> {
    %0 = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %1 = tensor.extract_slice %0[0][128][1] : tensor<128xi32> to tensor<128xi32>
    tt.return %1 : tensor<128xi32>
}

// RUN: triton-opt --discrete-mask-access-conversion "--triton-to-linalg=global-kernel=false named-ops=True" --split-input-file %s | FileCheck %s

// Case 1: other = uitofp(mask) -> select(mask, loaded, zeros)
// CHECK-LABEL: func.func @load_other_mask
// CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[MASK:.*]] = arith.cmpi slt
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<128xf32>
// CHECK: memref.copy
// CHECK: %[[LOADED:.*]] = bufferization.to_tensor %[[ALLOC]]
// CHECK: linalg.fill ins(%[[CST]] : f32)
// CHECK: arith.select %[[MASK]], %[[LOADED]], {{.*}}
// CHECK-NOT: tensor.insert_slice

module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @load_other_mask(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32) attributes {noinline = false} {
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %3 = tt.splat %1 : i32 -> tensor<128xi32>
    %4 = arith.addi %3, %2 : tensor<128xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<128xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<128xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %9 = arith.uitofp %6 : tensor<128xi1> to tensor<128xf32>
    %10 = tt.load %8, %6, %9 : tensor<128x!tt.ptr<f32>>
    %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %12, %10, %6 : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Case 1b: other = extui(mask) for int8 — same select with zeros.
// CHECK-LABEL: func.func @load_other_mask_i8
// CHECK: %[[CST:.*]] = arith.constant 0 : i8
// CHECK: %[[MASK:.*]] = arith.cmpi slt
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<128xi8>
// CHECK: memref.copy
// CHECK: %[[LOADED:.*]] = bufferization.to_tensor %[[ALLOC]]
// CHECK: linalg.fill ins(%[[CST]] : i8)
// CHECK: arith.select %[[MASK]], %[[LOADED]], {{.*}}
// CHECK-NOT: tensor.insert_slice

module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @load_other_mask_i8(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: i32) attributes {noinline = false} {
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %3 = tt.splat %1 : i32 -> tensor<128xi32>
    %4 = arith.addi %3, %2 : tensor<128xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<128xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<128xi32>
    %7 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<128x!tt.ptr<i8>>
    %8 = tt.addptr %7, %4 : tensor<128x!tt.ptr<i8>>, tensor<128xi32>
    %9 = arith.extui %6 : tensor<128xi1> to tensor<128xi8>
    %10 = tt.load %8, %6, %9 : tensor<128x!tt.ptr<i8>>
    %11 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<128x!tt.ptr<i8>>
    %12 = tt.addptr %11, %4 : tensor<128x!tt.ptr<i8>>, tensor<128xi32>
    tt.store %12, %10, %6 : tensor<128x!tt.ptr<i8>>
    tt.return
  }
}

// -----

// Case 2: other = prior full load tensor -> select(mask, loaded, other)
// CHECK-LABEL: func.func @load_other_tensor
// CHECK: %[[MASK:.*]] = arith.cmpi slt
// CHECK: memref.copy
// CHECK: %[[OTHER:.*]] = bufferization.to_tensor
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<128xf32>
// CHECK: memref.copy
// CHECK: %[[LOADED:.*]] = bufferization.to_tensor %[[ALLOC]]
// CHECK: arith.select %[[MASK]], %[[LOADED]], %[[OTHER]]
// CHECK-NOT: tensor.insert_slice

module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @load_other_tensor(%in_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %fill_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %N: i32) attributes {noinline = false} {
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %3 = tt.splat %1 : i32 -> tensor<128xi32>
    %4 = arith.addi %3, %2 : tensor<128xi32>
    %5 = tt.splat %N : i32 -> tensor<128xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<128xi32>
    %7 = tt.splat %fill_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %9 = tt.load %8 : tensor<128x!tt.ptr<f32>>
    %10 = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %11 = tt.addptr %10, %4 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %12 = tt.load %11, %6, %9 : tensor<128x!tt.ptr<f32>>
    %13 = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %14 = tt.addptr %13, %4 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %14, %12 : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Case 3: Scalar other=0.0 (unchanged fill path)
// CHECK-LABEL: func.func @load_other_scalar_zero
// CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<128xf32>
// CHECK: linalg.fill ins(%[[CST]] : f32) outs(%[[ALLOC]] : memref<128xf32>)
// CHECK: memref.copy
// CHECK: bufferization.to_tensor %[[ALLOC]]

module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @load_other_scalar_zero(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32) attributes {noinline = false} {
    %c128_i32 = arith.constant 128 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128xf32>
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %3 = tt.splat %1 : i32 -> tensor<128xi32>
    %4 = arith.addi %3, %2 : tensor<128xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<128xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<128xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %9 = tt.load %8, %6, %cst : tensor<128x!tt.ptr<f32>>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %11 = tt.addptr %10, %4 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %11, %9, %6 : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Case 4: discrete mask (offs % 2 == 0) + other = prior tensor.
// discrete-mask-access-conversion already rewrites to full load + arith.select.
// CHECK-LABEL: func.func @load_other_tensor_discrete_mask
// CHECK: memref.copy
// CHECK: %[[OTHER:.*]] = bufferization.to_tensor
// CHECK: memref.copy
// CHECK: %[[LOADED:.*]] = bufferization.to_tensor
// CHECK: arith.select {{.*}}, %[[LOADED]], %[[OTHER]]
// CHECK-NOT: tensor.insert_slice

module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @load_other_tensor_discrete_mask(%in_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %other_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %N: i32) attributes {noinline = false} {
    %c128_i32 = arith.constant 128 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %3 = tt.splat %1 : i32 -> tensor<128xi32>
    %4 = arith.addi %3, %2 : tensor<128xi32>
    %5 = tt.splat %c2_i32 : i32 -> tensor<128xi32>
    %6 = arith.remsi %4, %5 : tensor<128xi32>
    %7 = tt.splat %c0_i32 : i32 -> tensor<128xi32>
    %8 = arith.cmpi eq, %6, %7 : tensor<128xi32>
    %9 = tt.splat %other_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %10 = tt.addptr %9, %4 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %11 = tt.load %10 : tensor<128x!tt.ptr<f32>>
    %12 = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %13 = tt.addptr %12, %4 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %14 = tt.load %13, %8, %11 : tensor<128x!tt.ptr<f32>>
    %15 = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %16 = tt.addptr %15, %4 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %16, %14 : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}

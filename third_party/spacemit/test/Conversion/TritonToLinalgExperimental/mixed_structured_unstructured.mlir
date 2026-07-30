// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s
//
// This test verifies that we correctly handle the IR when there is a mix of
// structured and unstructured accesses in loops.
// Generated from the following dummy triton program:
//
// @triton.jit
// def test_mixed(in_ptr_1, in_ptr_2, out_ptr, stride_m, stride_n):
//     offs_am = tl.arange(0, 2)
//     offs_an = tl.arange(0, 2)
//     structured_ptrs = in_ptr_1 + (offs_am[:, None] * stride_m +
//                         offs_an[None, :] * stride_n)
//     unstructured_offset_ptrs = in_ptr_1 + ((offs_am[:, None] // 4) * stride_m +
//                         (offs_an[None, :] % 3 + 2) * stride_n)
//     if stride_m == 10:
//         base_ptr = in_ptr_1
//     else:
//         base_ptr = in_ptr_2
//     unstructured_raw_ptrs = base_ptr + ((offs_am[:, None] // 4) * stride_m +
//                         (offs_an[None, :] % 3 + 2) * stride_n)
//     offs_cm = tl.arange(0, 2)
//     offs_cn = tl.arange(0, 2)
//     c_ptrs = out_ptr + stride_m * offs_cm[:, None] + stride_n * offs_cn[
//         None, :]
//     for i in range(0, 2):
//         structured_ptrs += stride_n
//         unstructured_offset_ptrs += stride_n
//         unstructured_raw_ptrs += stride_n
//         for j in range(0, 2):
//             t1 = tl.load(structured_ptrs)
//             t2 = tl.load(unstructured_offset_ptrs)
//             t3 = tl.load(unstructured_raw_ptrs)
//             tl.store(c_ptrs, t1)
//             c_ptrs += stride_n
//             tl.store(c_ptrs, t2)
//             c_ptrs += stride_n
//             tl.store(c_ptrs, t3)
//             structured_ptrs += stride_n
//             unstructured_offset_ptrs += stride_n
//             unstructured_raw_ptrs += stride_n
//         structured_ptrs += stride_n
//         unstructured_offset_ptrs += stride_n
//         unstructured_raw_ptrs += stride_n
// def test():
//     src = triton.compiler.ASTSource(
//         fn=test_mixed,
//         signature={"in_ptr_1": "*fp32", "in_ptr_2": "*fp32", "out_ptr": "*fp32", "stride_m": "i32", "stride_n": "i32"},
//     )
//     ret = triton.compile(
//         src,
//     )
//     print(ret.asm["ttir"])
//     print('Pass')

module {
  tt.func public @test_mixed(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<2> : tensor<1x2xi32>
    %cst_0 = arith.constant dense<3> : tensor<1x2xi32>
    %cst_1 = arith.constant dense<4> : tensor<2x1xi32>
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %2 = tt.splat %arg2 : i32 -> tensor<2x1xi32>
    %3 = arith.muli %1, %2 : tensor<2x1xi32>
    %4 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<1x2xi32>
    %6 = arith.muli %4, %5 : tensor<1x2xi32>
    %7 = tt.broadcast %3 : tensor<2x1xi32> -> tensor<2x2xi32>
    %8 = tt.broadcast %6 : tensor<1x2xi32> -> tensor<2x2xi32>
    %9 = arith.addi %7, %8 : tensor<2x2xi32>
    %10 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
    %11 = tt.addptr %10, %9 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %12 = arith.divsi %1, %cst_1 : tensor<2x1xi32>
    %13 = arith.muli %12, %2 : tensor<2x1xi32>
    %14 = arith.remsi %4, %cst_0 : tensor<1x2xi32>
    %15 = arith.addi %14, %cst : tensor<1x2xi32>
    %16 = arith.muli %15, %5 : tensor<1x2xi32>
    %17 = tt.broadcast %13 : tensor<2x1xi32> -> tensor<2x2xi32>
    %18 = tt.broadcast %16 : tensor<1x2xi32> -> tensor<2x2xi32>
    %19 = arith.addi %17, %18 : tensor<2x2xi32>
    %20 = tt.addptr %10, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %21 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %22 = tt.addptr %21, %3 : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
    %23 = tt.broadcast %22 : tensor<2x1x!tt.ptr<f32>> -> tensor<2x2x!tt.ptr<f32>>
    %24 = tt.addptr %23, %8 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %25:3 = scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg5 = %11, %arg6 = %20, %arg7 = %24) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
      %26 = tt.splat %arg3 : i32 -> tensor<2x2xi32>
      %27 = tt.addptr %arg5, %26 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %28 = tt.addptr %arg6, %26 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %29:3 = scf.for %arg8 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg9 = %arg7, %arg10 = %27, %arg11 = %28) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
        %32 = tt.load %arg10 : tensor<2x2x!tt.ptr<f32>>
        %33 = tt.load %arg11 : tensor<2x2x!tt.ptr<f32>>
        tt.store %arg9, %32 : tensor<2x2x!tt.ptr<f32>>
        %34 = tt.addptr %arg9, %26 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        tt.store %34, %33 : tensor<2x2x!tt.ptr<f32>>
        %35 = tt.addptr %arg10, %26 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        %36 = tt.addptr %arg11, %26 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        scf.yield %34, %35, %36 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
      }
      %30 = tt.addptr %29#1, %26 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %31 = tt.addptr %29#2, %26 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      scf.yield %30, %31, %29#0 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
    }
    tt.return
  }
}

// CHECK-NOT: unrealized_conversion_cast

// ---

module {
  tt.func public @test_mixed(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32
    %cst = arith.constant dense<2> : tensor<1x2xi32>
    %cst_0 = arith.constant dense<3> : tensor<1x2xi32>
    %cst_1 = arith.constant dense<4> : tensor<2x1xi32>
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %2 = tt.splat %arg3 : i32 -> tensor<2x1xi32>
    %3 = arith.muli %1, %2 : tensor<2x1xi32>
    %4 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
    %5 = tt.splat %arg4 : i32 -> tensor<1x2xi32>
    %6 = arith.muli %4, %5 : tensor<1x2xi32>
    %7 = tt.broadcast %3 : tensor<2x1xi32> -> tensor<2x2xi32>
    %8 = tt.broadcast %6 : tensor<1x2xi32> -> tensor<2x2xi32>
    %9 = arith.addi %7, %8 : tensor<2x2xi32>
    %10 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
    %11 = tt.addptr %10, %9 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %12 = arith.divsi %1, %cst_1 : tensor<2x1xi32>
    %13 = arith.muli %12, %2 : tensor<2x1xi32>
    %14 = arith.remsi %4, %cst_0 : tensor<1x2xi32>
    %15 = arith.addi %14, %cst : tensor<1x2xi32>
    %16 = arith.muli %15, %5 : tensor<1x2xi32>
    %17 = tt.broadcast %13 : tensor<2x1xi32> -> tensor<2x2xi32>
    %18 = tt.broadcast %16 : tensor<1x2xi32> -> tensor<2x2xi32>
    %19 = arith.addi %17, %18 : tensor<2x2xi32>
    %20 = tt.addptr %10, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %21 = arith.cmpi eq, %arg3, %c10_i32 : i32
    %22 = arith.select %21, %arg0, %arg1 : !tt.ptr<f32>
    %23 = tt.splat %22 : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
    %24 = tt.addptr %23, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %25 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %26 = tt.addptr %25, %3 : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
    %27 = tt.broadcast %26 : tensor<2x1x!tt.ptr<f32>> -> tensor<2x2x!tt.ptr<f32>>
    %28 = tt.addptr %27, %8 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %29:4 = scf.for %arg5 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg6 = %11, %arg7 = %20, %arg8 = %24, %arg9 = %28) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
      %30 = tt.splat %arg4 : i32 -> tensor<2x2xi32>
      %31 = tt.addptr %arg6, %30 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %32 = tt.addptr %arg7, %30 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %33 = tt.addptr %arg8, %30 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %34:4 = scf.for %arg10 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg11 = %arg9, %arg12 = %31, %arg13 = %32, %arg14 = %33) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
        %38 = tt.load %arg12 : tensor<2x2x!tt.ptr<f32>>
        %39 = tt.load %arg13 : tensor<2x2x!tt.ptr<f32>>
        %40 = tt.load %arg14 : tensor<2x2x!tt.ptr<f32>>
        tt.store %arg11, %38 : tensor<2x2x!tt.ptr<f32>>
        %41 = tt.addptr %arg11, %30 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        tt.store %41, %39 : tensor<2x2x!tt.ptr<f32>>
        %42 = tt.addptr %41, %30 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        tt.store %42, %40 : tensor<2x2x!tt.ptr<f32>>
        %43 = tt.addptr %arg12, %30 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        %44 = tt.addptr %arg13, %30 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        %45 = tt.addptr %arg14, %30 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        scf.yield %42, %43, %44, %45 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
      }
      %35 = tt.addptr %34#1, %30 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %36 = tt.addptr %34#2, %30 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %37 = tt.addptr %34#3, %30 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      scf.yield %35, %36, %37, %34#0 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
    }
    tt.return
  }
}

// CHECK-NOT: unrealized_conversion_cast

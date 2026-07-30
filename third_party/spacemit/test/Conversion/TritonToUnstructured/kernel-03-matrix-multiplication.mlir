// RUN: triton-shared-opt --triton-to-unstructured %s | FileCheck %s

module {
  tt.func public @matmul_kernel_0123456789101112131415(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<bf16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) {
    %c63_i32 = arith.constant 63 : i32
    %c255_i32 = arith.constant 255 : i32
    %c127_i32 = arith.constant 127 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg4, %c255_i32 : i32
    %4 = arith.divsi %3, %c256_i32 : i32
    %5 = arith.addi %arg5, %c63_i32 : i32
    %6 = arith.divsi %5, %c64_i32 : i32
    %7 = arith.muli %4, %c8_i32 : i32
    %8 = arith.divsi %0, %7 : i32
    %9 = arith.muli %8, %c8_i32 : i32
    %10 = arith.subi %2, %9 : i32
    %11 = arith.cmpi slt, %10, %c8_i32 : i32
    %12 = arith.select %11, %10, %c8_i32 : i32
    %13 = arith.remsi %0, %12 : i32
    %14 = arith.addi %9, %13 : i32
    %15 = arith.remsi %0, %7 : i32
    %16 = arith.divsi %15, %12 : i32
    %17 = arith.muli %14, %c128_i32 : i32
    %18 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %19 = tt.splat %17 : i32 -> tensor<128xi32>
    %20 = arith.addi %19, %18 : tensor<128xi32>
    %21 = arith.muli %16, %c256_i32 : i32
    %22 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %23 = tt.splat %21 : i32 -> tensor<256xi32>
    %24 = arith.addi %23, %22 : tensor<256xi32>
    %25 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %26 = tt.expand_dims %20 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %27 = tt.splat %arg6 : i32 -> tensor<128x1xi32>
    %28 = arith.muli %26, %27 : tensor<128x1xi32>
    %29 = tt.expand_dims %25 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %30 = tt.splat %arg7 : i32 -> tensor<1x64xi32>
    %31 = arith.muli %29, %30 : tensor<1x64xi32>
    %32 = tt.broadcast %28 : tensor<128x1xi32> -> tensor<128x64xi32>
    %33 = tt.broadcast %31 : tensor<1x64xi32> -> tensor<128x64xi32>
    %34 = arith.addi %32, %33 : tensor<128x64xi32>
    %35 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>>
    %36 = tt.addptr %35, %34 : tensor<128x64x!tt.ptr<bf16>>, tensor<128x64xi32>
    %37 = tt.expand_dims %25 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %38 = tt.splat %arg8 : i32 -> tensor<64x1xi32>
    %39 = arith.muli %37, %38 : tensor<64x1xi32>
    %40 = tt.expand_dims %24 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %41 = tt.splat %arg9 : i32 -> tensor<1x256xi32>
    %42 = arith.muli %40, %41 : tensor<1x256xi32>
    %43 = tt.broadcast %39 : tensor<64x1xi32> -> tensor<64x256xi32>
    %44 = tt.broadcast %42 : tensor<1x256xi32> -> tensor<64x256xi32>
    %45 = arith.addi %43, %44 : tensor<64x256xi32>
    %46 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<64x256x!tt.ptr<bf16>>
    %47 = tt.addptr %46, %45 : tensor<64x256x!tt.ptr<bf16>>, tensor<64x256xi32>
    %48 = tt.splat %cst : f32 -> tensor<128x256xf32>
    %49 = arith.muli %arg7, %c64_i32 : i32
    %50 = tt.splat %49 : i32 -> tensor<128x64xi32>
    %51 = arith.muli %arg8, %c64_i32 : i32
    %52 = tt.splat %51 : i32 -> tensor<64x256xi32>
    %53:3 = scf.for %arg12 = %c0_i32 to %6 step %c1_i32 iter_args(%arg13 = %48, %arg14 = %36, %arg15 = %47) -> (tensor<128x256xf32>, tensor<128x64x!tt.ptr<bf16>>, tensor<64x256x!tt.ptr<bf16>>)  : i32 {
      %71 = tt.load %arg14 : tensor<128x64x!tt.ptr<bf16>>
      %72 = tt.load %arg15 : tensor<64x256x!tt.ptr<bf16>>
      %73 = tt.dot %71, %72, %48 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xbf16> * tensor<64x256xbf16> -> tensor<128x256xf32>
      %74 = arith.addf %arg13, %73 : tensor<128x256xf32>
      %75 = tt.addptr %arg14, %50 : tensor<128x64x!tt.ptr<bf16>>, tensor<128x64xi32>
      %76 = tt.addptr %arg15, %52 : tensor<64x256x!tt.ptr<bf16>>, tensor<64x256xi32>
      scf.yield %74, %75, %76 : tensor<128x256xf32>, tensor<128x64x!tt.ptr<bf16>>, tensor<64x256x!tt.ptr<bf16>>
    }
    %54 = arith.truncf %53#0 : tensor<128x256xf32> to tensor<128x256xbf16>
    %55 = tt.splat %arg10 : i32 -> tensor<128x1xi32>
    %56 = arith.muli %55, %26 : tensor<128x1xi32>
    %57 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
    %58 = tt.addptr %57, %56 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
    %59 = tt.splat %arg11 : i32 -> tensor<1x256xi32>
    %60 = arith.muli %59, %40 : tensor<1x256xi32>
    %61 = tt.broadcast %58 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x256x!tt.ptr<bf16>>
    %62 = tt.broadcast %60 : tensor<1x256xi32> -> tensor<128x256xi32>
    %63 = tt.addptr %61, %62 : tensor<128x256x!tt.ptr<bf16>>, tensor<128x256xi32>
    %64 = tt.splat %arg3 : i32 -> tensor<128x1xi32>
    %65 = arith.cmpi slt, %26, %64 : tensor<128x1xi32>
    %66 = tt.splat %arg4 : i32 -> tensor<1x256xi32>
    %67 = arith.cmpi slt, %40, %66 : tensor<1x256xi32>
    %68 = tt.broadcast %65 : tensor<128x1xi1> -> tensor<128x256xi1>
    %69 = tt.broadcast %67 : tensor<1x256xi1> -> tensor<128x256xi1>
    %70 = arith.andi %68, %69 : tensor<128x256xi1>
    tt.store %63, %54, %70 : tensor<128x256x!tt.ptr<bf16>>
    tt.return
  }
}

// CHECK-NOT: tt.addptr
// CHECK-NOT: tt.load
// CHECK-NOT: tt.store

// CHECK-COUNT-2: tts.gather %arg{{[0-9]+}}
// CHECK-NOT: tts.gather %arg{{[0-9]+}}
// CHECK-COUNT-1: tts.scatter {{.+}} into %arg{{[0-9]+}}
// CHECK-NOT:    tts.scatter {{.+}} into %arg{{[0-9]+}}

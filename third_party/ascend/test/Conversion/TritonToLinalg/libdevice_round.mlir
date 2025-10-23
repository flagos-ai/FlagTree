// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s
module {
tt.func public @test_round(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 :i32
    %c32_i32 = arith.constant 32 :i32
    %c0_i32 = arith.constant 0 :i32
    %c64_i32 = arith.constant 64 :i32
    %c2048_i32 = arith.constant 2048 :i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c2048_i32 :i32
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %4 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 : i32 {
        %5 = arith.muli %arg2, %c64_i32 : i32
        %6 = arith.addi %1, %5 : i32
        %7 = tt.splat %6 : i32 -> tensor<64xi32>
        %8 = arith.addi %7, %2 : tensor<64xi32>
        %9 = tt.addptr %3, %8 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
        %10 = tt.load %9 : tensor<64x!tt.ptr<f32>>
        %11 = tt.extern_elementwise %10 {libname ="", libpath= "", pure = true, symbol= "__hmf_roundf"} : (tensor<64xf32>) -> tensor<64xf32>
        %12 = tt.addptr %4, %8 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
        tt.store %12, %11 : tensor<64x!tt.ptr<f32>>
    }
    tt.return
}
}


//CHECK: func.func private @__hmf_roundf(f32) -> f32 attributes {llvm.readnone}
//CHECK: %[[RESULT:.*]] = linalg.map { func.call {callee = @__hmf_roundf} } ins(%[[SOURCE:.*]] : tensor<64xf32>) outs(%[[SOURCE]] : tensor<64xf32>)
//CHECK: bufferization.materialize_in_destination %[[RESULT]] in writable %[[DST:.*]] : (tensor<64xf32>, memref<64xf32, strided<[1], offset: ?>>) -> ()
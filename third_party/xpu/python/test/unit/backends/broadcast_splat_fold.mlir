#row = #triton_xpu.cluster<{sizePerCore = [1, 1], coresPerGroup = [1, 8], groupsPerCluster = [8, 1], order = [0, 1]}>
#vec = #triton_xpu.cluster<{sizePerCore = [1, 2], coresPerGroup = [1, 8], groupsPerCluster = [8, 1], order = [0, 1]}>

module attributes {triton_xpu.target = "xpu:3", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 8 : i32} {
  tt.func public @broadcast(%out: tensor<8x256x!tt.ptr<f32, 0>, #vec>) {
    %zero = arith.constant dense<0.0> : tensor<8x1xf32, #row>
    %result = triton_xpu.broadcast %zero : tensor<8x1xf32, #row> -> tensor<8x16xvector<16xf32>, #vec>
    "triton_xpu.store"(%out, %result) <{bf16Tofp32Unordered = false, dtype = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 0>, syncMode = 1 : i32, tensorColSize = -1 : si32}> : (tensor<8x256x!tt.ptr<f32, 0>, #vec>, tensor<8x16xvector<16xf32>, #vec>) -> ()
    tt.return
  }
}

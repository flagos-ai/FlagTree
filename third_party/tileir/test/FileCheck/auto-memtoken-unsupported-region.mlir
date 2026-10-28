// RUN: triton-cuda-tile-opt %s -split-input-file --pass-pipeline="builtin.module(cuda_tile.module(cuda_tile.entry(auto-gen-memory-token{autogen-alias-memtoken=true})))" -verify-diagnostics

// A public but unhandled region containing memory effects must fail explicitly.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_unhandled_region(%arg0: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_1_f32 = cuda_tile.constant <f32: 1.000000e+00> : tile<f32>
      // Outer store so the hazard check upstream decides to transform
      // the function (otherwise the pass bails before reaching the
      // unsupported-region diagnostic path).
      %s0 = cuda_tile.store_ptr_tko weak %arg0, %cst_1_f32 : tile<ptr<f32>>, tile<f32> -> token
      // expected-error@+1 {{cannot tokenize memory ops inside unsupported region op}}
      scf.execute_region {
        %t = cuda_tile.store_ptr_tko weak %arg0, %cst_1_f32 : tile<ptr<f32>>, tile<f32> -> token
        scf.yield
      }
      cuda_tile.return
    }
  }
}

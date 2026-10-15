// REQUIRES: flagtree-common-ir
// RUN: triton-opt %s --convert-common-ir-to-ttgir='enable-async-copy=true' | FileCheck %s --check-prefix=ASYNC --implicit-check-not=tile. --implicit-check-not=tle. --implicit-check-not=builtin.unrealized_conversion_cast
// RUN: triton-opt %s --convert-common-ir-to-ttgir='enable-async-copy=true' --convert-triton-to-tritongpu='target=cuda:90 num-warps=4' | FileCheck %s --check-prefix=LAYOUT --implicit-check-not=builtin.unrealized_conversion_cast
// RUN: triton-opt %s --convert-common-ir-to-ttgir='enable-async-copy=false' | FileCheck %s --check-prefix=SYNC --implicit-check-not=ttg.async

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

module {
  tt.func public @copy(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %index = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %s = tt.splat %src : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %d = tt.splat %dst : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %sp = tt.addptr %s, %index : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %dp = tt.addptr %d, %index : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %buf = tile.alloc {space = 7 : i64, tle.gpu_layout = #shared} : <[128], f32, shared>
    tile.copy %sp -> %buf : tensor<128x!tt.ptr<f32>>, !tile.buf<[128], f32, shared>
    tile.copy %buf -> %dp : !tile.buf<[128], f32, shared>, tensor<128x!tt.ptr<f32>>
    tt.return
  }
}

// ASYNC: %[[BUF:.*]] = ttg.local_alloc
// ASYNC: %[[COPY:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[BUF]]
// ASYNC: %[[COMMIT:.*]] = ttg.async_commit_group tokens %[[COPY]]
// ASYNC: ttg.async_wait %[[COMMIT]] {num = 0 : i32}
// ASYNC: %[[VALUE:.*]] = ttg.local_load %[[BUF]]
// ASYNC: tt.store %{{.*}}, %[[VALUE]]
// LAYOUT: ttg.async_copy_global_to_local {{.*}} : tensor<128x!tt.ptr<f32>, #blocked>
// SYNC: tt.load
// SYNC: ttg.local_store
// SYNC: ttg.local_load
// SYNC: tt.store

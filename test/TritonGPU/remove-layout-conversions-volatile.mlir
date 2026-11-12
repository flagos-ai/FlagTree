// RUN: triton-opt %s -tritongpu-remove-layout-conversions | FileCheck %s --check-prefixes=COMMON,OFF
// RUN: triton-opt %s -tritongpu-remove-layout-conversions="enable-rlc-enhance=true enable-cost-based-resolution=false" | FileCheck %s --check-prefixes=COMMON,ON
// RUN: triton-opt %s -tritongpu-remove-layout-conversions="enable-rlc-enhance=true enable-cost-based-resolution=false enable-store-layout-rematerialization=false" | FileCheck %s --check-prefixes=COMMON,OFF

// Isolate the Phase 3 duplicable-load exception from propagation/component
// solving. There are 128 elements and 128 threads: unlike the upstream
// 16-element example in combine.mlir, canBeRemat considers these loads expensive.
// The MMA slice preserves the 1-D memory order and does not narrow contiguity.
// One i32 per thread is below the 128-bit bound. Both layouts have live uses,
// and the conversion is a final boundary, so the non-volatile control should
// duplicate the load only when store-layout rematerialization is enabled.

// COMMON-DAG: #[[$SRC:blocked[0-9]*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// COMMON-DAG: #[[$MMA:mma[0-9]*]] = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#slice = #ttg.slice<{dim = 0, parent = #mma}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:80"} {
  // COMMON-LABEL: @no_duplicate_volatile_vector
  // COMMON-NOT: tt.load
  // COMMON: %[[LOAD:.+]] = tt.load {{.*}}{isVolatile = true} : tensor<128x!tt.ptr<i32>, #[[$SRC]]>
  // COMMON-NOT: tt.load
  // COMMON: %[[CONVERTED:.+]] = ttg.convert_layout %[[LOAD]] : tensor<128xi32, #[[$SRC]]> -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #[[$MMA]]}>>
  // COMMON-NOT: tt.load
  // COMMON: tt.return %[[CONVERTED]], %[[LOAD]]
  tt.func public @no_duplicate_volatile_vector(%arg: !tt.ptr<i32> {tt.divisibility = 16 : i32}) -> (tensor<128xi32, #slice>, tensor<128xi32, #blocked>) {
    %base = tt.splat %arg : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #blocked>
    %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #blocked>
    %ptr = tt.addptr %base, %range : tensor<128x!tt.ptr<i32>, #blocked>, tensor<128xi32, #blocked>
    %value = tt.load %ptr {isVolatile = true} : tensor<128x!tt.ptr<i32>, #blocked>
    %converted = ttg.convert_layout %value : tensor<128xi32, #blocked> -> tensor<128xi32, #slice>
    tt.return %converted, %value : tensor<128xi32, #slice>, tensor<128xi32, #blocked>
  }

  // COMMON-LABEL: @duplicate_nonvolatile_vector
  // OFF-NOT: tt.load
  // OFF: %[[LOAD:.+]] = tt.load {{.*}} : tensor<128x!tt.ptr<i32>, #[[$SRC]]>
  // OFF-NOT: tt.load
  // OFF: %[[CONVERTED:.+]] = ttg.convert_layout %[[LOAD]]
  // OFF-NOT: tt.load
  // OFF: tt.return %[[CONVERTED]], %[[LOAD]]
  // The slice rewrite inserts clones before the original operations.
  // ON-NOT: tt.load
  // ON-NOT: ttg.convert_layout
  // ON: %[[DUPLICATE:.+]] = tt.load {{.*}} : tensor<128x!tt.ptr<i32>, #ttg.slice<{dim = 0, parent = #[[$MMA]]}>>
  // ON-NOT: tt.load
  // ON-NOT: ttg.convert_layout
  // ON: %[[LOAD:.+]] = tt.load {{.*}} : tensor<128x!tt.ptr<i32>, #[[$SRC]]>
  // ON-NOT: tt.load
  // ON-NOT: ttg.convert_layout
  // ON: tt.return %[[DUPLICATE]], %[[LOAD]]
  tt.func public @duplicate_nonvolatile_vector(%arg: !tt.ptr<i32> {tt.divisibility = 16 : i32}) -> (tensor<128xi32, #slice>, tensor<128xi32, #blocked>) {
    %base = tt.splat %arg : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #blocked>
    %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #blocked>
    %ptr = tt.addptr %base, %range : tensor<128x!tt.ptr<i32>, #blocked>, tensor<128xi32, #blocked>
    %value = tt.load %ptr : tensor<128x!tt.ptr<i32>, #blocked>
    %converted = ttg.convert_layout %value : tensor<128xi32, #blocked> -> tensor<128xi32, #slice>
    tt.return %converted, %value : tensor<128xi32, #slice>, tensor<128xi32, #blocked>
  }
}

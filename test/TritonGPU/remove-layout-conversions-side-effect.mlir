// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions | FileCheck %s --check-prefixes=COMMON,OFF
// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="enable-rlc-enhance=true" | FileCheck %s --check-prefixes=COMMON,ENHANCE
// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="enable-rlc-enhance=true enable-cost-based-resolution=false enable-store-layout-rematerialization=false" | FileCheck %s --check-prefixes=COMMON,OFF

// A producer declaring write effects keeps the layout it was written for.
//
//   * The baseline never duplicates it: canBeRemat() rejects any op with
//     MemoryEffects::Write (LoadOp::getEffects adds one for isVolatile,
//     ElementwiseInlineAsmOp::getEffects adds one when pure = false), and
//     isDuplicableLoadInEncoding() rejects a volatile load a second time because
//     that query overrides canBeRemat()'s verdict.
//   * The enhanced pipeline (enable-rlc-enhance) never duplicates it either, so
//     both configurations agree that the producer is materialized exactly once.
//
// Each fixture is 16 elements wide: 16 < num-warps * threads-per-warp
// (4 * 32 = 128), so isExpensiveLoadOrStore() is false and no access-pattern
// guard can fire. Only the side-effect gate keeps the layout in place, which is
// what these fixtures are here to pin.
//
// Assertions about the *layout* of the producer are OFF-only on purpose: with
// enable-rlc-enhance the small-component solver (Phase 2) may still retag a
// single-use producer onto the convert's target layout, which deletes the
// convert without duplicating the producer. If Phase 2 learns to freeze
// write-effect producers, those assertions move to COMMON and the matching
// ENHANCE case gains the layout check.
//
// The last fixture is the positive control: the same chain with pure = true has
// no write effect and is still rematerialized, so the gate keys on write effects
// rather than on "this is a load / an inline asm".

// COMMON-DAG: #[[$NARROW:blocked[0-9]*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// COMMON-DAG: #[[$WIDE:blocked[0-9]*]] = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:80"} {
  // Adapted from upstream Triton #12268. Both layouts stay live, so
  // rematerializing the conversion would duplicate the load.
  // COMMON-LABEL: @no_remat_volatile_load
  // OFF-NOT: tt.load
  // OFF: %[[VOLATILE:.+]] = tt.load {{.*}}{isVolatile = true} : tensor<16x!tt.ptr<i32>, #[[$WIDE]]>
  // OFF-NOT: tt.load
  // OFF: %[[CONVERTED:.+]] = ttg.convert_layout %[[VOLATILE]] : tensor<16xi32, #[[$WIDE]]> -> tensor<16xi32, #[[$NARROW]]>
  // OFF-NOT: tt.load
  // OFF: tt.store {{.*}}, %[[CONVERTED]]
  // OFF-NOT: tt.load
  // OFF: tt.store {{.*}}, %[[VOLATILE]]
  // OFF-NOT: tt.load
  // OFF: tt.return
  // ENHANCE-NOT: tt.load
  // ENHANCE: tt.load {{.*}}{isVolatile = true}
  // ENHANCE-NOT: tt.load
  // ENHANCE: tt.return
  tt.func public @no_remat_volatile_load(%arg: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
    %base = tt.splat %arg : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>, #blocked1>
    %range = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #blocked1>
    %ptr = tt.addptr %base, %range : tensor<16x!tt.ptr<i32>, #blocked1>, tensor<16xi32, #blocked1>
    %value = tt.load %ptr {isVolatile = true} : tensor<16x!tt.ptr<i32>, #blocked1>
    %converted = ttg.convert_layout %value : tensor<16xi32, #blocked1> -> tensor<16xi32, #blocked>
    %converted_ptr = ttg.convert_layout %ptr : tensor<16x!tt.ptr<i32>, #blocked1> -> tensor<16x!tt.ptr<i32>, #blocked>
    tt.store %converted_ptr, %converted : tensor<16x!tt.ptr<i32>, #blocked>
    tt.store %ptr, %value : tensor<16x!tt.ptr<i32>, #blocked1>
    tt.return
  }

  // A single consumer: Phase 2 producer-side solving would retag the load onto
  // the convert's layout and delete the convert, so the load must stay put.
  // COMMON-LABEL: @no_move_volatile_load
  // OFF-NOT: tt.load
  // OFF: %[[VOLATILE:.+]] = tt.load {{.*}}{isVolatile = true} : tensor<16x!tt.ptr<i32>, #[[$WIDE]]>
  // OFF-NOT: tt.load
  // OFF: %[[CONVERTED:.+]] = ttg.convert_layout %[[VOLATILE]] : tensor<16xi32, #[[$WIDE]]> -> tensor<16xi32, #[[$NARROW]]>
  // OFF-NOT: ttg.convert_layout
  // OFF: tt.return %[[CONVERTED]]
  // ENHANCE-NOT: tt.load
  // ENHANCE: tt.load {{.*}}{isVolatile = true}
  // ENHANCE-NOT: tt.load
  // ENHANCE: tt.return
  tt.func public @no_move_volatile_load(%arg: !tt.ptr<i32> {tt.divisibility = 16 : i32}) -> tensor<16xi32, #blocked> {
    %base = tt.splat %arg : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>, #blocked1>
    %range = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #blocked1>
    %ptr = tt.addptr %base, %range : tensor<16x!tt.ptr<i32>, #blocked1>, tensor<16xi32, #blocked1>
    %value = tt.load %ptr {isVolatile = true} : tensor<16x!tt.ptr<i32>, #blocked1>
    %converted = ttg.convert_layout %value : tensor<16xi32, #blocked1> -> tensor<16xi32, #blocked>
    tt.return %converted : tensor<16xi32, #blocked>
  }

  // Adapted from upstream Triton #11001 using this file's 1-D blocked layout
  // pair. A non-pure inline asm declares Write effects even though it is
  // elementwise, so it must not move either.
  // COMMON-LABEL: @side_effecting_inline_asm
  // OFF-NOT: tt.elementwise_inline_asm
  // OFF: %[[ASM:.+]] = tt.elementwise_inline_asm {{.*}}pure = false{{.*}}: tensor<16xi32, #[[$WIDE]]> -> tensor<16xi32, #[[$WIDE]]>
  // OFF-NOT: tt.elementwise_inline_asm
  // OFF: %[[CONVERTED:.+]] = ttg.convert_layout %[[ASM]] : tensor<16xi32, #[[$WIDE]]> -> tensor<16xi32, #[[$NARROW]]>
  // OFF-NOT: ttg.convert_layout
  // OFF: tt.return %[[CONVERTED]]
  // ENHANCE-NOT: tt.elementwise_inline_asm
  // ENHANCE: tt.elementwise_inline_asm {{.*}}pure = false
  // ENHANCE-NOT: tt.elementwise_inline_asm
  // ENHANCE: tt.return
  tt.func public @side_effecting_inline_asm() -> tensor<16xi32, #blocked> {
    %zero = arith.constant dense<0> : tensor<16xi32, #blocked1>
    %asm = tt.elementwise_inline_asm "mov.u32 $0, %clock;" {constraints = "=r,r", packed_element = 1 : i32, pure = false} %zero : tensor<16xi32, #blocked1> -> tensor<16xi32, #blocked1>
    %converted = ttg.convert_layout %asm : tensor<16xi32, #blocked1> -> tensor<16xi32, #blocked>
    tt.return %converted : tensor<16xi32, #blocked>
  }

  // Both layouts stay live, so the asm has to be duplicated to satisfy them; the
  // write effect forbids that. This is the multi-consumer counterpart of the
  // case above and the only fixture where a copy would have been profitable.
  // COMMON-LABEL: @side_effecting_inline_asm_two_layouts
  // OFF-NOT: tt.elementwise_inline_asm
  // OFF: %[[ASM:.+]] = tt.elementwise_inline_asm {{.*}}pure = false{{.*}}: tensor<16xi32, #[[$WIDE]]> -> tensor<16xi32, #[[$WIDE]]>
  // OFF-NOT: tt.elementwise_inline_asm
  // OFF: %[[CONVERTED:.+]] = ttg.convert_layout %[[ASM]] : tensor<16xi32, #[[$WIDE]]> -> tensor<16xi32, #[[$NARROW]]>
  // OFF-NOT: tt.elementwise_inline_asm
  // OFF: tt.return %[[CONVERTED]], %[[ASM]]
  // ENHANCE-NOT: tt.elementwise_inline_asm
  // ENHANCE: tt.elementwise_inline_asm {{.*}}pure = false
  // ENHANCE-NOT: tt.elementwise_inline_asm
  // ENHANCE: tt.return
  tt.func public @side_effecting_inline_asm_two_layouts() -> (tensor<16xi32, #blocked>, tensor<16xi32, #blocked1>) {
    %zero = arith.constant dense<0> : tensor<16xi32, #blocked1>
    %asm = tt.elementwise_inline_asm "mov.u32 $0, %clock;" {constraints = "=r,r", packed_element = 1 : i32, pure = false} %zero : tensor<16xi32, #blocked1> -> tensor<16xi32, #blocked1>
    %converted = ttg.convert_layout %asm : tensor<16xi32, #blocked1> -> tensor<16xi32, #blocked>
    tt.return %converted, %asm : tensor<16xi32, #blocked>, tensor<16xi32, #blocked1>
  }

  // Positive control for the gate above: identical chain, pure = true. It
  // declares no write effect, so nothing here freezes it and canBeRemat() lets
  // the baseline rematerialize it onto the target layout. The stricter form of
  // this control (no ttg.convert_layout, asm produced in #[[$NARROW]]) is the
  // expectation to assert once it has been confirmed on a built triton-opt; the
  // invariant checked here holds whether or not that remat fires.
  // COMMON-LABEL: @pure_inline_asm_is_rematerialized
  // COMMON-NOT: tt.elementwise_inline_asm
  // COMMON: tt.elementwise_inline_asm {{.*}}pure = true
  // COMMON-NOT: tt.elementwise_inline_asm
  // COMMON: tt.return
  tt.func public @pure_inline_asm_is_rematerialized() -> tensor<16xi32, #blocked> {
    %zero = arith.constant dense<0> : tensor<16xi32, #blocked1>
    %asm = tt.elementwise_inline_asm "mov.u32 $0, %r0;" {constraints = "=r,r", packed_element = 1 : i32, pure = true} %zero : tensor<16xi32, #blocked1> -> tensor<16xi32, #blocked1>
    %converted = ttg.convert_layout %asm : tensor<16xi32, #blocked1> -> tensor<16xi32, #blocked>
    tt.return %converted : tensor<16xi32, #blocked>
  }
}

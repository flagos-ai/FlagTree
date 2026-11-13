// RUN: triton-opt -split-input-file --ub-overflow-check %s | FileCheck %s

// ============================================================================
// Test 1: Safe — UB estimate under threshold, all marks preserved.
//   Two 128x128xf32 annot buffers (alignedSize=262144 bits each).
//   Wave UB = 2 x (262144 x 2) = 1048576 bits < 2031616 => SAFE.
// ============================================================================

func.func @safe_no_pruning() {
  %c0_i32 = arith.constant 0 : i32
  %c8_i32 = arith.constant 8 : i32
  %c1_i32 = arith.constant 1 : i32

  // CHECK-LABEL: func.func @safe_no_pruning

  scope.scope : () -> () {
    scf.for %i = %c0_i32 to %c8_i32 step %c1_i32 : i32 {
      // CHECK: memref.alloc() : memref<128x128xf32>
      // CHECK-NEXT: annotation.mark %{{.*}} {hivm.multi_buffer = 2 : i32} : memref<128x128xf32>
      %a1 = memref.alloc() : memref<128x128xf32>
      annotation.mark %a1 {hivm.multi_buffer = 2 : i32} : memref<128x128xf32>
      // CHECK: memref.alloc() : memref<128x128xf32>
      // CHECK-NEXT: annotation.mark %{{.*}} {hivm.multi_buffer = 2 : i32} : memref<128x128xf32>
      %a2 = memref.alloc() : memref<128x128xf32>
      annotation.mark %a2 {hivm.multi_buffer = 2 : i32} : memref<128x128xf32>
    }
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  return
}

// -----

// ============================================================================
// Test 2: Overflow — largest multi_buffer mark pruned.
//   One 256x256xf32 (alignedSize=1048576, expanded=2097152)
//   Two 64x64xf32 (alignedSize=65536 each, expanded=131072 each)
//   Wave UB = 2097152 + 2x131072 = 2359296 > 2031616 => OVERFLOW.
//   Prune largest (256x256): annot→131072x2=262144, unannot→1048576.
//   Wave UB = 1310720 < 2031616 => SAFE.
// ============================================================================

func.func @overflow_prune_largest() {
  %c0_i32 = arith.constant 0 : i32
  %c8_i32 = arith.constant 8 : i32
  %c1_i32 = arith.constant 1 : i32

  // CHECK-LABEL: func.func @overflow_prune_largest

  scope.scope : () -> () {
    scf.for %i = %c0_i32 to %c8_i32 step %c1_i32 : i32 {
      // 256x256 mark pruned: markOp still exists but multi_buffer attr removed.
      // CHECK: memref.alloc() : memref<256x256xf32>
      // CHECK-NOT: multi_buffer
      // CHECK: annotation.mark
      %huge = memref.alloc() : memref<256x256xf32>
      annotation.mark %huge {hivm.multi_buffer = 2 : i32} : memref<256x256xf32>

      // Small allocs retain marks.
      // CHECK: memref.alloc() : memref<64x64xf32>
      // CHECK-NEXT: annotation.mark %{{.*}} {hivm.multi_buffer = 2 : i32} : memref<64x64xf32>
      %small1 = memref.alloc() : memref<64x64xf32>
      annotation.mark %small1 {hivm.multi_buffer = 2 : i32} : memref<64x64xf32>

      // CHECK: memref.alloc() : memref<64x64xf32>
      // CHECK-NEXT: annotation.mark %{{.*}} {hivm.multi_buffer = 2 : i32} : memref<64x64xf32>
      %small2 = memref.alloc() : memref<64x64xf32>
      annotation.mark %small2 {hivm.multi_buffer = 2 : i32} : memref<64x64xf32>
    }
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  return
}

// -----

// ============================================================================
// Test 3: CUBE scope skipped (only VECTOR scopes are checked).
// ============================================================================

func.func @skip_cube_scope() {
  %c0_i32 = arith.constant 0 : i32
  %c8_i32 = arith.constant 8 : i32
  %c1_i32 = arith.constant 1 : i32

  // CHECK-LABEL: func.func @skip_cube_scope
  // CHECK: annotation.mark %{{.*}} {hivm.multi_buffer = 2 : i32} : memref<256x256xf32>

  scope.scope : () -> () {
    scf.for %i = %c0_i32 to %c8_i32 step %c1_i32 : i32 {
      %a1 = memref.alloc() : memref<256x256xf32>
      annotation.mark %a1 {hivm.multi_buffer = 2 : i32} : memref<256x256xf32>
    }
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
  return
}

// -----

// ============================================================================
// Test 4: Hint-protected mark survives pruning; gm_load_hint cleaned up.
//   A (hint, gm_load_hint + multi_buffer=2): 256x192xf32 (alignedSize=786432)
//   B (auto, multi_buffer=2): 256x192xf32 (alignedSize=786432)
//   Wave UB = 2 x 1572864 = 3145728 > 2031616 => OVERFLOW.
//   A is hint-protected, cannot prune. Prune B.
//   Wave UB = 1572864 < 2031616 => SAFE.
// ============================================================================

func.func @hint_protected_kept() {
  %c0_i32 = arith.constant 0 : i32
  %c8_i32 = arith.constant 8 : i32
  %c1_i32 = arith.constant 1 : i32

  // CHECK-LABEL: func.func @hint_protected_kept
  // CHECK-NOT: gm_load_hint

  scope.scope : () -> () {
    // Hint-protected (A): mark kept, gm_load_hint removed.
    scf.for %i = %c0_i32 to %c8_i32 step %c1_i32 : i32 {
      // CHECK: memref.alloc() : memref<256x192xf32>
      // CHECK-NEXT: annotation.mark %{{.*}} {hivm.multi_buffer = 2 : i32} : memref<256x192xf32>
      %a = memref.alloc() : memref<256x192xf32>
      annotation.mark %a {gm_load_hint, hivm.multi_buffer = 2 : i32} : memref<256x192xf32>
    }
    // Auto (B): pruned.
    scf.for %j = %c0_i32 to %c8_i32 step %c1_i32 : i32 {
      // CHECK: memref.alloc() : memref<256x192xf32>
      // CHECK-NEXT: annotation.mark %{{.*}} : memref<256x192xf32>
      %b = memref.alloc() : memref<256x192xf32>
      annotation.mark %b {hivm.multi_buffer = 2 : i32} : memref<256x192xf32>
    }
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  return
}


// -----

// ============================================================================
// Test 5: Hint protection survives multiple pruning rounds.
//   Hint: 256x144xf32, alignedSize=589824, expanded=1179648.
//   Auto: two 256x96xf32, alignedSize=393216, expanded=786432 each.
//   Initial=2752512; after one prune=2359296; after two=1966080.
// ============================================================================

func.func @multi_round_hint_protection() {
  // CHECK-LABEL: func.func @multi_round_hint_protection
  // CHECK-NOT: gm_load_hint

  scope.scope : () -> () {
    // CHECK: memref.alloc() : memref<256x144xf32>
    // CHECK-NEXT: annotation.mark %{{.*}} {hivm.multi_buffer = 2 : i32} : memref<256x144xf32>
    %hint = memref.alloc() : memref<256x144xf32>
    annotation.mark %hint {gm_load_hint, hivm.multi_buffer = 2 : i32} : memref<256x144xf32>

    // CHECK: memref.alloc() : memref<256x96xf32>
    // CHECK-NEXT: annotation.mark %{{.*}} : memref<256x96xf32>
    %auto1 = memref.alloc() : memref<256x96xf32>
    annotation.mark %auto1 {hivm.multi_buffer = 2 : i32} : memref<256x96xf32>

    // CHECK: memref.alloc() : memref<256x96xf32>
    // CHECK-NEXT: annotation.mark %{{.*}} : memref<256x96xf32>
    %auto2 = memref.alloc() : memref<256x96xf32>
    annotation.mark %auto2 {hivm.multi_buffer = 2 : i32} : memref<256x96xf32>
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  return
}

// -----

// ============================================================================
// Test 6: A blind tensor.empty contributes to UB and triggers pruning.
//   Alloc expanded=1572864; tensor=524288; total=2097152 > 2031616.
// ============================================================================

func.func @blind_tensor_triggers_pruning() {
  // CHECK-LABEL: func.func @blind_tensor_triggers_pruning

  scope.scope : () -> () {
    // CHECK: memref.alloc() : memref<256x192xf32>
    // CHECK-NEXT: annotation.mark %{{.*}} : memref<256x192xf32>
    %alloc = memref.alloc() : memref<256x192xf32>
    annotation.mark %alloc {hivm.multi_buffer = 2 : i32} : memref<256x192xf32>

    // CHECK: tensor.empty() : tensor<256x128xf32>
    %empty = tensor.empty() : tensor<256x128xf32>
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  return
}

// -----

// ============================================================================
// Test 7: A same-shape tensor.empty is still counted independently.
//   Alloc expanded=1572864; tensor=786432; total=2359296 > 2031616.
//   Removing the mark leaves 1572864 bits, which is safe.
// ============================================================================

func.func @same_shape_tensor_counted() {
  // CHECK-LABEL: func.func @same_shape_tensor_counted

  scope.scope : () -> () {
    // CHECK: memref.alloc() : memref<256x192xf32>
    // CHECK-NEXT: annotation.mark %{{.*}} : memref<256x192xf32>
    %alloc = memref.alloc() : memref<256x192xf32>
    annotation.mark %alloc {hivm.multi_buffer = 2 : i32} : memref<256x192xf32>

    // CHECK: tensor.empty() : tensor<256x192xf32>
    %empty = tensor.empty() : tensor<256x192xf32>
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  return
}

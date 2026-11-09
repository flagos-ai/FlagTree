// RUN: triton-opt --exp-subf-pattern %s | FileCheck %s

module {
  // ============================================
  // Test 1: Simple pattern - subf -> exp
  // ============================================
  // CHECK-LABEL: func @test_simple_pattern
  func.func @test_simple_pattern(%arg0: f32, %arg1: f32) -> f32 {
    %cst = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 1.000000e+00 : f32
    %sub = arith.subf %arg0, %arg1 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : f32
    // CHECK: math.exp %{{.*}} {ssbuffer.block_id = 1 : i32
    %exp = math.exp %sub {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : f32
    %mulf = arith.mulf %exp, %cst {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : f32
    %addf = arith.addf %exp, %cst {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : f32
    return %addf : f32
  }

  // ============================================
  // Test 2: Extended pattern - extf(f16->f32) -> subf -> exp
  // ============================================
  // CHECK-LABEL: func @test_extended_pattern
  func.func @test_extended_pattern(%arg0: f16, %arg1: f16) -> f32 {
    // CHECK: arith.extf %{{.*}} {ssbuffer.block_id = 1 : i32
    %extf1 = arith.extf %arg0 {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : f16 to f32
    // CHECK: arith.extf %{{.*}} {ssbuffer.block_id = 1 : i32
    %extf2 = arith.extf %arg1 {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR"} : f16 to f32
    %sub = arith.subf %extf1, %extf2 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : f32
    // CHECK: math.exp %{{.*}} {ssbuffer.block_id = 1 : i32
    %exp = math.exp %sub {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : f32
    %mulf = arith.mulf %exp, %exp {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : f32
    return %mulf : f32
  }

  // ============================================
  // Test 3: Extended pattern with matching blockId
  // ============================================
  // CHECK-LABEL: func @test_extended_pattern_matching_blockid
  func.func @test_extended_pattern_matching_blockid(%arg0: f16, %arg1: f16) -> f32 {
    // CHECK: arith.extf %{{.*}} {ssbuffer.block_id = 1 : i32
    %extf1 = arith.extf %arg0 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : f16 to f32
    // CHECK: arith.extf %{{.*}} {ssbuffer.block_id = 1 : i32
    %extf2 = arith.extf %arg1 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : f16 to f32
    %sub = arith.subf %extf1, %extf2 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : f32
    // CHECK: math.exp %{{.*}} {ssbuffer.block_id = 1 : i32
    %exp = math.exp %sub {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : f32
    %mulf = arith.mulf %exp, %exp {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : f32
    return %mulf : f32
  }

  // ============================================
  // Test 4: No match - subf has multiple users
  // ============================================
  // CHECK-LABEL: func @test_no_match_subf_multiple_users
  func.func @test_no_match_subf_multiple_users(%arg0: f32, %arg1: f32) -> f32 {
    %sub = arith.subf %arg0, %arg1 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : f32
    // CHECK: math.exp %{{.*}} {ssbuffer.block_id = 2 : i32
    %exp = math.exp %sub {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : f32
    %addf = arith.addf %sub, %arg0 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : f32
    return %addf : f32
  }

  // ============================================
  // Test 5: No match extended - extf has multiple users
  // ============================================
  // CHECK-LABEL: func @test_no_match_extf_multiple_users
  func.func @test_no_match_extf_multiple_users(%arg0: f16, %arg1: f16) -> f32 {
    // CHECK: arith.extf %{{.*}} {ssbuffer.block_id = 5 : i32
    %extf1 = arith.extf %arg0 {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : f16 to f32
    %extf2 = arith.extf %arg1 {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR"} : f16 to f32
    %sub = arith.subf %extf1, %extf2 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : f32
    // CHECK: math.exp %{{.*}} {ssbuffer.block_id = 1 : i32
    %exp = math.exp %sub {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : f32
    %addf = arith.addf %extf1, %extf2 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : f32
    %mulf = arith.mulf %exp, %addf {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : f32
    return %mulf : f32
  }

  // ============================================
  // Test 6: No match extended - wrong type conversion
  // ============================================
  // CHECK-LABEL: func @test_no_match_wrong_type_conversion
  func.func @test_no_match_wrong_type_conversion(%arg0: bf16, %arg1: bf16) -> f32 {
    // CHECK: arith.extf %{{.*}} {ssbuffer.block_id = 5 : i32
    %extf1 = arith.extf %arg0 {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : bf16 to f32
    // CHECK: arith.extf %{{.*}} {ssbuffer.block_id = 6 : i32
    %extf2 = arith.extf %arg1 {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR"} : bf16 to f32
    %sub = arith.subf %extf1, %extf2 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : f32
    // CHECK: math.exp %{{.*}} {ssbuffer.block_id = 1 : i32
    %exp = math.exp %sub {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : f32
    return %exp : f32
  }

  // ============================================
  // Test 7: Mixed pattern - one operand with extf, one without
  // ============================================
  // CHECK-LABEL: func @test_mixed_pattern
  func.func @test_mixed_pattern(%arg0: f16, %arg1: f32) -> f32 {
    // CHECK: arith.extf %{{.*}} {ssbuffer.block_id = 5 : i32
    %extf1 = arith.extf %arg0 {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : f16 to f32
    %sub = arith.subf %extf1, %arg1 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : f32
    // CHECK: math.exp %{{.*}} {ssbuffer.block_id = 1 : i32
    %exp = math.exp %sub {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : f32
    return %exp : f32
  }
}

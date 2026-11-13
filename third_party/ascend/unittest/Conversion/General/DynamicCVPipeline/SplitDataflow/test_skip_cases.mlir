// RUN: triton-opt -split-input-file --analyze-cube-control-flow-input-chain -verify-diagnostics -debug %s 2>&1 | FileCheck %s

// ============================================================================
// Scenario 1: Fallback case for linalg.reduce (Vector Only Op)
// linalg.reduce is unconditionally considered vector-only and triggers fallback.
// ============================================================================

func.func @test_scf_if_with_linalg_reduce(%input : tensor<16x16xf32>, %init : tensor<16xf32>, %c0 : index, %zero : f32) {
  scope.scope : () -> () {
    %reduced = linalg.reduce ins(%input : tensor<16x16xf32>) outs(%init : tensor<16xf32>) dimensions = [0]
      (%in: f32, %out: f32) {
        %sum = arith.addf %in, %out : f32
        linalg.yield %sum : f32
      }
    %val = tensor.extract %reduced[%c0] : tensor<16xf32>
    %cond = arith.cmpf ogt, %val, %zero : f32

    // CHECK: [analyze-cube-control-flow-input-chain] Fallback reason: incompatible upstream op for control flow: {{.*}} = linalg.reduce
    scf.if %cond {
      // some body
    }
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
  return
}

// -----

// ============================================================================
// Scenario 2: Fallback case for arith.select with RankedTensorType result
// arith.select returning a RankedTensorType is considered vector-only.
// ============================================================================

func.func @test_scf_for_with_tensor_select(%cond : i1, %t1 : tensor<16xindex>, %t2 : tensor<16xindex>, %c0 : index, %step : index) {
  scope.scope : () -> () {
    %select_tensor = arith.select %cond, %t1, %t2 : i1, tensor<16xindex>
    %ub = tensor.extract %select_tensor[%c0] : tensor<16xindex>
    %lb = arith.constant 0 : index

    // CHECK: [analyze-cube-control-flow-input-chain] Fallback reason: incompatible upstream op for control flow: {{.*}} = arith.select
    scf.for %i = %lb to %ub step %step {
      // some body
    }
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
  return
}

// -----

// ============================================================================
// Scenario 3: Fallback case for math.floor with RankedTensorType result
// math.floor returning a RankedTensorType is considered vector-only.
// ============================================================================

func.func @test_scf_while_with_tensor_floor(%input : tensor<16xf32>, %c0 : index) {
  scope.scope : () -> () {
    %floor_tensor = math.floor %input : tensor<16xf32>
    %init_val = tensor.extract %floor_tensor[%c0] : tensor<16xf32>

    // CHECK: [analyze-cube-control-flow-input-chain] Fallback reason: incompatible upstream op for control flow: {{.*}} = math.floor
    %result = scf.while (%arg = %init_val) : (f32) -> f32 {
      %cond = arith.constant true
      scf.condition(%cond) %arg : f32
    } do {
    ^bb0(%arg2: f32):
      scf.yield %arg2 : f32
    }
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
  return
}

// -----

// ============================================================================
// Scenario 4: Successful case with scalar select (No Fallback)
// arith.select returning a scalar is compatible and should NOT trigger fallback.
// ============================================================================

// CHECK-LABEL: func.func @test_scf_if_with_scalar_select
func.func @test_scf_if_with_scalar_select(%cond : i1, %val_a : f32, %val_b : f32) {
  scope.scope : () -> () {
    // This select returns a scalar (f32), which is NOT a RankedTensorType.
    %selected = arith.select %cond, %val_a, %val_b : f32
    %zero = arith.constant 0.0 : f32
    %if_cond = arith.cmpf ogt, %selected, %zero : f32

    // CHECK-NOT: Fallback reason:
    scf.if %if_cond {
      // some body
    }
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
  return
}

// -----

// ============================================================================
// Scenario 5: Successful case with scalar floor (No Fallback)
// math.floor returning a scalar is compatible and should NOT trigger fallback.
// ============================================================================

// CHECK-LABEL: func.func @test_scf_if_with_scalar_floor
func.func @test_scf_if_with_scalar_floor(%val : f32) {
  scope.scope : () -> () {
    // This math.floor returns a scalar (f32), which is NOT a RankedTensorType.
    %floored = math.floor %val : f32
    %zero = arith.constant 0.0 : f32
    %if_cond = arith.cmpf ogt, %floored, %zero : f32

    // CHECK-NOT: Fallback reason:
    scf.if %if_cond {
      // some body
    }
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
  return
}

// -----

// ============================================================================
// Scenario 6: Normal control flow without any vector-only ops (No Fallback)
// ============================================================================

// CHECK-LABEL: func.func @test_scf_if_fully_compatible
func.func @test_scf_if_fully_compatible(%val_a : f32, %val_b : f32) {
  scope.scope : () -> () {
    %cond = arith.cmpf ogt, %val_a, %val_b : f32

    // CHECK-NOT: Fallback reason:
    scf.if %cond {
      // some body
    }
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
  return
}

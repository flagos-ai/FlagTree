// RUN: triton-opt --verify-each %s -graph-optimize='rule-mask=2' -o - | FileCheck %s

// This is deliberately pre-layout TTIR. A direct tt.trans -> pointwise ->
// tt.dot chain is valid before GPU layouts are assigned; once GPU encodings
// exist, tt.dot requires DotOperandEncodingAttr while tt.trans inference only
// supports Blocked/Shared inputs. The rule keeps its encoded-input preflight
// fail-closed, but its structural rewrite is exercised here at the pipeline
  // stage where the chain can exist.

// Rule 2 asks the registered TritonGPU TensorOrMemDesc type interface to
// preflight a moved tt.trans. Keep every test function itself pre-layout
// TTIR, but instantiate the dialect once so standalone triton-opt has the
// same interface availability as the compiler pipeline.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>

module {
  // CHECK-LABEL: tt.func @positive_a_truncf_bitcast(
  // CHECK-SAME: %[[A_SRC:arg[0-9]+]]: tensor<8x16xf32>
  // CHECK-SAME: %[[A_B:arg[0-9]+]]: tensor<8x16xi16>
  // CHECK-SAME: %[[A_C:arg[0-9]+]]: tensor<16x16xi32>
  // CHECK-NOT: tt.trans %[[A_SRC]]
  // CHECK: %[[A_TRUNC:.*]] = arith.truncf %[[A_SRC]] : tensor<8x16xf32> to tensor<8x16xf16>
  // CHECK-NEXT: %[[A_BITCAST:.*]] = arith.bitcast %[[A_TRUNC]] : tensor<8x16xf16> to tensor<8x16xi16>
  // CHECK-NEXT: %[[A_TRANS:.*]] = tt.trans %[[A_BITCAST]] {order = array<i32: 1, 0>} : tensor<8x16xi16> -> tensor<16x8xi16>
  // CHECK-NEXT: %[[A_DOT:.*]] = tt.dot %[[A_TRANS]], %[[A_B]], %[[A_C]] : tensor<16x8xi16> * tensor<8x16xi16> -> tensor<16x16xi32>
  tt.func @positive_a_truncf_bitcast(
      %a_src: tensor<8x16xf32>,
      %b: tensor<8x16xi16>,
      %c: tensor<16x16xi32>) {
    %a_trans = tt.trans %a_src {order = array<i32: 1, 0>} : tensor<8x16xf32> -> tensor<16x8xf32>
    %a_trunc = arith.truncf %a_trans : tensor<16x8xf32> to tensor<16x8xf16>
    %a_bitcast = arith.bitcast %a_trunc : tensor<16x8xf16> to tensor<16x8xi16>
    %dot = tt.dot %a_bitcast, %b, %c : tensor<16x8xi16> * tensor<8x16xi16> -> tensor<16x16xi32>
    tt.return
  }

  // CHECK-LABEL: tt.func @positive_b_extf(
  // CHECK-SAME: %[[B_A:arg[0-9]+]]: tensor<16x8xf32>
  // CHECK-SAME: %[[B_SRC:arg[0-9]+]]: tensor<16x8xf16>
  // CHECK-SAME: %[[B_C:arg[0-9]+]]: tensor<16x16xf32>
  // CHECK-NOT: tt.trans %[[B_SRC]]
  // CHECK: %[[B_EXT:.*]] = arith.extf %[[B_SRC]] : tensor<16x8xf16> to tensor<16x8xf32>
  // CHECK-NEXT: %[[B_TRANS:.*]] = tt.trans %[[B_EXT]] {order = array<i32: 1, 0>} : tensor<16x8xf32> -> tensor<8x16xf32>
  // CHECK-NEXT: %[[B_DOT:.*]] = tt.dot %[[B_A]], %[[B_TRANS]], %[[B_C]] : tensor<16x8xf32> * tensor<8x16xf32> -> tensor<16x16xf32>
  tt.func @positive_b_extf(
      %a: tensor<16x8xf32>,
      %b_src: tensor<16x8xf16>,
      %c: tensor<16x16xf32>) {
    %b_trans = tt.trans %b_src {order = array<i32: 1, 0>} : tensor<16x8xf16> -> tensor<8x16xf16>
    %b_ext = arith.extf %b_trans : tensor<8x16xf16> to tensor<8x16xf32>
    %dot = tt.dot %a, %b_ext, %c : tensor<16x8xf32> * tensor<8x16xf32> -> tensor<16x16xf32>
    tt.return
  }

  // CHECK-LABEL: tt.func @positive_a_mixed_cast_math_chain(
  // CHECK-SAME: %[[M_SRC:arg[0-9]+]]: tensor<8x16xf32>
  // CHECK-SAME: %[[M_B:arg[0-9]+]]: tensor<8x16xi16>
  // CHECK-SAME: %[[M_C:arg[0-9]+]]: tensor<16x16xi32>
  // CHECK-NOT: tt.trans %[[M_SRC]]
  // CHECK: %[[M_NEG:.*]] = arith.negf %[[M_SRC]] : tensor<8x16xf32>
  // CHECK-NEXT: %[[M_ABS:.*]] = math.absf %[[M_NEG]] : tensor<8x16xf32>
  // CHECK-NEXT: %[[M_TRUNC:.*]] = arith.truncf %[[M_ABS]] : tensor<8x16xf32> to tensor<8x16xf16>
  // CHECK-NEXT: %[[M_EXP:.*]] = math.exp %[[M_TRUNC]] : tensor<8x16xf16>
  // CHECK-NEXT: %[[M_BITS:.*]] = arith.bitcast %[[M_EXP]] : tensor<8x16xf16> to tensor<8x16xi16>
  // CHECK-NEXT: %[[M_TRANS:.*]] = tt.trans %[[M_BITS]] {order = array<i32: 1, 0>} : tensor<8x16xi16> -> tensor<16x8xi16>
  // CHECK-NEXT: %{{.*}} = tt.dot %[[M_TRANS]], %[[M_B]], %[[M_C]] : tensor<16x8xi16> * tensor<8x16xi16> -> tensor<16x16xi32>
  tt.func @positive_a_mixed_cast_math_chain(
      %a_src: tensor<8x16xf32>,
      %b: tensor<8x16xi16>,
      %c: tensor<16x16xi32>) {
    %trans = tt.trans %a_src {order = array<i32: 1, 0>} : tensor<8x16xf32> -> tensor<16x8xf32>
    %neg = arith.negf %trans : tensor<16x8xf32>
    %abs = math.absf %neg : tensor<16x8xf32>
    %trunc = arith.truncf %abs : tensor<16x8xf32> to tensor<16x8xf16>
    %exp = math.exp %trunc : tensor<16x8xf16>
    %bits = arith.bitcast %exp : tensor<16x8xf16> to tensor<16x8xi16>
    %dot = tt.dot %bits, %b, %c : tensor<16x8xi16> * tensor<8x16xi16> -> tensor<16x16xi32>
    tt.return
  }

  // CHECK-LABEL: tt.func @positive_a_long_float_unary_chain(
  // CHECK-SAME: %[[L_SRC:arg[0-9]+]]: tensor<8x16xf32>
  // CHECK-NOT: tt.trans %[[L_SRC]]
  // CHECK: %[[L0:.*]] = arith.negf %[[L_SRC]] : tensor<8x16xf32>
  // CHECK-NEXT: %[[L1:.*]] = math.absf %[[L0]] : tensor<8x16xf32>
  // CHECK-NEXT: %[[L2:.*]] = math.ceil %[[L1]] : tensor<8x16xf32>
  // CHECK-NEXT: %[[L3:.*]] = math.floor %[[L2]] : tensor<8x16xf32>
  // CHECK-NEXT: %[[L4:.*]] = math.cos %[[L3]] : tensor<8x16xf32>
  // CHECK-NEXT: %[[L5:.*]] = math.sin %[[L4]] : tensor<8x16xf32>
  // CHECK-NEXT: %[[L6:.*]] = math.erf %[[L5]] : tensor<8x16xf32>
  // CHECK-NEXT: %[[L7:.*]] = math.exp %[[L6]] : tensor<8x16xf32>
  // CHECK-NEXT: %[[L8:.*]] = math.exp2 %[[L7]] : tensor<8x16xf32>
  // CHECK-NEXT: %[[L9:.*]] = math.log %[[L8]] : tensor<8x16xf32>
  // CHECK-NEXT: %[[L10:.*]] = math.log2 %[[L9]] : tensor<8x16xf32>
  // CHECK-NEXT: %[[L11:.*]] = math.sqrt %[[L10]] : tensor<8x16xf32>
  // CHECK-NEXT: %[[L12:.*]] = math.rsqrt %[[L11]] : tensor<8x16xf32>
  // CHECK-NEXT: %[[L13:.*]] = math.tanh %[[L12]] : tensor<8x16xf32>
  // CHECK-NEXT: %[[L14:.*]] = tt.precise_sqrt %[[L13]] : tensor<8x16xf32>
  // CHECK-NEXT: %[[L_TRANS:.*]] = tt.trans %[[L14]] {order = array<i32: 1, 0>} : tensor<8x16xf32> -> tensor<16x8xf32>
  // CHECK-NEXT: %{{.*}} = tt.dot %[[L_TRANS]], %{{.*}}, %{{.*}} : tensor<16x8xf32> * tensor<8x16xf32> -> tensor<16x16xf32>
  tt.func @positive_a_long_float_unary_chain(
      %a_src: tensor<8x16xf32>,
      %b: tensor<8x16xf32>,
      %c: tensor<16x16xf32>) {
    %trans = tt.trans %a_src {order = array<i32: 1, 0>} : tensor<8x16xf32> -> tensor<16x8xf32>
    %v0 = arith.negf %trans : tensor<16x8xf32>
    %v1 = math.absf %v0 : tensor<16x8xf32>
    %v2 = math.ceil %v1 : tensor<16x8xf32>
    %v3 = math.floor %v2 : tensor<16x8xf32>
    %v4 = math.cos %v3 : tensor<16x8xf32>
    %v5 = math.sin %v4 : tensor<16x8xf32>
    %v6 = math.erf %v5 : tensor<16x8xf32>
    %v7 = math.exp %v6 : tensor<16x8xf32>
    %v8 = math.exp2 %v7 : tensor<16x8xf32>
    %v9 = math.log %v8 : tensor<16x8xf32>
    %v10 = math.log2 %v9 : tensor<16x8xf32>
    %v11 = math.sqrt %v10 : tensor<16x8xf32>
    %v12 = math.rsqrt %v11 : tensor<16x8xf32>
    %v13 = math.tanh %v12 : tensor<16x8xf32>
    %v14 = tt.precise_sqrt %v13 : tensor<16x8xf32>
    %dot = tt.dot %v14, %b, %c : tensor<16x8xf32> * tensor<8x16xf32> -> tensor<16x16xf32>
    tt.return
  }

  // CHECK-LABEL: tt.func @positive_a_absi(
  // CHECK-SAME: %[[I_SRC:arg[0-9]+]]: tensor<8x16xi16>
  // CHECK-NOT: tt.trans %[[I_SRC]]
  // CHECK: %[[I_ABS:.*]] = math.absi %[[I_SRC]] : tensor<8x16xi16>
  // CHECK-NEXT: %[[I_TRANS:.*]] = tt.trans %[[I_ABS]] {order = array<i32: 1, 0>} : tensor<8x16xi16> -> tensor<16x8xi16>
  // CHECK-NEXT: %{{.*}} = tt.dot %[[I_TRANS]], %{{.*}}, %{{.*}} : tensor<16x8xi16> * tensor<8x16xi16> -> tensor<16x16xi32>
  tt.func @positive_a_absi(
      %a_src: tensor<8x16xi16>,
      %b: tensor<8x16xi16>,
      %c: tensor<16x16xi32>) {
    %trans = tt.trans %a_src {order = array<i32: 1, 0>} : tensor<8x16xi16> -> tensor<16x8xi16>
    %abs = math.absi %trans : tensor<16x8xi16>
    %dot = tt.dot %abs, %b, %c : tensor<16x8xi16> * tensor<8x16xi16> -> tensor<16x16xi32>
    tt.return
  }

  // CHECK-LABEL: tt.func @reject_external_trans_user(
  // CHECK: %[[EXTERNAL_TRANS:.*]] = tt.trans %{{.*}} {order = array<i32: 1, 0>} : tensor<8x16xf32> -> tensor<16x8xf32>
  // CHECK-NEXT: %[[EXTERNAL_CAST:.*]] = arith.truncf %[[EXTERNAL_TRANS]] : tensor<16x8xf32> to tensor<16x8xf16>
  // CHECK-NEXT: %{{.*}} = tt.trans %[[EXTERNAL_TRANS]] {order = array<i32: 1, 0>} : tensor<16x8xf32> -> tensor<8x16xf32>
  // CHECK-NEXT: %[[EXTERNAL_DOT:.*]] = tt.dot %[[EXTERNAL_CAST]], %{{.*}}, %{{.*}} : tensor<16x8xf16> * tensor<8x16xf16> -> tensor<16x16xf32>
  tt.func @reject_external_trans_user(
      %a_src: tensor<8x16xf32>,
      %b: tensor<8x16xf16>,
      %c: tensor<16x16xf32>) {
    %a_trans = tt.trans %a_src {order = array<i32: 1, 0>} : tensor<8x16xf32> -> tensor<16x8xf32>
    %a_trunc = arith.truncf %a_trans : tensor<16x8xf32> to tensor<16x8xf16>
    %extra = tt.trans %a_trans {order = array<i32: 1, 0>} : tensor<16x8xf32> -> tensor<8x16xf32>
    %dot = tt.dot %a_trunc, %b, %c : tensor<16x8xf16> * tensor<8x16xf16> -> tensor<16x16xf32>
    tt.return
  }

  // CHECK-LABEL: tt.func @reject_addf(
  // CHECK: %[[ADD_TRANS:.*]] = tt.trans %{{.*}} {order = array<i32: 1, 0>} : tensor<8x16xf16> -> tensor<16x8xf16>
  // CHECK-NEXT: %[[ADD:.*]] = arith.addf %[[ADD_TRANS]], %{{.*}} : tensor<16x8xf16>
  // CHECK-NEXT: %[[ADD_DOT:.*]] = tt.dot %[[ADD]], %{{.*}}, %{{.*}} : tensor<16x8xf16> * tensor<8x16xf16> -> tensor<16x16xf32>
  tt.func @reject_addf(
      %a_src: tensor<8x16xf16>,
      %b: tensor<8x16xf16>,
      %c: tensor<16x16xf32>) {
    %zero = arith.constant dense<0.0> : tensor<16x8xf16>
    %a_trans = tt.trans %a_src {order = array<i32: 1, 0>} : tensor<8x16xf16> -> tensor<16x8xf16>
    %a_add = arith.addf %a_trans, %zero : tensor<16x8xf16>
    %dot = tt.dot %a_add, %b, %c : tensor<16x8xf16> * tensor<8x16xf16> -> tensor<16x16xf32>
    tt.return
  }

  // CHECK-LABEL: tt.func @positive_a_generic_elementwise(
  // CHECK-SAME: %[[G_SRC:arg[0-9]+]]: tensor<8x16xf32>
  // CHECK-NOT: tt.trans %[[G_SRC]]
  // CHECK: %[[G_ATAN:.*]] = math.atan %[[G_SRC]] : tensor<8x16xf32>
  // CHECK-NEXT: %[[G_TAN:.*]] = math.tan %[[G_ATAN]] : tensor<8x16xf32>
  // CHECK-NEXT: %[[G_TRANS:.*]] = tt.trans %[[G_TAN]] {order = array<i32: 1, 0>} : tensor<8x16xf32> -> tensor<16x8xf32>
  // CHECK-NEXT: %{{.*}} = tt.dot %[[G_TRANS]], %{{.*}}, %{{.*}} : tensor<16x8xf32> * tensor<8x16xf32> -> tensor<16x16xf32>
  tt.func @positive_a_generic_elementwise(
      %a_src: tensor<8x16xf32>,
      %b: tensor<8x16xf32>,
      %c: tensor<16x16xf32>) {
    %trans = tt.trans %a_src {order = array<i32: 1, 0>} : tensor<8x16xf32> -> tensor<16x8xf32>
    %atan = math.atan %trans : tensor<16x8xf32>
    %tan = math.tan %atan : tensor<16x8xf32>
    %dot = tt.dot %tan, %b, %c : tensor<16x8xf32> * tensor<8x16xf32> -> tensor<16x16xf32>
    tt.return
  }

  // CHECK-LABEL: tt.func @reject_type_changing_unary(
  // CHECK-SAME: %[[U_SRC:arg[0-9]+]]: tensor<8x16xf32>
  // CHECK-NOT: arith.fptosi %[[U_SRC]]
  // CHECK: %[[U_TRANS:.*]] = tt.trans %[[U_SRC]] {order = array<i32: 1, 0>} : tensor<8x16xf32> -> tensor<16x8xf32>
  // CHECK-NEXT: %[[U_CONVERTED:.*]] = arith.fptosi %[[U_TRANS]] : tensor<16x8xf32> to tensor<16x8xi16>
  // CHECK-NEXT: %{{.*}} = tt.dot %[[U_CONVERTED]], %{{.*}}, %{{.*}} : tensor<16x8xi16> * tensor<8x16xi16> -> tensor<16x16xi32>
  tt.func @reject_type_changing_unary(
      %a_src: tensor<8x16xf32>,
      %b: tensor<8x16xi16>,
      %c: tensor<16x16xi32>) {
    %trans = tt.trans %a_src {order = array<i32: 1, 0>} : tensor<8x16xf32> -> tensor<16x8xf32>
    %converted = arith.fptosi %trans : tensor<16x8xf32> to tensor<16x8xi16>
    %dot = tt.dot %converted, %b, %c : tensor<16x8xi16> * tensor<8x16xi16> -> tensor<16x16xi32>
    tt.return
  }

  // CHECK-LABEL: tt.func @reject_unary_external_user(
  // CHECK: %[[E_TRANS:.*]] = tt.trans %{{.*}} {order = array<i32: 1, 0>} : tensor<8x16xf32> -> tensor<16x8xf32>
  // CHECK-NEXT: %[[E_EXP:.*]] = math.exp %[[E_TRANS]] : tensor<16x8xf32>
  // CHECK-NEXT: %{{.*}} = math.cos %[[E_EXP]] : tensor<16x8xf32>
  // CHECK-NEXT: %{{.*}} = tt.dot %[[E_EXP]], %{{.*}}, %{{.*}} : tensor<16x8xf32> * tensor<8x16xf32> -> tensor<16x16xf32>
  tt.func @reject_unary_external_user(
      %a_src: tensor<8x16xf32>,
      %b: tensor<8x16xf32>,
      %c: tensor<16x16xf32>) {
    %trans = tt.trans %a_src {order = array<i32: 1, 0>} : tensor<8x16xf32> -> tensor<16x8xf32>
    %exp = math.exp %trans : tensor<16x8xf32>
    %extra = math.cos %exp : tensor<16x8xf32>
    %dot = tt.dot %exp, %b, %c : tensor<16x8xf32> * tensor<8x16xf32> -> tensor<16x16xf32>
    tt.return
  }

  // The accumulator has the matching shape but is deliberately not an A/B
  // candidate. Its original transpose/cast chain must stay untouched.
  // CHECK-LABEL: tt.func @reject_dot_accumulator(
  // CHECK: %[[C_TRANS:.*]] = tt.trans %{{.*}} {order = array<i32: 1, 0>} : tensor<16x16xf64> -> tensor<16x16xf64>
  // CHECK-NEXT: %[[C_TRUNC:.*]] = arith.truncf %[[C_TRANS]] : tensor<16x16xf64> to tensor<16x16xf32>
  // CHECK-NEXT: %[[C_DOT:.*]] = tt.dot %{{.*}}, %{{.*}}, %[[C_TRUNC]] : tensor<16x8xf16> * tensor<8x16xf16> -> tensor<16x16xf32>
  tt.func @reject_dot_accumulator(
      %a: tensor<16x8xf16>,
      %b: tensor<8x16xf16>,
      %c_src: tensor<16x16xf64>) {
    %c_trans = tt.trans %c_src {order = array<i32: 1, 0>} : tensor<16x16xf64> -> tensor<16x16xf64>
    %c_trunc = arith.truncf %c_trans : tensor<16x16xf64> to tensor<16x16xf32>
    %dot = tt.dot %a, %b, %c_trunc : tensor<16x8xf16> * tensor<8x16xf16> -> tensor<16x16xf32>
    tt.return
  }

  // CHECK-LABEL: tt.func @reject_region_boundary(
  // CHECK: %[[REGION_TRANS:.*]] = tt.trans %{{.*}} {order = array<i32: 1, 0>} : tensor<8x16xf32> -> tensor<16x8xf32>
  // CHECK: scf.if %{{.*}} -> (tensor<16x16xf32>) {
  // CHECK: %[[REGION_CAST:.*]] = arith.truncf %[[REGION_TRANS]] : tensor<16x8xf32> to tensor<16x8xf16>
  // CHECK-NEXT: %[[REGION_DOT:.*]] = tt.dot %[[REGION_CAST]], %{{.*}}, %{{.*}} : tensor<16x8xf16> * tensor<8x16xf16> -> tensor<16x16xf32>
  tt.func @reject_region_boundary(
      %condition: i1,
      %a_src: tensor<8x16xf32>,
      %b: tensor<8x16xf16>,
      %c: tensor<16x16xf32>) {
    %a_trans = tt.trans %a_src {order = array<i32: 1, 0>} : tensor<8x16xf32> -> tensor<16x8xf32>
    %result = scf.if %condition -> (tensor<16x16xf32>) {
      %a_trunc = arith.truncf %a_trans : tensor<16x8xf32> to tensor<16x8xf16>
      %dot = tt.dot %a_trunc, %b, %c : tensor<16x8xf16> * tensor<8x16xf16> -> tensor<16x16xf32>
      scf.yield %dot : tensor<16x16xf32>
    } else {
      scf.yield %c : tensor<16x16xf32>
    }
    tt.return
  }
}

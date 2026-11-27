// RUN: triton-cuda-tile-opt %s -split-input-file -verify-diagnostics --pass-pipeline="builtin.module(convert-triton-to-cuda-tile{approx-modifier=true flush-to-zero-modifier=true num-warps-in-cta=8},reconcile-unrealized-casts)" | FileCheck --check-prefixes=APPROX,NATIVE,ASM %s
// RUN: triton-cuda-tile-opt %s -split-input-file -verify-diagnostics --pass-pipeline="builtin.module(convert-triton-to-cuda-tile{approx-modifier=false flush-to-zero-modifier=false num-warps-in-cta=4},reconcile-unrealized-casts)" | FileCheck --check-prefixes=FULL,NATIVE,ASM %s

module {
  tt.func public @exp_precision(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>) {
    %x = tt.load %input : !tt.ptr<f32>
    %value = math.exp %x : f32
    tt.store %output, %value : !tt.ptr<f32>
    tt.return
  }
}
// APPROX-LABEL: entry @exp_precision
// APPROX-SAME: num_worker_warps_per_cta = 8
// APPROX: exp {{.*}} rounding<approx>
// FULL-LABEL: entry @exp_precision
// FULL-SAME: num_worker_warps_per_cta = 4
// FULL: exp %{{[^ ]+}} : tile<f32>

// -----

module {
  tt.func public @native_libdevice_f32(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %c: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %x = tt.load %a : !tt.ptr<f32>
    %y = tt.load %b : !tt.ptr<f32>
    %z = tt.load %c : !tt.ptr<f32>
    %v0 = tt.extern_elementwise %x, %y {libname = "", libpath = "", pure = true, symbol = "__nv_atan2f"} : (f32, f32) -> f32
    tt.store %out, %v0 : !tt.ptr<f32>
    %v1 = tt.extern_elementwise %x {libname = "", libpath = "", pure = true, symbol = "__nv_coshf"} : (f32) -> f32
    tt.store %out, %v1 : !tt.ptr<f32>
    %v2 = tt.extern_elementwise %x {libname = "", libpath = "", pure = true, symbol = "__nv_sinhf"} : (f32) -> f32
    tt.store %out, %v2 : !tt.ptr<f32>
    %v3 = tt.extern_elementwise %x {libname = "", libpath = "", pure = true, symbol = "__nv_fabsf"} : (f32) -> f32
    tt.store %out, %v3 : !tt.ptr<f32>
    %v4 = tt.extern_elementwise %x {libname = "", libpath = "", pure = true, symbol = "__nv_logf"} : (f32) -> f32
    tt.store %out, %v4 : !tt.ptr<f32>
    %v5 = tt.extern_elementwise %x, %y {libname = "", libpath = "", pure = true, symbol = "__nv_fmodf"} : (f32, f32) -> f32
    tt.store %out, %v5 : !tt.ptr<f32>
    %v6 = tt.extern_elementwise %x, %y, %z {libname = "", libpath = "", pure = true, symbol = "__nv_fmaf"} : (f32, f32, f32) -> f32
    tt.store %out, %v6 : !tt.ptr<f32>
    %v7 = tt.extern_elementwise %x, %y {libname = "", libpath = "", pure = true, symbol = "__nv_fdiv_rn"} : (f32, f32) -> f32
    tt.store %out, %v7 : !tt.ptr<f32>
    %v8 = tt.extern_elementwise %x, %y {libname = "", libpath = "", pure = true, symbol = "__nv_fdiv_rz"} : (f32, f32) -> f32
    tt.store %out, %v8 : !tt.ptr<f32>
    %v9 = tt.extern_elementwise %x, %y {libname = "", libpath = "", pure = true, symbol = "__nv_fdiv_rd"} : (f32, f32) -> f32
    tt.store %out, %v9 : !tt.ptr<f32>
    %v10 = tt.extern_elementwise %x, %y {libname = "", libpath = "", pure = true, symbol = "__nv_fdiv_ru"} : (f32, f32) -> f32
    tt.store %out, %v10 : !tt.ptr<f32>
    %v11 = tt.extern_elementwise %x, %y {libname = "", libpath = "", pure = true, symbol = "__nv_fadd_rz"} : (f32, f32) -> f32
    tt.store %out, %v11 : !tt.ptr<f32>
    %v12 = tt.extern_elementwise %x, %y {libname = "", libpath = "", pure = true, symbol = "__nv_fsub_rd"} : (f32, f32) -> f32
    tt.store %out, %v12 : !tt.ptr<f32>
    %v13 = tt.extern_elementwise %x, %y {libname = "", libpath = "", pure = true, symbol = "__nv_fmul_ru"} : (f32, f32) -> f32
    tt.store %out, %v13 : !tt.ptr<f32>
    %v14 = tt.extern_elementwise %x {libname = "", libpath = "", pure = true, symbol = "__nv_fsqrt_rz"} : (f32) -> f32
    tt.store %out, %v14 : !tt.ptr<f32>
    %v15 = tt.extern_elementwise %x {libname = "", libpath = "", pure = true, symbol = "__nv_frcp_rn"} : (f32) -> f32
    tt.store %out, %v15 : !tt.ptr<f32>
    %bits = tt.extern_elementwise %x {libname = "", libpath = "", pure = true, symbol = "__nv_float_as_int"} : (f32) -> i32
    %back = tt.extern_elementwise %bits {libname = "", libpath = "", pure = true, symbol = "__nv_int_as_float"} : (i32) -> f32
    tt.store %out, %back : !tt.ptr<f32>
    tt.return
  }
}
// NATIVE-LABEL: entry @native_libdevice_f32
// NATIVE: atan2 {{.*}}: tile<f32>
// NATIVE: cosh {{.*}}: tile<f32>
// NATIVE: sinh {{.*}}: tile<f32>
// NATIVE: absf {{.*}}: tile<f32>
// NATIVE: log {{.*}}: tile<f32>
// NATIVE: remf {{.*}}: tile<f32>
// NATIVE: fma {{.*}}: tile<f32>
// NATIVE: divf {{.*}}: tile<f32>
// NATIVE: divf {{.*}}rounding<zero> : tile<f32>
// NATIVE: divf {{.*}}rounding<negative_inf> : tile<f32>
// NATIVE: divf {{.*}}rounding<positive_inf> : tile<f32>
// NATIVE: addf {{.*}}rounding<zero> : tile<f32>
// NATIVE: subf {{.*}}rounding<negative_inf> : tile<f32>
// NATIVE: mulf {{.*}}rounding<positive_inf> : tile<f32>
// NATIVE: sqrt {{.*}}rounding<zero> : tile<f32>
// NATIVE: divf {{.*}}: tile<f32>
// NATIVE: bitcast {{.*}} : tile<f32> -> tile<i32>
// NATIVE: bitcast {{.*}} : tile<i32> -> tile<f32>

// -----

module {
  tt.func public @native_libdevice_f64(%a: !tt.ptr<f64>, %b: !tt.ptr<f64>, %c: !tt.ptr<f64>, %out: !tt.ptr<f64>) {
    %x = tt.load %a : !tt.ptr<f64>
    %y = tt.load %b : !tt.ptr<f64>
    %z = tt.load %c : !tt.ptr<f64>
    %v0 = tt.extern_elementwise %x, %y {libname = "", libpath = "", pure = true, symbol = "__nv_atan2"} : (f64, f64) -> f64
    tt.store %out, %v0 : !tt.ptr<f64>
    %v1 = tt.extern_elementwise %x {libname = "", libpath = "", pure = true, symbol = "__nv_cosh"} : (f64) -> f64
    tt.store %out, %v1 : !tt.ptr<f64>
    %v2 = tt.extern_elementwise %x {libname = "", libpath = "", pure = true, symbol = "__nv_sinh"} : (f64) -> f64
    tt.store %out, %v2 : !tt.ptr<f64>
    %v3 = tt.extern_elementwise %x {libname = "", libpath = "", pure = true, symbol = "__nv_fabs"} : (f64) -> f64
    tt.store %out, %v3 : !tt.ptr<f64>
    %v4 = tt.extern_elementwise %x {libname = "", libpath = "", pure = true, symbol = "__nv_log"} : (f64) -> f64
    tt.store %out, %v4 : !tt.ptr<f64>
    %v5 = tt.extern_elementwise %x, %y {libname = "", libpath = "", pure = true, symbol = "__nv_fmod"} : (f64, f64) -> f64
    tt.store %out, %v5 : !tt.ptr<f64>
    %v6 = tt.extern_elementwise %x, %y, %z {libname = "", libpath = "", pure = true, symbol = "__nv_fma"} : (f64, f64, f64) -> f64
    tt.store %out, %v6 : !tt.ptr<f64>
    %v7 = tt.extern_elementwise %x, %y {libname = "", libpath = "", pure = true, symbol = "__nv_ddiv_rn"} : (f64, f64) -> f64
    tt.store %out, %v7 : !tt.ptr<f64>
    %v8 = tt.extern_elementwise %x, %y {libname = "", libpath = "", pure = true, symbol = "__nv_ddiv_rz"} : (f64, f64) -> f64
    tt.store %out, %v8 : !tt.ptr<f64>
    %v9 = tt.extern_elementwise %x, %y {libname = "", libpath = "", pure = true, symbol = "__nv_ddiv_rd"} : (f64, f64) -> f64
    tt.store %out, %v9 : !tt.ptr<f64>
    %v10 = tt.extern_elementwise %x, %y {libname = "", libpath = "", pure = true, symbol = "__nv_ddiv_ru"} : (f64, f64) -> f64
    tt.store %out, %v10 : !tt.ptr<f64>
    %v11 = tt.extern_elementwise %x, %y {libname = "", libpath = "", pure = true, symbol = "__nv_dadd_rz"} : (f64, f64) -> f64
    tt.store %out, %v11 : !tt.ptr<f64>
    %v12 = tt.extern_elementwise %x, %y {libname = "", libpath = "", pure = true, symbol = "__nv_dsub_rd"} : (f64, f64) -> f64
    tt.store %out, %v12 : !tt.ptr<f64>
    %v13 = tt.extern_elementwise %x, %y {libname = "", libpath = "", pure = true, symbol = "__nv_dmul_ru"} : (f64, f64) -> f64
    tt.store %out, %v13 : !tt.ptr<f64>
    %v14 = tt.extern_elementwise %x {libname = "", libpath = "", pure = true, symbol = "__nv_dsqrt_rz"} : (f64) -> f64
    tt.store %out, %v14 : !tt.ptr<f64>
    %v15 = tt.extern_elementwise %x {libname = "", libpath = "", pure = true, symbol = "__nv_drcp_rn"} : (f64) -> f64
    tt.store %out, %v15 : !tt.ptr<f64>
    %bits = tt.extern_elementwise %x {libname = "", libpath = "", pure = true, symbol = "__nv_double_as_longlong"} : (f64) -> i64
    %back = tt.extern_elementwise %bits {libname = "", libpath = "", pure = true, symbol = "__nv_longlong_as_double"} : (i64) -> f64
    tt.store %out, %back : !tt.ptr<f64>
    tt.return
  }
}
// NATIVE-LABEL: entry @native_libdevice_f64
// NATIVE: atan2 {{.*}}: tile<f64>
// NATIVE: cosh {{.*}}: tile<f64>
// NATIVE: sinh {{.*}}: tile<f64>
// NATIVE: absf {{.*}}: tile<f64>
// NATIVE: log {{.*}}: tile<f64>
// NATIVE: remf {{.*}}: tile<f64>
// NATIVE: fma {{.*}}: tile<f64>
// NATIVE: divf {{.*}}: tile<f64>
// NATIVE: divf {{.*}}rounding<zero> : tile<f64>
// NATIVE: divf {{.*}}rounding<negative_inf> : tile<f64>
// NATIVE: divf {{.*}}rounding<positive_inf> : tile<f64>
// NATIVE: addf {{.*}}rounding<zero> : tile<f64>
// NATIVE: subf {{.*}}rounding<negative_inf> : tile<f64>
// NATIVE: mulf {{.*}}rounding<positive_inf> : tile<f64>
// NATIVE: sqrt {{.*}}rounding<zero> : tile<f64>
// NATIVE: divf {{.*}}: tile<f64>
// NATIVE: bitcast {{.*}} : tile<f64> -> tile<i64>
// NATIVE: bitcast {{.*}} : tile<i64> -> tile<f64>

// -----
// Native forms must ignore general approximation/FTZ pass options. Tensor
// packing also checks the PTX operand-to-nibble ordering across rows.
module {
  tt.func public @asm_fp4_pack(%hi: f32, %lo: f32, %out: !tt.ptr<i8>) {
    %h = tt.splat %hi : f32 -> tensor<2x4xf32>
    %l = tt.splat %lo : f32 -> tensor<2x4xf32>
    %r = tt.elementwise_inline_asm "\0A { .reg .b8 r;\0A cvt.rn.satfinite.e2m1x2.f32 r, $1, $2; mov.b32 $0, {r, r, r, r}; }\0A" {constraints = "=r,f,f", packed_element = 1 : i32, pure = true} %h, %l : tensor<2x4xf32>, tensor<2x4xf32> -> tensor<2x4xi8>
    %p = tt.splat %out : !tt.ptr<i8> -> tensor<2x4x!tt.ptr<i8>>
    tt.store %p, %r : tensor<2x4x!tt.ptr<i8>>
    tt.return
  }
}
// ASM-LABEL: entry @asm_fp4_pack
// ASM: %[[HI:.*]] = broadcast {{.*}} -> tile<2x4xf32>
// ASM: %[[LO:.*]] = broadcast {{.*}} -> tile<2x4xf32>
// ASM: %[[L:.*]] = reshape %[[LO]] : tile<2x4xf32> -> tile<8x1xf32>
// ASM: %[[H:.*]] = reshape %[[HI]] : tile<2x4xf32> -> tile<8x1xf32>
// ASM: %[[PAIR:.*]] = cat %[[L]], %[[H]] dim = 1
// ASM: %[[FLAT:.*]] = reshape %[[PAIR]] : tile<8x2xf32> -> tile<16xf32>
// ASM: %[[F4:.*]] = ftof %[[FLAT]] : tile<16xf32> -> tile<16xf4E2M1FN>
// ASM: %[[BYTES:.*]] = pack %[[F4]] : tile<16xf4E2M1FN> -> tile<8xi8>
// ASM: reshape %[[BYTES]] : tile<8xi8> -> tile<2x4xi8>
// ASM-NOT: elementwise_inline_asm

// -----
module {
  tt.func public @asm_exp2_ftz(%x: f32, %out: !tt.ptr<f32>) {
    %r = tt.elementwise_inline_asm "ex2.approx.ftz.f32 $0, $1;" {constraints = "=r, r", packed_element = 1 : i32, pure = true} %x : f32 -> f32
    tt.store %out, %r : !tt.ptr<f32>
    tt.return
  }
}
// ASM-LABEL: entry @asm_exp2_ftz
// ASM: exp2 %{{[^ ]+}} flush_to_zero : tile<f32>
// ASM-NOT: elementwise_inline_asm

// -----
module {
  tt.func public @asm_tf32_rn(%x: f32, %out: !tt.ptr<f32>) {
    %r = tt.elementwise_inline_asm "cvt.rn.tf32.f32 $0, $1;" {constraints = "=r, r", packed_element = 1 : i32, pure = true} %x : f32 -> f32
    tt.store %out, %r : !tt.ptr<f32>
    tt.return
  }
}
// ASM-LABEL: entry @asm_tf32_rn
// ASM: %[[TF32:.*]] = ftof %{{[^ ]+}} : tile<f32> -> tile<tf32>
// ASM: ftof %[[TF32]] : tile<tf32> -> tile<f32>
// ASM-NOT: elementwise_inline_asm

// -----
module {
  tt.func public @asm_max_nan_xorsign_abs(%a: f32, %b: f32, %out: !tt.ptr<f32>) {
    %r = tt.elementwise_inline_asm "{\0A max.NaN.xorsign.abs.f32 $0, $1, $2;\0A }" {constraints = "=r,r,r", packed_element = 1 : i32, pure = true} %a, %b : f32, f32 -> f32
    tt.store %out, %r : !tt.ptr<f32>
    tt.return
  }
}
// ASM-LABEL: entry @asm_max_nan_xorsign_abs
// ASM: xori
// ASM: %[[MAG:.*]] = maxf {{.*}} propagate_nan : tile<f32>
// ASM: %[[MB:.*]] = bitcast %[[MAG]] : tile<f32> -> tile<i32>
// ASM: cmpi {{.*}} unsigned
// ASM: %[[SIGNED:.*]] = ori %[[MB]],
// ASM: select {{.*}}%[[MB]], %[[SIGNED]]
// ASM-NOT: flush_to_zero
// ASM-NOT: elementwise_inline_asm

// -----
module {
  tt.func public @reject_exp2_without_ftz(%a0: f32, %out: !tt.ptr<f32>) {
    // expected-error@+1 {{failed to legalize operation 'tt.elementwise_inline_asm'}}
    %r = tt.elementwise_inline_asm "ex2.approx.f32 $0, $1;" {constraints = "=r,r", packed_element = 1 : i32, pure = true} %a0 : f32 -> f32
    tt.store %out, %r : !tt.ptr<f32>
    tt.return
  }
}

// -----
module {
  tt.func public @reject_exp2_wrong_constraint(%a0: f32, %out: !tt.ptr<f32>) {
    // expected-error@+1 {{failed to legalize operation 'tt.elementwise_inline_asm'}}
    %r = tt.elementwise_inline_asm "ex2.approx.ftz.f32 $0, $1;" {constraints = "=f,f", packed_element = 1 : i32, pure = true} %a0 : f32 -> f32
    tt.store %out, %r : !tt.ptr<f32>
    tt.return
  }
}

// -----
module {
  tt.func public @reject_exp2_wrong_type(%a0: f16, %out: !tt.ptr<f16>) {
    // expected-error@+1 {{failed to legalize operation 'tt.elementwise_inline_asm'}}
    %r = tt.elementwise_inline_asm "ex2.approx.ftz.f32 $0, $1;" {constraints = "=r,r", packed_element = 1 : i32, pure = true} %a0 : f16 -> f16
    tt.store %out, %r : !tt.ptr<f16>
    tt.return
  }
}

// -----
module {
  tt.func public @reject_exp2_impure(%a0: f32, %out: !tt.ptr<f32>) {
    // expected-error@+1 {{failed to legalize operation 'tt.elementwise_inline_asm'}}
    %r = tt.elementwise_inline_asm "ex2.approx.ftz.f32 $0, $1;" {constraints = "=r,r", packed_element = 1 : i32, pure = false} %a0 : f32 -> f32
    tt.store %out, %r : !tt.ptr<f32>
    tt.return
  }
}

// -----
module {
  tt.func public @reject_exp2_pack_two(%a0: f32, %out: !tt.ptr<f32>) {
    // expected-error@+1 {{failed to legalize operation 'tt.elementwise_inline_asm'}}
    %r = tt.elementwise_inline_asm "ex2.approx.ftz.f32 $0, $1;" {constraints = "=r,r", packed_element = 2 : i32, pure = true} %a0 : f32 -> f32
    tt.store %out, %r : !tt.ptr<f32>
    tt.return
  }
}

// -----
module {
  tt.func public @reject_exp2_extra_instruction(%a0: f32, %out: !tt.ptr<f32>) {
    // expected-error@+1 {{failed to legalize operation 'tt.elementwise_inline_asm'}}
    %r = tt.elementwise_inline_asm "ex2.approx.ftz.f32 $0, $1; neg.f32 $0, $0;" {constraints = "=r,r", packed_element = 1 : i32, pure = true} %a0 : f32 -> f32
    tt.store %out, %r : !tt.ptr<f32>
    tt.return
  }
}

// -----
module {
  tt.func public @reject_fp4_observable_i32(%a0: f32, %a1: f32, %out: !tt.ptr<i32>) {
    // expected-error@+1 {{failed to legalize operation 'tt.elementwise_inline_asm'}}
    %r = tt.elementwise_inline_asm "{ .reg .b8 r; cvt.rn.satfinite.e2m1x2.f32 r, $1, $2; mov.b32 $0, {r, r, r, r}; }" {constraints = "=r,f,f", packed_element = 1 : i32, pure = true} %a0, %a1 : f32, f32 -> i32
    tt.store %out, %r : !tt.ptr<i32>
    tt.return
  }
}

// -----
module {
  tt.func public @reject_fp4_different_low_byte(%a0: f32, %a1: f32, %out: !tt.ptr<i8>) {
    // expected-error@+1 {{failed to legalize operation 'tt.elementwise_inline_asm'}}
    %r = tt.elementwise_inline_asm "{ .reg .b8 r; cvt.rn.satfinite.e2m1x2.f32 r, $1, $2; mov.b32 $0, {0, r, r, r}; }" {constraints = "=r,f,f", packed_element = 1 : i32, pure = true} %a0, %a1 : f32, f32 -> i8
    tt.store %out, %r : !tt.ptr<i8>
    tt.return
  }
}

// -----
module {
  tt.func public @reject_tf32_other_rounding(%a0: f32, %out: !tt.ptr<f32>) {
    // expected-error@+1 {{failed to legalize operation 'tt.elementwise_inline_asm'}}
    %r = tt.elementwise_inline_asm "cvt.rna.tf32.f32 $0, $1;" {constraints = "=r,r", packed_element = 1 : i32, pure = true} %a0 : f32 -> f32
    tt.store %out, %r : !tt.ptr<f32>
    tt.return
  }
}

// -----
module {
  tt.func public @reject_max_without_nan(%a0: f32, %a1: f32, %out: !tt.ptr<f32>) {
    // expected-error@+1 {{failed to legalize operation 'tt.elementwise_inline_asm'}}
    %r = tt.elementwise_inline_asm "{max.xorsign.abs.f32 $0, $1, $2;}" {constraints = "=r,r,r", packed_element = 1 : i32, pure = true} %a0, %a1 : f32, f32 -> f32
    tt.store %out, %r : !tt.ptr<f32>
    tt.return
  }
}

// -----
// This i32 is a pair of half bit patterns, not an integer conversion result.
module {
  tt.func public @asm_fp4_upcast_pair(%input: i8, %out: !tt.ptr<i32>) {
    %x = tt.splat %input : i8 -> tensor<2x4xi8>
    %r = tt.elementwise_inline_asm "{ .reg .b8 in_8; .reg .f16x2 out; cvt.u8.u32 in_8, $1; cvt.rn.f16x2.e2m1x2 out, in_8; mov.b32 $0, out; }" {constraints = "=r,r", packed_element = 1 : i32, pure = true} %x : tensor<2x4xi8> -> tensor<2x4xi32>
    %p = tt.splat %out : !tt.ptr<i32> -> tensor<2x4x!tt.ptr<i32>>
    tt.store %p, %r : tensor<2x4x!tt.ptr<i32>>
    tt.return
  }
}
// ASM-LABEL: entry @asm_fp4_upcast_pair
// ASM: %[[BYTES:.*]] = reshape {{.*}} : tile<2x4xi8> -> tile<8xi8>
// ASM: %[[F4:.*]] = unpack %[[BYTES]] : tile<8xi8> -> tile<16xf4E2M1FN>
// ASM: %[[HALVES:.*]] = ftof %[[F4]] : tile<16xf4E2M1FN> -> tile<16xf16>
// ASM: %[[HALF_BYTES:.*]] = pack %[[HALVES]] : tile<16xf16> -> tile<32xi8>
// ASM: %[[WORDS:.*]] = unpack %[[HALF_BYTES]] : tile<32xi8> -> tile<8xi32>
// ASM: reshape %[[WORDS]] : tile<8xi32> -> tile<2x4xi32>
// ASM-NOT: elementwise_inline_asm

// -----
module {
  tt.func public @reject_fp4_upcast_impure(%input: i8, %out: !tt.ptr<i32>) {
    // expected-error@+1 {{failed to legalize operation 'tt.elementwise_inline_asm'}}
    %r = tt.elementwise_inline_asm "{ .reg .b8 in_8; .reg .f16x2 out; cvt.u8.u32 in_8, $1; cvt.rn.f16x2.e2m1x2 out, in_8; mov.b32 $0, out; }" {constraints = "=r,r", packed_element = 1 : i32, pure = false} %input : i8 -> i32
    tt.store %out, %r : !tt.ptr<i32>
    tt.return
  }
}

// -----
module {
  tt.func public @reject_fp4_upcast_pack_two(%input: i8, %out: !tt.ptr<i32>) {
    // expected-error@+1 {{failed to legalize operation 'tt.elementwise_inline_asm'}}
    %r = tt.elementwise_inline_asm "{ .reg .b8 in_8; .reg .f16x2 out; cvt.u8.u32 in_8, $1; cvt.rn.f16x2.e2m1x2 out, in_8; mov.b32 $0, out; }" {constraints = "=r,r", packed_element = 2 : i32, pure = true} %input : i8 -> i32
    tt.store %out, %r : !tt.ptr<i32>
    tt.return
  }
}

// -----
module {
  tt.func public @reject_fp4_upcast_wide_input(%input: i32, %out: !tt.ptr<i32>) {
    // expected-error@+1 {{failed to legalize operation 'tt.elementwise_inline_asm'}}
    %r = tt.elementwise_inline_asm "{ .reg .b8 in_8; .reg .f16x2 out; cvt.u8.u32 in_8, $1; cvt.rn.f16x2.e2m1x2 out, in_8; mov.b32 $0, out; }" {constraints = "=r,r", packed_element = 1 : i32, pure = true} %input : i32 -> i32
    tt.store %out, %r : !tt.ptr<i32>
    tt.return
  }
}

// -----
module {
  tt.func public @reject_fp4_upcast_float_result(%input: i8, %out: !tt.ptr<f32>) {
    // expected-error@+1 {{failed to legalize operation 'tt.elementwise_inline_asm'}}
    %r = tt.elementwise_inline_asm "{ .reg .b8 in_8; .reg .f16x2 out; cvt.u8.u32 in_8, $1; cvt.rn.f16x2.e2m1x2 out, in_8; mov.b32 $0, out; }" {constraints = "=r,r", packed_element = 1 : i32, pure = true} %input : i8 -> f32
    tt.store %out, %r : !tt.ptr<f32>
    tt.return
  }
}

// -----
module {
  tt.func public @reject_fp4_upcast_constraints(%input: i8, %out: !tt.ptr<i32>) {
    // expected-error@+1 {{failed to legalize operation 'tt.elementwise_inline_asm'}}
    %r = tt.elementwise_inline_asm "{ .reg .b8 in_8; .reg .f16x2 out; cvt.u8.u32 in_8, $1; cvt.rn.f16x2.e2m1x2 out, in_8; mov.b32 $0, out; }" {constraints = "=r,r,~{memory}", packed_element = 1 : i32, pure = true} %input : i8 -> i32
    tt.store %out, %r : !tt.ptr<i32>
    tt.return
  }
}

// -----
module {
  tt.func public @reject_fp4_upcast_relu(%input: i8, %out: !tt.ptr<i32>) {
    // expected-error@+1 {{failed to legalize operation 'tt.elementwise_inline_asm'}}
    %r = tt.elementwise_inline_asm "{ .reg .b8 in_8; .reg .f16x2 out; cvt.u8.u32 in_8, $1; cvt.rn.relu.f16x2.e2m1x2 out, in_8; mov.b32 $0, out; }" {constraints = "=r,r", packed_element = 1 : i32, pure = true} %input : i8 -> i32
    tt.store %out, %r : !tt.ptr<i32>
    tt.return
  }
}

// -----
// NATIVE-LABEL: entry @gather_4_4_8_4_0_f32_i64_computed
// NATIVE: %[[GATHER_VALUES:.*]] = addf
// NATIVE: %[[GATHER_BITS:.*]] = bitcast %[[GATHER_VALUES]] : tile<4x4xf32> -> tile<4x4xi32>
// NATIVE: extract %[[GATHER_BITS]]
// NATIVE: loop iter_values
// NATIVE: cmpi equal
// NATIVE: exti {{.*}} unsigned : tile<i32> -> tile<i64>
// NATIVE: cmpi equal {{.*}} : tile<8x4xi64> -> tile<8x4xi1>
// NATIVE: extract %[[GATHER_BITS]]
// NATIVE: select {{.*}} : tile<8x4xi1>, tile<8x4xi32>
// NATIVE: continue
// NATIVE: bitcast {{.*}} : tile<8x4xi32> -> tile<8x4xf32>
module {
  tt.func public @gather_4_4_8_4_0_f32_i64_computed(%input: !tt.ptr<f32>, %indices: !tt.ptr<i64>, %output: !tt.ptr<f32>) {
    %source_offsets = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %output_offsets = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %source_base = tt.splat %input : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %index_base = tt.splat %indices : !tt.ptr<i64> -> tensor<32x!tt.ptr<i64>>
    %output_base = tt.splat %output : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %source_ptrs = tt.addptr %source_base, %source_offsets : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %index_ptrs = tt.addptr %index_base, %output_offsets : tensor<32x!tt.ptr<i64>>, tensor<32xi32>
    %output_ptrs = tt.addptr %output_base, %output_offsets : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %flat_loaded = tt.load %source_ptrs : tensor<16x!tt.ptr<f32>>
    %flat_indices = tt.load %index_ptrs : tensor<32x!tt.ptr<i64>>
    %loaded = tt.reshape %flat_loaded : tensor<16xf32> -> tensor<4x4xf32>
    %idx = tt.reshape %flat_indices : tensor<32xi64> -> tensor<8x4xi64>
    %values = arith.addf %loaded, %loaded : tensor<4x4xf32>
    %result = tt.gather %values[%idx] {axis = 0 : i32} : (tensor<4x4xf32>, tensor<8x4xi64>) -> tensor<8x4xf32>
    %flat_result = tt.reshape %result : tensor<8x4xf32> -> tensor<32xf32>
    tt.store %output_ptrs, %flat_result : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// NATIVE-LABEL: entry @gather_512_4_0_i64_i8_raw
// NATIVE: exti {{.*}} unsigned : tile<4xi8> -> tile<4xi32>
// NATIVE: loop iter_values
// NATIVE: cmpi equal {{.*}} : tile<4xi32> -> tile<4xi1>
// NATIVE: select {{.*}} : tile<4xi1>, tile<4xi64>
// NATIVE-NOT: trunci
// NATIVE: return
module {
  tt.func public @gather_512_4_0_i64_i8_raw(%input: !tt.ptr<i64>, %indices: !tt.ptr<i8>, %output: !tt.ptr<i64>) {
    %source_offsets = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32>
    %output_offsets = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %source_base = tt.splat %input : !tt.ptr<i64> -> tensor<512x!tt.ptr<i64>>
    %index_base = tt.splat %indices : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
    %output_base = tt.splat %output : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %source_ptrs = tt.addptr %source_base, %source_offsets : tensor<512x!tt.ptr<i64>>, tensor<512xi32>
    %index_ptrs = tt.addptr %index_base, %output_offsets : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
    %output_ptrs = tt.addptr %output_base, %output_offsets : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %flat_loaded = tt.load %source_ptrs : tensor<512x!tt.ptr<i64>>
    %flat_indices = tt.load %index_ptrs : tensor<4x!tt.ptr<i8>>
    %loaded = tt.reshape %flat_loaded : tensor<512xi64> -> tensor<512xi64>
    %idx = tt.reshape %flat_indices : tensor<4xi8> -> tensor<4xi8>
    %values = tt.reshape %loaded : tensor<512xi64> -> tensor<512xi64>
    %result = tt.gather %values[%idx] {axis = 0 : i32} : (tensor<512xi64>, tensor<4xi8>) -> tensor<4xi64>
    %flat_result = tt.reshape %result : tensor<4xi64> -> tensor<4xi64>
    tt.store %output_ptrs, %flat_result : tensor<4x!tt.ptr<i64>>
    tt.return
  }
}

// -----
// NATIVE-LABEL: entry @gather_4_1_4_8_1_f16_i32_raw
// NATIVE: bitcast {{.*}} : tile<4x1xf16> -> tile<4x1xi16>
// NATIVE: broadcast {{.*}} : tile<4x1xi16> -> tile<4x8xi16>
// NATIVE: bitcast {{.*}} : tile<4x8xi16> -> tile<4x8xf16>
// NATIVE-NOT: loop
// NATIVE-NOT: cmpi
// NATIVE: return
module {
  tt.func public @gather_4_1_4_8_1_f16_i32_raw(%input: !tt.ptr<f16>, %indices: !tt.ptr<i32>, %output: !tt.ptr<f16>) {
    %source_offsets = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %output_offsets = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %source_base = tt.splat %input : !tt.ptr<f16> -> tensor<4x!tt.ptr<f16>>
    %index_base = tt.splat %indices : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>>
    %output_base = tt.splat %output : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
    %source_ptrs = tt.addptr %source_base, %source_offsets : tensor<4x!tt.ptr<f16>>, tensor<4xi32>
    %index_ptrs = tt.addptr %index_base, %output_offsets : tensor<32x!tt.ptr<i32>>, tensor<32xi32>
    %output_ptrs = tt.addptr %output_base, %output_offsets : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
    %flat_loaded = tt.load %source_ptrs : tensor<4x!tt.ptr<f16>>
    %flat_indices = tt.load %index_ptrs : tensor<32x!tt.ptr<i32>>
    %loaded = tt.reshape %flat_loaded : tensor<4xf16> -> tensor<4x1xf16>
    %idx = tt.reshape %flat_indices : tensor<32xi32> -> tensor<4x8xi32>
    %values = tt.reshape %loaded : tensor<4x1xf16> -> tensor<4x1xf16>
    %result = tt.gather %values[%idx] {axis = 1 : i32} : (tensor<4x1xf16>, tensor<4x8xi32>) -> tensor<4x8xf16>
    %flat_result = tt.reshape %result : tensor<4x8xf16> -> tensor<32xf16>
    tt.store %output_ptrs, %flat_result : tensor<32x!tt.ptr<f16>>
    tt.return
  }
}

// -----

module {
  tt.func public @unit_scale_e4m3_lhs(%a_ptr: !tt.ptr<f8E4M3FN>, %b_ptr: !tt.ptr<f8E4M3FN>, %c_ptr: !tt.ptr<f32>, %sa_ptr: !tt.ptr<i8>) {
    %a_range = tt.make_range {start = 0 : i32, end = 2048 : i32} : tensor<2048xi32>
    %a_offset = tt.reshape %a_range : tensor<2048xi32> -> tensor<32x64xi32>
    %a_base = tt.splat %a_ptr : !tt.ptr<f8E4M3FN> -> tensor<32x64x!tt.ptr<f8E4M3FN>>
    %a_ptrs = tt.addptr %a_base, %a_offset : tensor<32x64x!tt.ptr<f8E4M3FN>>, tensor<32x64xi32>
    %a = tt.load %a_ptrs : tensor<32x64x!tt.ptr<f8E4M3FN>>
    %b_range = tt.make_range {start = 0 : i32, end = 4096 : i32} : tensor<4096xi32>
    %b_offset = tt.reshape %b_range : tensor<4096xi32> -> tensor<64x64xi32>
    %b_base = tt.splat %b_ptr : !tt.ptr<f8E4M3FN> -> tensor<64x64x!tt.ptr<f8E4M3FN>>
    %b_ptrs = tt.addptr %b_base, %b_offset : tensor<64x64x!tt.ptr<f8E4M3FN>>, tensor<64x64xi32>
    %b = tt.load %b_ptrs : tensor<64x64x!tt.ptr<f8E4M3FN>>
    %c_range = tt.make_range {start = 0 : i32, end = 2048 : i32} : tensor<2048xi32>
    %c_offset = tt.reshape %c_range : tensor<2048xi32> -> tensor<32x64xi32>
    %c_base = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>>
    %c_ptrs = tt.addptr %c_base, %c_offset : tensor<32x64x!tt.ptr<f32>>, tensor<32x64xi32>
    %sa_range = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %sa_offset = tt.reshape %sa_range : tensor<64xi32> -> tensor<32x2xi32>
    %sa_base = tt.splat %sa_ptr : !tt.ptr<i8> -> tensor<32x2x!tt.ptr<i8>>
    %sa_ptrs = tt.addptr %sa_base, %sa_offset : tensor<32x2x!tt.ptr<i8>>, tensor<32x2xi32>
    %sa = tt.load %sa_ptrs : tensor<32x2x!tt.ptr<i8>>
    %zero = arith.constant dense<0.0> : tensor<32x64xf32>
    %result = tt.dot_scaled %a scale %sa, %b, %zero lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<32x64xf8E4M3FN>, tensor<32x2xi8> * tensor<64x64xf8E4M3FN> -> tensor<32x64xf32>
    tt.store %c_ptrs, %result : tensor<32x64x!tt.ptr<f32>>
    tt.return
  }
}

// NATIVE-LABEL: entry @unit_scale_e4m3_lhs
// NATIVE: %[[UNIT_unit_scale_e4m3_lhs:.*]] = constant <i8: 127> : tile<64x2xi8>
// NATIVE: bitcast %[[UNIT_unit_scale_e4m3_lhs]] : tile<64x2xi8> -> tile<64x2xf8E8M0FNU>
// NATIVE: mmaf_scaled
// NATIVE-NOT: mulf
// NATIVE: return

// -----

module {
  tt.func public @unit_scale_e5m2_rhs(%a_ptr: !tt.ptr<f8E5M2>, %b_ptr: !tt.ptr<f8E5M2>, %c_ptr: !tt.ptr<f32>, %sb_ptr: !tt.ptr<i8>) {
    %a_range = tt.make_range {start = 0 : i32, end = 2048 : i32} : tensor<2048xi32>
    %a_offset = tt.reshape %a_range : tensor<2048xi32> -> tensor<32x64xi32>
    %a_base = tt.splat %a_ptr : !tt.ptr<f8E5M2> -> tensor<32x64x!tt.ptr<f8E5M2>>
    %a_ptrs = tt.addptr %a_base, %a_offset : tensor<32x64x!tt.ptr<f8E5M2>>, tensor<32x64xi32>
    %a = tt.load %a_ptrs : tensor<32x64x!tt.ptr<f8E5M2>>
    %b_range = tt.make_range {start = 0 : i32, end = 4096 : i32} : tensor<4096xi32>
    %b_offset = tt.reshape %b_range : tensor<4096xi32> -> tensor<64x64xi32>
    %b_base = tt.splat %b_ptr : !tt.ptr<f8E5M2> -> tensor<64x64x!tt.ptr<f8E5M2>>
    %b_ptrs = tt.addptr %b_base, %b_offset : tensor<64x64x!tt.ptr<f8E5M2>>, tensor<64x64xi32>
    %b = tt.load %b_ptrs : tensor<64x64x!tt.ptr<f8E5M2>>
    %c_range = tt.make_range {start = 0 : i32, end = 2048 : i32} : tensor<2048xi32>
    %c_offset = tt.reshape %c_range : tensor<2048xi32> -> tensor<32x64xi32>
    %c_base = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>>
    %c_ptrs = tt.addptr %c_base, %c_offset : tensor<32x64x!tt.ptr<f32>>, tensor<32x64xi32>
    %sb_range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %sb_offset = tt.reshape %sb_range : tensor<128xi32> -> tensor<64x2xi32>
    %sb_base = tt.splat %sb_ptr : !tt.ptr<i8> -> tensor<64x2x!tt.ptr<i8>>
    %sb_ptrs = tt.addptr %sb_base, %sb_offset : tensor<64x2x!tt.ptr<i8>>, tensor<64x2xi32>
    %sb = tt.load %sb_ptrs : tensor<64x2x!tt.ptr<i8>>
    %zero = arith.constant dense<0.0> : tensor<32x64xf32>
    %result = tt.dot_scaled %a, %b scale %sb, %zero lhs = e5m2 rhs = e5m2 {fastMath = false} : tensor<32x64xf8E5M2> * tensor<64x64xf8E5M2>, tensor<64x2xi8> -> tensor<32x64xf32>
    tt.store %c_ptrs, %result : tensor<32x64x!tt.ptr<f32>>
    tt.return
  }
}

// NATIVE-LABEL: entry @unit_scale_e5m2_rhs
// NATIVE: %[[UNIT_unit_scale_e5m2_rhs:.*]] = constant <i8: 127> : tile<32x2xi8>
// NATIVE: bitcast %[[UNIT_unit_scale_e5m2_rhs]] : tile<32x2xi8> -> tile<32x2xf8E8M0FNU>
// NATIVE: mmaf_scaled
// NATIVE-NOT: mulf
// NATIVE: return

// -----

module {
  tt.func public @unit_scale_batched_e5m2_rhs(%a_ptr: !tt.ptr<f8E5M2>, %b_ptr: !tt.ptr<f8E5M2>, %c_ptr: !tt.ptr<f32>, %sb_ptr: !tt.ptr<i8>) {
    %a_range = tt.make_range {start = 0 : i32, end = 16384 : i32} : tensor<16384xi32>
    %a_offset = tt.reshape %a_range : tensor<16384xi32> -> tensor<2x64x128xi32>
    %a_base = tt.splat %a_ptr : !tt.ptr<f8E5M2> -> tensor<2x64x128x!tt.ptr<f8E5M2>>
    %a_ptrs = tt.addptr %a_base, %a_offset : tensor<2x64x128x!tt.ptr<f8E5M2>>, tensor<2x64x128xi32>
    %a = tt.load %a_ptrs : tensor<2x64x128x!tt.ptr<f8E5M2>>
    %b_range = tt.make_range {start = 0 : i32, end = 8192 : i32} : tensor<8192xi32>
    %b_offset = tt.reshape %b_range : tensor<8192xi32> -> tensor<2x128x32xi32>
    %b_base = tt.splat %b_ptr : !tt.ptr<f8E5M2> -> tensor<2x128x32x!tt.ptr<f8E5M2>>
    %b_ptrs = tt.addptr %b_base, %b_offset : tensor<2x128x32x!tt.ptr<f8E5M2>>, tensor<2x128x32xi32>
    %b = tt.load %b_ptrs : tensor<2x128x32x!tt.ptr<f8E5M2>>
    %c_range = tt.make_range {start = 0 : i32, end = 4096 : i32} : tensor<4096xi32>
    %c_offset = tt.reshape %c_range : tensor<4096xi32> -> tensor<2x64x32xi32>
    %c_base = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<2x64x32x!tt.ptr<f32>>
    %c_ptrs = tt.addptr %c_base, %c_offset : tensor<2x64x32x!tt.ptr<f32>>, tensor<2x64x32xi32>
    %sb_range = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
    %sb_offset = tt.reshape %sb_range : tensor<256xi32> -> tensor<2x32x4xi32>
    %sb_base = tt.splat %sb_ptr : !tt.ptr<i8> -> tensor<2x32x4x!tt.ptr<i8>>
    %sb_ptrs = tt.addptr %sb_base, %sb_offset : tensor<2x32x4x!tt.ptr<i8>>, tensor<2x32x4xi32>
    %sb = tt.load %sb_ptrs : tensor<2x32x4x!tt.ptr<i8>>
    %zero = arith.constant dense<0.0> : tensor<2x64x32xf32>
    %result = tt.dot_scaled %a, %b scale %sb, %zero lhs = e5m2 rhs = e5m2 {fastMath = false} : tensor<2x64x128xf8E5M2> * tensor<2x128x32xf8E5M2>, tensor<2x32x4xi8> -> tensor<2x64x32xf32>
    tt.store %c_ptrs, %result : tensor<2x64x32x!tt.ptr<f32>>
    tt.return
  }
}

// NATIVE-LABEL: entry @unit_scale_batched_e5m2_rhs
// NATIVE: %[[UNIT_unit_scale_batched_e5m2_rhs:.*]] = constant <i8: 127> : tile<2x64x4xi8>
// NATIVE: bitcast %[[UNIT_unit_scale_batched_e5m2_rhs]] : tile<2x64x4xi8> -> tile<2x64x4xf8E8M0FNU>
// NATIVE: mmaf_scaled
// NATIVE-NOT: mulf
// NATIVE: return

// -----

module {
  tt.func public @both_missing(%a_ptr: !tt.ptr<f8E4M3FN>, %b_ptr: !tt.ptr<f8E4M3FN>, %c_ptr: !tt.ptr<f32>) {
    %a_range = tt.make_range {start = 0 : i32, end = 2048 : i32} : tensor<2048xi32>
    %a_offset = tt.reshape %a_range : tensor<2048xi32> -> tensor<32x64xi32>
    %a_base = tt.splat %a_ptr : !tt.ptr<f8E4M3FN> -> tensor<32x64x!tt.ptr<f8E4M3FN>>
    %a_ptrs = tt.addptr %a_base, %a_offset : tensor<32x64x!tt.ptr<f8E4M3FN>>, tensor<32x64xi32>
    %a = tt.load %a_ptrs : tensor<32x64x!tt.ptr<f8E4M3FN>>
    %b_range = tt.make_range {start = 0 : i32, end = 4096 : i32} : tensor<4096xi32>
    %b_offset = tt.reshape %b_range : tensor<4096xi32> -> tensor<64x64xi32>
    %b_base = tt.splat %b_ptr : !tt.ptr<f8E4M3FN> -> tensor<64x64x!tt.ptr<f8E4M3FN>>
    %b_ptrs = tt.addptr %b_base, %b_offset : tensor<64x64x!tt.ptr<f8E4M3FN>>, tensor<64x64xi32>
    %b = tt.load %b_ptrs : tensor<64x64x!tt.ptr<f8E4M3FN>>
    %c_range = tt.make_range {start = 0 : i32, end = 2048 : i32} : tensor<2048xi32>
    %c_offset = tt.reshape %c_range : tensor<2048xi32> -> tensor<32x64xi32>
    %c_base = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>>
    %c_ptrs = tt.addptr %c_base, %c_offset : tensor<32x64x!tt.ptr<f32>>, tensor<32x64xi32>
    %zero = arith.constant dense<0.0> : tensor<32x64xf32>
    // expected-error @below {{failed to legalize operation 'tt.dot_scaled'}}
    %result = tt.dot_scaled %a, %b, %zero lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<32x64xf8E4M3FN> * tensor<64x64xf8E4M3FN> -> tensor<32x64xf32>
    tt.store %c_ptrs, %result : tensor<32x64x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
  tt.func public @single_fp4(%a_ptr: !tt.ptr<i8>, %b_ptr: !tt.ptr<i8>, %c_ptr: !tt.ptr<f32>, %sa_ptr: !tt.ptr<i8>) {
    %a_range = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32>
    %a_offset = tt.reshape %a_range : tensor<1024xi32> -> tensor<32x32xi32>
    %a_base = tt.splat %a_ptr : !tt.ptr<i8> -> tensor<32x32x!tt.ptr<i8>>
    %a_ptrs = tt.addptr %a_base, %a_offset : tensor<32x32x!tt.ptr<i8>>, tensor<32x32xi32>
    %a = tt.load %a_ptrs : tensor<32x32x!tt.ptr<i8>>
    %b_range = tt.make_range {start = 0 : i32, end = 2048 : i32} : tensor<2048xi32>
    %b_offset = tt.reshape %b_range : tensor<2048xi32> -> tensor<32x64xi32>
    %b_base = tt.splat %b_ptr : !tt.ptr<i8> -> tensor<32x64x!tt.ptr<i8>>
    %b_ptrs = tt.addptr %b_base, %b_offset : tensor<32x64x!tt.ptr<i8>>, tensor<32x64xi32>
    %b = tt.load %b_ptrs : tensor<32x64x!tt.ptr<i8>>
    %c_range = tt.make_range {start = 0 : i32, end = 2048 : i32} : tensor<2048xi32>
    %c_offset = tt.reshape %c_range : tensor<2048xi32> -> tensor<32x64xi32>
    %c_base = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>>
    %c_ptrs = tt.addptr %c_base, %c_offset : tensor<32x64x!tt.ptr<f32>>, tensor<32x64xi32>
    %sa_range = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %sa_offset = tt.reshape %sa_range : tensor<64xi32> -> tensor<32x2xi32>
    %sa_base = tt.splat %sa_ptr : !tt.ptr<i8> -> tensor<32x2x!tt.ptr<i8>>
    %sa_ptrs = tt.addptr %sa_base, %sa_offset : tensor<32x2x!tt.ptr<i8>>, tensor<32x2xi32>
    %sa = tt.load %sa_ptrs : tensor<32x2x!tt.ptr<i8>>
    %zero = arith.constant dense<0.0> : tensor<32x64xf32>
    // expected-error @below {{failed to legalize operation 'tt.dot_scaled'}}
    %result = tt.dot_scaled %a scale %sa, %b, %zero lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<32x32xi8>, tensor<32x2xi8> * tensor<32x64xi8> -> tensor<32x64xf32>
    tt.store %c_ptrs, %result : tensor<32x64x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
  tt.func public @mixed_fp8(%a_ptr: !tt.ptr<f8E4M3FN>, %b_ptr: !tt.ptr<f8E5M2>, %c_ptr: !tt.ptr<f32>, %sa_ptr: !tt.ptr<i8>) {
    %a_range = tt.make_range {start = 0 : i32, end = 2048 : i32} : tensor<2048xi32>
    %a_offset = tt.reshape %a_range : tensor<2048xi32> -> tensor<32x64xi32>
    %a_base = tt.splat %a_ptr : !tt.ptr<f8E4M3FN> -> tensor<32x64x!tt.ptr<f8E4M3FN>>
    %a_ptrs = tt.addptr %a_base, %a_offset : tensor<32x64x!tt.ptr<f8E4M3FN>>, tensor<32x64xi32>
    %a = tt.load %a_ptrs : tensor<32x64x!tt.ptr<f8E4M3FN>>
    %b_range = tt.make_range {start = 0 : i32, end = 4096 : i32} : tensor<4096xi32>
    %b_offset = tt.reshape %b_range : tensor<4096xi32> -> tensor<64x64xi32>
    %b_base = tt.splat %b_ptr : !tt.ptr<f8E5M2> -> tensor<64x64x!tt.ptr<f8E5M2>>
    %b_ptrs = tt.addptr %b_base, %b_offset : tensor<64x64x!tt.ptr<f8E5M2>>, tensor<64x64xi32>
    %b = tt.load %b_ptrs : tensor<64x64x!tt.ptr<f8E5M2>>
    %c_range = tt.make_range {start = 0 : i32, end = 2048 : i32} : tensor<2048xi32>
    %c_offset = tt.reshape %c_range : tensor<2048xi32> -> tensor<32x64xi32>
    %c_base = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>>
    %c_ptrs = tt.addptr %c_base, %c_offset : tensor<32x64x!tt.ptr<f32>>, tensor<32x64xi32>
    %sa_range = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %sa_offset = tt.reshape %sa_range : tensor<64xi32> -> tensor<32x2xi32>
    %sa_base = tt.splat %sa_ptr : !tt.ptr<i8> -> tensor<32x2x!tt.ptr<i8>>
    %sa_ptrs = tt.addptr %sa_base, %sa_offset : tensor<32x2x!tt.ptr<i8>>, tensor<32x2xi32>
    %sa = tt.load %sa_ptrs : tensor<32x2x!tt.ptr<i8>>
    %zero = arith.constant dense<0.0> : tensor<32x64xf32>
    // expected-error @below {{failed to legalize operation 'tt.dot_scaled'}}
    %result = tt.dot_scaled %a scale %sa, %b, %zero lhs = e4m3 rhs = e5m2 {fastMath = false} : tensor<32x64xf8E4M3FN>, tensor<32x2xi8> * tensor<64x64xf8E5M2> -> tensor<32x64xf32>
    tt.store %c_ptrs, %result : tensor<32x64x!tt.ptr<f32>>
    tt.return
  }
}

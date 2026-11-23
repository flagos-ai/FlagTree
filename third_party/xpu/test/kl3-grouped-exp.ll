; RUN: opt -load-pass-plugin=%plugin -passes=kl3-grouped-exp,kl3-grouped-exp,verify -S %s | FileCheck %s
; RUN: opt -mtriple=xpu2-baidu-none-gnu -load-pass-plugin=%plugin -passes=kl3-grouped-exp,verify -S %s | FileCheck %s --check-prefix=KL2 --implicit-check-not="asm sideeffect"
target triple = "xpu3-baidu-none-gnu"

declare <16 x float> @llvm.exp.v16f32(<16 x float>)
declare <8 x float> @llvm.exp.v8f32(<8 x float>)
declare <16 x float> @llvm.exp2.v16f32(<16 x float>)

; CHECK-LABEL: define void @batch(
; CHECK: [[S:%[^ ]+]] = alloca <16 x float>, i32 4, align 64
; CHECK-NOT: alloca
; CHECK: store volatile <16 x float> %a
; CHECK: store volatile <16 x float> %b
; CHECK: store volatile <16 x float> %c
; CHECK: store volatile <16 x float> %d
; CHECK: asm sideeffect "mfence { lm }
; CHECK-SAME: loadli.w $15, 252($16)
; CHECK-SAME: storeli.w $15, 252($16)
; CHECK: load volatile <16 x float>
; CHECK: store volatile <16 x float> %e
; CHECK: asm sideeffect "mfence { lm }
; CHECK-SAME: storeli.w $15, 60($16)
; CHECK-NOT: asm sideeffect
; CHECK: ret void
; KL2-LABEL: define void @batch(
; KL2-COUNT-5: call <16 x float> @llvm.exp.v16f32
define void @batch(ptr %out, <16 x float> %a, <16 x float> %b, <16 x float> %c, <16 x float> %d, <16 x float> %e) {
  %a1 = call <16 x float> @llvm.exp.v16f32(<16 x float> %a)
  %b1 = call <16 x float> @llvm.exp.v16f32(<16 x float> %b)
  %c1 = call <16 x float> @llvm.exp.v16f32(<16 x float> %c)
  %d1 = call <16 x float> @llvm.exp.v16f32(<16 x float> %d)
  %e1 = call <16 x float> @llvm.exp.v16f32(<16 x float> %e)
  store volatile <16 x float> %a1, ptr %out
  store volatile <16 x float> %b1, ptr %out
  store volatile <16 x float> %c1, ptr %out
  store volatile <16 x float> %d1, ptr %out
  store volatile <16 x float> %e1, ptr %out
  ret void
}

; CHECK-LABEL: define <16 x float> @dependent(
; CHECK: asm sideeffect
; CHECK: [[PREV:%[^ ]+]] = load volatile <16 x float>
; CHECK: store volatile <16 x float> [[PREV]],
; CHECK: asm sideeffect
; CHECK: ret
define <16 x float> @dependent(<16 x float> %x) {
  %a = call <16 x float> @llvm.exp.v16f32(<16 x float> %x)
  %b = call <16 x float> @llvm.exp.v16f32(<16 x float> %a)
  ret <16 x float> %b
}

; CHECK-LABEL: define <16 x float> @base_two(
; CHECK-NEXT: %r = call <16 x float> @llvm.exp2.v16f32
define <16 x float> @base_two(<16 x float> %x) {
  %r = call <16 x float> @llvm.exp2.v16f32(<16 x float> %x)
  ret <16 x float> %r
}

; CHECK-LABEL: define <8 x float> @narrow(
; CHECK-NEXT: %r = call <8 x float> @llvm.exp.v8f32
define <8 x float> @narrow(<8 x float> %x) {
  %r = call <8 x float> @llvm.exp.v8f32(<8 x float> %x)
  ret <8 x float> %r
}

; CHECK-LABEL: define <16 x float> @sdnn(
; CHECK-NEXT: %r = call <16 x float> @llvm.exp.v16f32
define <16 x float> @sdnn(<16 x float> %x) #0 {
  %r = call <16 x float> @llvm.exp.v16f32(<16 x float> %x)
  ret <16 x float> %r
}
attributes #0 = { "target-features"="+cdnn3" }

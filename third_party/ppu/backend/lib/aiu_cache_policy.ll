; Copyright 2025-     FlagOS Contributors
;
; Permission is hereby granted, free of charge, to any person obtaining a copy
; of this software and associated documentation files (the "Software"), to deal
; in the Software without restriction, including without limitation the rights
; to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
; copies of the Software, and to permit persons to whom the Software is
; furnished to do so, subject to the following conditions:
;
; The above copyright notice and this permission notice shall be included in all
; copies or substantial portions of the Software.
;
; THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
; IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
; FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
; AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
; LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
; OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
; SOFTWARE.

; PPU SDK 2.1 typed-pointer ABI bridge for PPU0010 AIU cache modes.
; The intrinsic signature is defined by the SDK's __aiu_load_tsm_params_2d.
; Only pointer conversion is performed here; all descriptor packing remains
; in the AIU lowering. llvm-as/llvm-link are the target SDK tools.
; kp1..kp6 only: kp7 is rejected by both the frontend and the lowering
; because GPU-produced input can be stale on SDK 2.1.
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-f16:16-f32:32-v16:16-v32:32-n16:32:64"

define linkonce_odr void @__flagtree_ppu_aiu_copy_kp1(i32 %shared, i64 %global, i32 %imm, i32 %count, i32 %x, i32 %extension, i32 %z, i32 %cube) alwaysinline convergent nounwind {
  %shared.ptr = inttoptr i32 %shared to i32 addrspace(3)*
  %global.ptr = inttoptr i64 %global to i32 addrspace(1)*
  call void @llvm.ppu.vmem.conv.ld.tsm.kp1(i32 addrspace(3)* %shared.ptr, i32 addrspace(1)* %global.ptr, i32 %imm, i32 %count, i32 %x, i32 %extension, i32 %z, i32 %cube)
  ret void
}
declare void @llvm.ppu.vmem.conv.ld.tsm.kp1(i32 addrspace(3)*, i32 addrspace(1)*, i32, i32, i32, i32, i32, i32) convergent nounwind


define linkonce_odr void @__flagtree_ppu_aiu_copy_kp2(i32 %shared, i64 %global, i32 %imm, i32 %count, i32 %x, i32 %extension, i32 %z, i32 %cube) alwaysinline convergent nounwind {
  %shared.ptr = inttoptr i32 %shared to i32 addrspace(3)*
  %global.ptr = inttoptr i64 %global to i32 addrspace(1)*
  call void @llvm.ppu.vmem.conv.ld.tsm.kp2(i32 addrspace(3)* %shared.ptr, i32 addrspace(1)* %global.ptr, i32 %imm, i32 %count, i32 %x, i32 %extension, i32 %z, i32 %cube)
  ret void
}
declare void @llvm.ppu.vmem.conv.ld.tsm.kp2(i32 addrspace(3)*, i32 addrspace(1)*, i32, i32, i32, i32, i32, i32) convergent nounwind


define linkonce_odr void @__flagtree_ppu_aiu_copy_kp3(i32 %shared, i64 %global, i32 %imm, i32 %count, i32 %x, i32 %extension, i32 %z, i32 %cube) alwaysinline convergent nounwind {
  %shared.ptr = inttoptr i32 %shared to i32 addrspace(3)*
  %global.ptr = inttoptr i64 %global to i32 addrspace(1)*
  call void @llvm.ppu.vmem.conv.ld.tsm.kp3(i32 addrspace(3)* %shared.ptr, i32 addrspace(1)* %global.ptr, i32 %imm, i32 %count, i32 %x, i32 %extension, i32 %z, i32 %cube)
  ret void
}
declare void @llvm.ppu.vmem.conv.ld.tsm.kp3(i32 addrspace(3)*, i32 addrspace(1)*, i32, i32, i32, i32, i32, i32) convergent nounwind


define linkonce_odr void @__flagtree_ppu_aiu_copy_kp4(i32 %shared, i64 %global, i32 %imm, i32 %count, i32 %x, i32 %extension, i32 %z, i32 %cube) alwaysinline convergent nounwind {
  %shared.ptr = inttoptr i32 %shared to i32 addrspace(3)*
  %global.ptr = inttoptr i64 %global to i32 addrspace(1)*
  call void @llvm.ppu.vmem.conv.ld.tsm.kp4(i32 addrspace(3)* %shared.ptr, i32 addrspace(1)* %global.ptr, i32 %imm, i32 %count, i32 %x, i32 %extension, i32 %z, i32 %cube)
  ret void
}
declare void @llvm.ppu.vmem.conv.ld.tsm.kp4(i32 addrspace(3)*, i32 addrspace(1)*, i32, i32, i32, i32, i32, i32) convergent nounwind


define linkonce_odr void @__flagtree_ppu_aiu_copy_kp5(i32 %shared, i64 %global, i32 %imm, i32 %count, i32 %x, i32 %extension, i32 %z, i32 %cube) alwaysinline convergent nounwind {
  %shared.ptr = inttoptr i32 %shared to i32 addrspace(3)*
  %global.ptr = inttoptr i64 %global to i32 addrspace(1)*
  call void @llvm.ppu.vmem.conv.ld.tsm.kp5(i32 addrspace(3)* %shared.ptr, i32 addrspace(1)* %global.ptr, i32 %imm, i32 %count, i32 %x, i32 %extension, i32 %z, i32 %cube)
  ret void
}
declare void @llvm.ppu.vmem.conv.ld.tsm.kp5(i32 addrspace(3)*, i32 addrspace(1)*, i32, i32, i32, i32, i32, i32) convergent nounwind


define linkonce_odr void @__flagtree_ppu_aiu_copy_kp6(i32 %shared, i64 %global, i32 %imm, i32 %count, i32 %x, i32 %extension, i32 %z, i32 %cube) alwaysinline convergent nounwind {
  %shared.ptr = inttoptr i32 %shared to i32 addrspace(3)*
  %global.ptr = inttoptr i64 %global to i32 addrspace(1)*
  call void @llvm.ppu.vmem.conv.ld.tsm.kp6(i32 addrspace(3)* %shared.ptr, i32 addrspace(1)* %global.ptr, i32 %imm, i32 %count, i32 %x, i32 %extension, i32 %z, i32 %cube)
  ret void
}
declare void @llvm.ppu.vmem.conv.ld.tsm.kp6(i32 addrspace(3)*, i32 addrspace(1)*, i32, i32, i32, i32, i32, i32) convergent nounwind

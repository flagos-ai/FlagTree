// Native instruction controls carried by the existing `cache` field.
#ifndef TRITON_NATIVE_CACHE_MODIFIERS
#define TRITON_NATIVE_CACHE_MODIFIERS
include "mlir/IR/EnumAttr.td"

defvar TT_NativeCacheModifiers = !listconcat(
  [I32EnumAttrCase<"PPU_LSU_DEFAULT", 11, "ppu_lsu_default">],
  !foreach(kp, [1, 2, 3, 4, 5, 6],
    I32EnumAttrCase<!strconcat("PPU_AIU_KP", !cast<string>(kp)),
                   !add(31, kp), !strconcat("ppu_aiu_kp", !cast<string>(kp))>),
  !listflatten(!foreach(inner, [0, 1, 2, 3, 4, 5],
    !listflatten(!foreach(outer, [0, 1, 2, 3, 4, 5],
      !foreach(l2, [0, 1],
        I32EnumAttrCase<
          !strconcat("MUSA_I", !cast<string>(inner), "_O", !cast<string>(outer), "_L", !cast<string>(l2)),
          !add(128, !mul(inner, 12), !mul(outer, 2), l2),
          !strconcat("musa_i", !cast<string>(inner), "_o", !cast<string>(outer), "_l", !cast<string>(l2))>))))));

// Keep NORMAL / EVICT_FIRST / EVICT_LAST and their legacy meaning unchanged.
defvar TT_NativeEvictionPolicies = [
  I32EnumAttrCase<"L1_EVICT_NORMAL", 4, "l1_evict_normal">,
  I32EnumAttrCase<"L1_EVICT_FIRST", 5, "l1_evict_first">,
  I32EnumAttrCase<"L1_EVICT_LAST", 6, "l1_evict_last">,
  I32EnumAttrCase<"L1_EVICT_UNCHANGED", 7, "l1_evict_unchanged">,
  I32EnumAttrCase<"L1_NO_ALLOCATE", 8, "l1_no_allocate">,
  I32EnumAttrCase<"L2_EVICT_NORMAL", 9, "l2_evict_normal">,
  I32EnumAttrCase<"L2_EVICT_FIRST", 10, "l2_evict_first">,
  I32EnumAttrCase<"L2_EVICT_LAST", 11, "l2_evict_last">,
  I32EnumAttrCase<"L2_EVICT_UNCHANGED", 12, "l2_evict_unchanged">
];
#endif

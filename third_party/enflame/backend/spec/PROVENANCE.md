# Enflame Backend Spec Overrides

These files are copies of shared Triton sources with Enflame-specific changes.
Keep each override synchronized with its shared counterpart and record the
intentional divergence below.

| Shared path | Enflame-specific change |
| --- | --- |
| `include/triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h` | `maybeDeduplicate` conservatively skips deduplication for non-power-of-two constancy reported by Enflame layouts. |

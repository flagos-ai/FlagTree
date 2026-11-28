# Ordered FP8 GLU cooperation

`tir.matmul_glu.cooperative_wgmma` is an optional SM90 implementation. Its
factory does not change the generic SM90 catalog or default selection.

The source is one complete scalar BF16 row, broadcast over the placement.
Gate/up FP8 weights have the same contiguous output-row ownership and complete
K128 groups, including multi-axis vector lanes along K. Both FP32 scale tables
are broadcast. The scalar BF16 result preserves the weight row ownership.
Partial/exclusive domains, split K, other activation quantization and native
encoded weights do not satisfy this contract.

Every CTA participates in a two-phase schedule:

1. Assign complete `(chip N64 tile, K128 block)` tasks over chip CTAs. Read gate
   and up rows through their actual canonical/compact physical ABI, retaining
   device coordinates while changing only block row-owner coordinates. Reuse
   the original SM90 activation quantization for both raw WGMMA products.
2. Publish both raw FP32 projections and the activation scale, then execute
   one full-grid barrier. Each original result owner merges exactly its real
   K blocks in increasing order and writes its original output rows.

The caller provides three invocation workspaces in chip-shared global memory:
`gate_partials[KB, Nchip]`, `up_partials[KB, Nchip]` and
`activation_scales[KB]`. They are contiguous scalar FP32 buffers. Only N tile
zero writes activation scales. Every live element is overwritten per call;
there is no cross-barrier register state or undeclared scratch.

Each merge computes `factor = mul_rn(A_scale, weight_scale)` and performs the
original FP32 `fma(raw, factor, accumulator)` separately for gate/up. It then
preserves the existing GLU boundaries: BF16 projections, BF16 after each
configured clamp, BF16 SiLU output, and BF16 final multiplication. Combining
all raw blocks before scaling, restarting or reassociating the K sum, or
removing a BF16 boundary changes the numerical contract.

The renderer proves chip visibility and compact owner addressing for **both**
weight buffers and both scale tables. It validates all three scratch ABIs and
all-CTA participation. The TIR candidate records chip read effects and one
internal grid barrier before selection, allocation and synchronization.

Regression coverage includes ordinary compile/dump/reload/publication,
canonical and compact physical addresses with CBB/BCB placements, two device
coordinates, both scale tables, illegal physical storage, FP64 cancellation
witnesses, partial N tiles, complete K4096/N4 stage execution, clamps,
repeated input/scale mutations, guards, a following cross-CTA consumer and
zero spill. Performance acceptance measures the entire ordinary stage,
including publication, against the unchanged MMA implementation.

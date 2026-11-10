# IQ4 MMVQ source lineage

Starting point: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp),
commit `5ce06fc71f5e938301f7dc51697a59ae308bd6b7` (2026-11-05), MIT.
The upstream license is preserved in `LICENSE.llama.cpp`.

| Upstream source/symbol | Local port | Adaptation |
| --- | --- | --- |
| `ggml/src/ggml-cuda/vecdotq.cuh::get_int_from_table_16`, CUDA branch | `lookup` in `templates/kernels/iq4_mmvq_helpers.cpp.jinja` | Same PRMT selection/reordering; fixed IQ4 table constant-folded into four words; HIP/MUSA branches omitted on this SM120-only recipe. |
| `vecdotq.cuh::vec_dot_iq4_xs_q8_1` | `dot` in the same helper | Same 4 packed words, 8 signed DP4A, 6-bit scale decode. `group/8` and `(group%8)*4` replace `kbx` and `iqs`; FTT byte-reader guards original packed-row alignment instead of assuming every byte-ABI pointer is aligned. |
| `ggml/src/ggml-cuda/quantize.cu::quantize_q8_1` | `quantize` in the same helper | A warp still reduces each 32-value amax and applies `roundf`. Source is BF16 from the nncase profile, not F32 llama activations. Temporary scale uses F32 (rather than the upstream half2 `d,sum` field) to reduce error; IQ4 does not use the `sum`. |
| MMVQ row/K cooperation | local block_fp8 and matmul_glu templates | A warp owns a row, its lanes own interleaved 32-value groups, lane 0 alone stores the complete sum; each CTA caches its Q8 activation once per call in a declared private shared arena. |

The optional second component quantizes the residual `x - q*d`, with explicit
FP32 multiply/subtract rounding. It is a local precision improvement, not an
upstream feature. `dual_glu` applies it to fused GLU; `dual_all` applies it to
all replaced matrices. Projection outputs retain the original storage dtype;
GLU keeps F32 gate/up projections through SiLU and rounds only its final result.
This approximation must pass the frozen independent greedy numerical gates;
source reuse is not numerical acceptance.

The original 136-byte IQ4_XS blocks, row strides and weight pools remain
unchanged. Q4_K and non-matrix families use their original core kernels.
Every quantizer publication and shared-arena reuse has a full-CTA barrier.
Uniform row masks apply to the entire warp, so complete warp shuffles execute
only on active rows, and only lane 0 is a physical output writer.

The package descriptor records exact configuration, source digest, all changed
call sites, shared bytes/alignment/lifetime/publication, and the upstream pin.
Generated C++ declares that scratch once; NVCC static shared usage and runtime
CUDA occupancy admission include it together with all core shared allocations.
No artifact source or binary is patched after normal code generation.

This first port has no producer/consumer prefetch roles. The whole model remains
one cooperative kernel; each matrix has quantize/read/reuse barriers. Weight
prefetch is synchronous and no cross-layer overlap is claimed. The next measured
step is to quantify whether warp-level IQ4 decoding or remaining GDN work bounds
the complete workload, then assess a pipelined upstream-based schedule.


## Interleaved attention projection extension (private a004 draft)

The same pinned IQ4 lookup, DP4A dot, and Q8 quantization helpers now serve
`interleaved_projection/simt` when `interleaved=single/dual` is explicit.
The template retains the core renderer's source row, native weight row, masks,
field selector, and separate BF16<8> destination offsets. A warp owns an output
row; only lane zero writes; both CTA barriers surround the reused Q8 arena.
No new weight format or external ABI is introduced. The initial trial uses
two Q8 components and retains the exact scalar-ordered GDN head cache.
This activation approximation is explicit local recipe behavior; it does not
change the generic op's mathematical contract. It requires full independent
first-eight-token validation. This draft has no compiled or device receipt.

## U4-L global Q8 and four-warp rows (K0 a0042)

The new `u4_common.cpp.jinja` lookup is a literal copy of the t9 local lookup
above, itself a port of `get_int_from_table_16` at the same pinned llama.cpp
commit. `u4_q8_encode.cpp.jinja` copies the t9 quantizer's amax, F32 scale,
`roundf`, zero handling and explicit residual RN body. Its group is now
`4*blockIdx.x+warp`; the full-vector loop and Shared publication are replaced
by complete global producer-kernel publication. F32 scales and residual Q8
are local adaptations, not the upstream Q8_1 half-scale format.

`u4_common.cpp.jinja::group_dot` preserves the upstream/t9 four-word low/high
nibble PRMT lookup and signed DP4A order. The existing unchanged FTT
`iq4_row_reader` supplies checked global aligned-u32 access or little-endian
byte fallback for all six header/scales/payload words. The public native
weight ABI remains alignment1 and row pitch K/256*136 bytes.

`u4_mmvq.cpp.jinja` changes row ownership to one CTA and group ownership to
`32*warp+lane+128*i`. Loaded Q8 records are reused across GLU gate/up. The
new local interwarp Shared partial publication, component-separate w0→w1→w2→w3
combine, and final shuffle-down tree implement the approved local numerical
ID `u4l-g128-w0123-comp-separate-sd16-v1`; this is not an upstream schedule or
a bitwise t9 result claim. Explicit RN scale multiply/FMA accumulation and RN
component/interwarp additions make the approved arithmetic inspectable.

`u4_gdn_z_gather.cpp.jinja` is new local coordinate-copy code. No GDN helper
or earlier helper file was modified. U4 has no cp.async/prefetch implementation.
All device correctness, native instruction, resource, NCU and measured
performance acceptance gates remain open until independently checked.

## Fixed GDN Q/K register hoist (A48, t18)

The `dual_q8_global_qkregs` recipe derives the existing MIT llama.cpp
`5ce06fc71f5e938301f7dc51697a59ae308bd6b7` warp/register GDN port from A117.
It retains the fixed96-CTA,128-thread,64-value mapping and global dual-Q8
workspace. Each lane loads and divides its four K and four Q values once after
the statistics CTA barrier; Q retains its separate F32 scale multiplication.
The values remain in registers across the existing sixteen value iterations.
This is local invariant motion; state arithmetic, reductions, stores, owners,
and synchronization remain the parent implementation. No reciprocal
approximation or upstream Q/K numerical contract is substituted.

## A49 input NormApply + C1 Q8

The fixed 40 CTA x 128 thread kernel reuses the literal C1 `u4_q8_encode`
body (llama.cpp MIT revision `5ce06fc71f5e938301f7dc51697a59ae308bd6b7`)
after authoritative FlagMega `norm_apply/_function.cpp.jinja` RMS arithmetic.
The normalized value goes through the original source and destination BF16
rounding, and only that BF16 value feeds amax/127/roundf. Scale and bias are
retained. The local change is register handoff between those bodies plus two
independent canonical outputs. Input hidden remains immutable through residual.
No new upstream/SOTA claim or native performance qualification is made.

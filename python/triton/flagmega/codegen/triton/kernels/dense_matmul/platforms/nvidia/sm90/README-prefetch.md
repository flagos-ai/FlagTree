# Ordered packed BF16 MMA prefetch

`packed_k_major_mma_prefetch` implements the existing packed matmul semantics.
`targets/nvidia/dense_matmul.py` provides an optional implementation factory;
the generic SM90 catalog does not select it automatically. The GLM profile
registers and selects it before LowerTIR using the ordinary selection plan.

The source, packed `k_major_n8_k16` weight and result are BF16. Each owner
consumes the complete logical K domain. The kernel keeps the existing M16/N16,
K32 MMA sequence and one continuous FP32 accumulator, then rounds the result
to BF16. Prefetching changes operand movement, not the reduction tree.

The kernel supports fixed local row loops, multiple N tiles and masked K/N
tails. The measured GLM policy chooses single rows with at most 16 physical
outputs and K at least 1024. This policy is a performance choice; it does not
restrict the implementation's arithmetic domain. Dtype and complete-K checks
remain the same as the original packed MMA. Invalid tile or Shared budgets
are rejected explicitly.

Each CTA owns private source and weight Shared buffers. For one N16 tile,
PN is the next power of two of the smaller of local N and 16. PK is bounded
by the implementation's power-of-two `prefetch_elements` and
`prefetch_max_k`, while retaining at least one K32 tile. The default budget
is 16384 BF16 weight elements with PK at most 2048; the two private buffers
use at most 36864 bytes. Actual complete-kernel resources remain subject to
the ordinary JIT resource and spill checks.

Weights use XOR-swizzled Shared rows (`vec=8`, `perPhase=1`, `maxPhase=8`).
Plain rows with a 2048-byte stride caused shared load bank conflicts in the
N16/K1024 prefetch tile. Bulk global offsets and masks have explicit blocked
copy layouts with eight consecutive logical BF16 K elements per active thread.
For contiguous packed K chunks this provides legal 16-byte transactions even
for short copy domains; a default
one-element-per-thread layout caused an invalid async-copy legalization
failure for M2/K80/N24. The implementation keeps normal async-copy and
pipeline passes enabled.

A CTA barrier publishes each prefetched chunk before consumption, and a
second protects reuse after all K32 MMA steps. The final chunk executes only
the original remaining K32 steps, including the original masked last step;
it does not append zero MMA steps. There are no new global workspaces,
aliases, remote reads or cross-CTA effects. External ABI, readonly packing
and graph-level evaluator contracts are unchanged.

Regression coverage is in `test_prefetch_mma_runtime.py`: selection and
emit/load, invalid parameters, 8/16 warps, K1536/K1552/K1152 tails, N8/N16,
M2/K80/N24, the seed712 input that distinguishes MMA from SIMT GEMV,
precision cancellation, complete outputs, guards and a sharded successor.
The local optimization log records failed reproductions, paired stage
measurements, hardware bank-conflict counters and full-model acceptance.

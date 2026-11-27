# Complete-K FP8 gate/up

`complete_k_wgmma` is an optional implementation of the existing
`nn.sparse_experts_gate_up` contract. It is not installed in the generic target
or the accepted full-model profile. Register
`sparse_experts_complete_k_implementations()` explicitly to experiment.

The first phase quantizes each chip-local `(token, route, K128)` once using the
existing UE8M0 quantizer. Its declared invocation workspaces are FP8
`[tokens, chip_routes, K]` and FP32 `[tokens, chip_routes, K/128]`. A grid barrier
publishes these values. The second phase owns complete output tasks, accumulates
all K128 blocks in ascending order, and implements the original projection
rounding, optional clamps, activation rounding and SwiGLU stores. It does not
materialize FP32 projection partials or perform a final K merge.

Each output task concatenates gate/up rows inside its matrix instruction:
N32 uses M64, N64 uses M128. Both retain the native FP8 WGMMA numerical method;
reducing matrix M below 64 can select a different instruction and is not an
interchangeable tiling choice. Each raw dot is multiplied by its B scale with
`mul_rn`, then merged with its A scale using the original FP32 FMA order.
`num_stages` explicitly controls one to four weight-prefetch stages.
Keep the full FP32 matrix accumulator in the WGMMA layout throughout K and
extract its real column only after reduction. Extracting that column on every
K128 block forces a shared-memory layout round trip. Load the quantized input
as a matrix with only column zero valid, rather than broadcasting a register
vector into shared memory: the matrix form participates in the same asynchronous
prefetch pipeline as the weights. The unused columns never escape the kernel;
their zeros do not change the real column's ordered arithmetic.

The implementation declares chip reads and chip writes. Tasks may span logical
block owners, so a block-private result is rejected. Peer addresses use the
actual canonical or compact-per-chip ABI, including route coordinates, all
packed output lanes and the per-device component map. It uses the ordinary
candidate, workspace, allocation, synchronization, serialization and codegen
paths; it needs SM90 FP8 WGMMA, cooperative-grid and grid-sync capabilities.

The initial complete-K design repeated quantization in each output task and was
slower than the existing cooperative implementation. Prequantization and packing
gate/up alone did not produce a robust gain. Retaining the matrix accumulator
and asynchronously loading the input removed the measured layout/prefetch
bottlenecks in the isolated cold-bank stage. Fewer scratch bytes are not
performance acceptance. Preserve the existing accepted profile until
isolated measurements, the actual whole-kernel resource context, TP4 execution
and the complete model pass their requested gates.

Regression coverage is in
`python/test/flagmega/codegen/triton/kernels/sparse_experts/test_complete_k_wgmma.py`:
bitwise device comparisons, mutations, packed lanes, N tails, one-CTA and
128-CTA execution, successor publication, guards, spill checks, physical peer
addresses, invalid contracts, and Python round trips.
The device-code regression also checks that the actual innermost K loop carries
an MMA matrix accumulator without a matrix-to-blocked conversion and pipelines
the FP8 activation matrix through asynchronous global-to-shared copies.

# KDA input projection

`nn.kda_input_projection` owns six independent BF16 projections of one decode
input: `(q, k, v, beta_logits, forget_a, output_a)`. Each dot product accumulates
in FP32 and rounds once to BF16. Convolution, gate activations, second low-rank
projections, normalization, and mutable state remain explicit consumers.

Q/K/V weights are `[heads, dim, K]` row-major views of three original matrices;
beta is `[heads,K]`, and the two low-rank matrices are `[rank,K]`. The importer
does not concatenate them. Weight views must have proven constant provenance
for offline materialization. Each output supports its own final-axis lane
shape, while the source can already pack its final K axis. There is no runtime
Pack or scalar reorder. Sharded beta currently requires scalar elements.

Distribution keeps K complete. Q/K/V and beta share contiguous head ownership
across chip/node axes; physical block axes remain broadcast. Both low-rank
inputs and results remain replicated across chips. A later explicit Boxing can
publish heads to block owners for convolution and recurrence.

Both implementations partition 64-row tasks across the physical block grid,
writing separate, dense chip output buffers. They declare chip read/write
effects and the `cooperative_grid` execution requirement independently of
broadcast result ownership. The ordinary planner owns allocation, publication,
and synchronization. A one-CTA launch would preserve values but serialize all
tasks; the launch regression checks this through normal compilation.

The SM90 WGMMA implementations require complete K512 tiles. Typed double-buffered
storage owns BF16 weight `[2,64,512]` and activation `[2,8,512]`
workspaces, aligned by the NVIDIA NVMMA storage policy. Six device tensor maps
retain the original weights. A seventh map reads `[1,K]` activation storage;
TMA supplies zero padding for seven unused rows. Both copies use the same TMA
transport, including when the planner aliases workspaces into one shared
arena. The computation waits for each tile, issues shared/shared WGMMA, completes
it before reusing the slot, then stores accumulator column zero directly
into the selected field ABI. Descriptor coordinates are checked to fit i32.

The default `wgmma_local` schedule keeps both transfers and computation within
the entry's existing eight compute warps. Two explicit TMA completion barriers
per slot permit prefetching the next tile while computing the current one;
WGMMA completes before that slot is refilled. Slot phases continue across
multiple output tasks on one CTA, including odd K-tile counts. The declared
Shared allocations map into an invocation-local frame with their proven
alignment and disjoint physical roots preserved. This schedule does not add a
global producer partition. The separate `wgmma` implementation places these
workspaces in a typed capacity-2 transfer channel and remains an explicit
warp-specialized choice for resource-compatible programs.

This distinction is a program resource contract: adding four producer warps
to an eight-warp megakernel expands every CTA from 256 to 384 threads. On the
complete GLM candidate that capped allocation at 168 registers per thread,
introducing spill in neighboring compute kernels that previously fit at 240
registers. The local schedule retains asynchronous double buffering and the
six-way projection while preserving the original thread budget. There is no
register-limit override or relaxed spill acceptance.

The MMA2 implementation covers other positive fixed K, including tails, with
masked K32 loads and the same six-result BF16 boundary. This is a separate
declared geometry, not a silent replacement for a selected WGMMA schedule.
Current device support is NVIDIA SM90; other targets are rejected explicitly.

Regression tests cover scalar/mixed packed outputs, editable Python round trips,
chip-head distribution and inverse demand, ordinary/large producer propagation,
shared scalar/tuple consumers, input views, typed workspaces/effects/descriptors,
the full block launch, actual device results and output guards. Complete model
token/state correctness and operator/full-model performance are separate gates;
an isolated kernel result does not establish either full-model claim.

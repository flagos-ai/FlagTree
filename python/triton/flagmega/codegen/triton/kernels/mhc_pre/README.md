# Cooperative ordered mHC projection

The optional NVIDIA implementations `cooperative_stream_fma` and
`cooperative_contiguous_tf32` implement the existing `nn.mhc_pre` contracts
`projection_reduction="stream_fma_128"` and `"contiguous_tf32"`, respectively.
Targets opt in through `mhc_cooperative_stream_fma_implementations()` or
`mhc_cooperative_contiguous_tf32_implementations()` after workload validation.
They introduce no semantic operation or default implementation change.

The schedule has three storage-separated phases:

1. Grid-strided tasks compute independent `(token, row pair, logical split)`
   projection partials and `(token, logical split)` square partials. Each pair
   task holds only `2 x 1 x 128` FP32 accumulators. For contiguous TF32, one
   `(token, logical split)` task computes all coefficient rows and its square
   partial with the existing 64-element TF32 dot blocks. This distributes the
   existing splits instead of recomputing their entire serial loop per CTA.
2. After a grid barrier, one CTA per token merges splits in increasing order,
   then computes the existing RMS, sigmoid, and Sinkhorn expressions. It
   writes post/comb outputs and publishes the pre coefficients.
3. After a second grid barrier, grid-strided `(token, hidden tile)` tasks
   collapse the streams with the original precision and reduction contract.

Each phase is a device function; the inlined wrapper owns the two grid
barriers. No computed register value crosses a phase boundary. All CTAs
participate, including CTAs without tasks. A single-CTA launch uses the same
phases with CTA barriers. Multiple tokens have separate workspace rows, so
they do not need a workspace-reuse barrier inside a token loop.

The invocation workspace `projection_partials` is a chip-shared, contiguous
FP32 tensor, aligned to 256 bytes. With `C = hc * (hc + 2)` and `S` logical
splits, each token has `(C + 1) * S + hc` elements:

| Element range | Contents |
| --- | --- |
| `[0, C * S)` | Projection partial at `row * S + split` |
| `[C * S, (C + 1) * S)` | Square partial at `C * S + split` |
| `[(C + 1) * S, (C + 1) * S + hc)` | Pre coefficients |

Inputs must be complete logical replicas without partial, exclusive, or
unproved backing aliases. Identical complete block-private input replicas
are valid: a task reads its own replica. All outputs and the workspace must
have canonical coordinates and chip-shared physical storage. The renderer
rejects other physical ABIs. Workspace allocation and the two internal grid
barrier facts must be present before bufferization and synchronization
planning; source-only artifact refresh cannot introduce this schedule.

Within each stream-FMA projection split, hidden iteration, stream FMA, lane
reduction and ordered warp addition follow the existing contract. Contiguous
TF32 shares the same single-split helper with its serial implementation:
64-element block order, TF32 dot precision, four square-accumulator lanes and
their reduction remain unchanged. It retains the coefficient-lane layout
during the split merge. Neither schedule changes the logical split count.
The next phase
merges split partials with explicit FP32 `add_rn`, in the original split
order. Storing FP32 partials introduces no additional precision boundary.
Post/comb formulas, collapse input conversion, sequential FMA or sum, output
conversion, and packed output addresses retain their semantic attributes.
The sum collapse pins its stream reduction to the original lane layout:
moving this axis into warps would change the FP32 addition tree and can cross
a BF16 rounding midpoint. Projection layouts also use the actual entry warp
count when another selected kernel requires more than the catalog's minimum.

Regression coverage lives in `test_cooperative_stream_fma.py` and
`test_cooperative_contiguous_tf32.py`: ordinary
selection/lowering and IR round trip, workspace/barrier and invalid ABI
checks, packed outputs, odd coefficient counts, repeated inputs, multiple
tokens, single-CTA and 128-CTA execution, and bitwise comparison of the whole
stage with the original implementation, and mixed selection of both variants
in one artifact. The existing contiguous-TF32 tests also compare the shared
split helper with an independent reference. Local microbenchmarks do not replace
full-model correctness, spill, or performance acceptance.

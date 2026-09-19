# Flash Attention Forward

This directory provides the Triton forward interface `triton.ops._flash_attention_forward`.

The interface is aligned with `torch.ops.aten._flash_attention_forward`.

## Layout

- Dense inputs use `BSHD`: `(batch_size, seqlen, nheads, headdim)`.
- Varlen inputs use packed `THD`: `(total_seqlen, nheads, headdim)`.
- The varlen path is selected when both `cum_seq_q` and `cum_seq_k` are provided.

## Supported Features

These features are currently supported in forward:

1. Causal masking.
2. Variable sequence lengths.
3. Arbitrary `seqlen_q` and `seqlen_k`.
4. Arbitrary head sizes.
5. Multi-head attention, multi-query attention, and grouped-query attention.
6. Dropout.
7. ALiBi.
8. Optional debug mask return.

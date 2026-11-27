# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Run in the pinned native environment to produce independent CP references."""

import sys
import torch
from flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_cp_sm90 import (
    cp_delta_rule_t_precompute_dsl_sm90, cp_delta_rule_mn_precompute_dsl_sm90,
    cp_delta_rule_fixup_dsl_sm90, cp_delta_rule_prefill_dsl_sm90,
)

tokens, output_path = int(sys.argv[1]), sys.argv[2]
torch.manual_seed(20261023)
heads, dimension = 2, 128
query = torch.randn(tokens, heads, dimension, device="cuda")
query = (query / torch.linalg.vector_norm(query, dim=-1, keepdim=True)).bfloat16().contiguous()
key = torch.randn_like(query, dtype=torch.float32)
key = (key / torch.linalg.vector_norm(key, dim=-1, keepdim=True)).bfloat16().contiguous()
value = torch.randn_like(query).contiguous()
alpha = torch.exp(-torch.rand(tokens, heads, device="cuda") * .05).contiguous()
beta = torch.sigmoid(torch.randn(tokens, heads, device="cuda")).contiguous()
cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int64)
t = cp_delta_rule_t_precompute_dsl_sm90(key, beta, cu, tokens, max_seqlen=tokens)
m, n = cp_delta_rule_mn_precompute_dsl_sm90(key, value, t, alpha, cu, tokens, cp_chunk_len=512, max_seqlen=tokens)
fixed = cp_delta_rule_fixup_dsl_sm90(m, n, cu, tokens, cp_chunk_len=512)
output, state = torch.empty_like(value), torch.empty(1, heads, dimension, dimension, device="cuda")
cp_delta_rule_prefill_dsl_sm90(output, state, query, key, value, t, fixed, alpha, dimension**-.5,
                              cu, tokens, cp_chunk_len=512, max_seqlen=tokens)
torch.cuda.synchronize()
torch.save({name: tensor.contiguous().cpu() for name, tensor in dict(
    query=query, key=key, value=value, alpha=alpha, coefficients=-t.transpose(-1, -2),
    transfer=m, local_state=n, fixed_state=fixed, output=output, state=state[0],
).items()}, output_path)

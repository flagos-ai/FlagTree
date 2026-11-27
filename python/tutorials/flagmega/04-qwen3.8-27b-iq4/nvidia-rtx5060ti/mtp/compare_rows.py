"""Row-by-row comparison of a multirow verify artifact against sequential decode.

Reference: run the T=1 artifact once per token, carrying GDN/paged state, and
record the logits after each step.  Candidate: run the T=1..N artifact once with
all tokens.  Row r of the candidate must reproduce reference step r.

Usage: compare_rows.py CANDIDATE_ARTIFACT T1_ARTIFACT [TOKENS...]
"""
import json
import sys
from pathlib import Path

import torch

from triton.flagmega.ir import VectorType
from triton.flagmega.runtime import load

ATOL = 0.125


def build(artifact, device='cuda:0'):
    runtime = load(Path(artifact).resolve(), device=device)
    from r054_runtime import wrap
    runtime = wrap(runtime)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    values = {}
    for argument in runtime.external_arguments:
        b = runtime.buffer_plan.buffer_map[argument['buffer']]
        lanes = b.dtype.lanes if isinstance(b.dtype, VectorType) else ()
        scalar = b.dtype.elem_type if lanes else b.dtype
        values[b.id] = torch.zeros((*b.shape, *lanes), dtype=getattr(torch, scalar.value), device=device)
    values['paged_state.block_table'].copy_(torch.arange(
        values['paged_state.block_table'].numel(),
        dtype=values['paged_state.block_table'].dtype,
        device=device).reshape_as(values['paged_state.block_table']))
    logits_key = next(name for name in values if name.startswith('logits'))
    return runtime, stream, values, logits_key


def zero_pools(runtime):
    """Clear the temporary pools only.

    ``rdata`` is the 14 GB constant weight pool; clearing it (as an earlier
    version of this harness did) removes every weight and makes all logits
    zero, which silently turns the comparison into a no-op.
    """
    for name in ('workspace', 'block_local_data'):
        tensor = runtime._pool_values.get(name)
        if isinstance(tensor, torch.Tensor) and tensor.numel():
            tensor.zero_()


candidate_path, t1_path = sys.argv[1], sys.argv[2]
tokens = [int(t) for t in sys.argv[3:]] or [1000, 2000, 3000, 4000, 5000]
steps = len(tokens)

report = {'tokens': tokens, 'atol': ATOL}

# ---- reference: sequential decode-1 -------------------------------------
r1, s1, v1, logits1 = build(t1_path)
v1['input_ids'].copy_(torch.tensor([tokens[0]], dtype=v1['input_ids'].dtype, device='cuda'))
args1 = tuple(v1[x['buffer']] for x in r1.external_arguments)
r1.prepare(*args1)
s1.synchronize()
zero_pools(r1)

reference = []
for step in range(steps):
    v1['input_ids'].copy_(torch.tensor([tokens[step]], dtype=v1['input_ids'].dtype, device='cuda'))
    # Shape-bucket convention: the call extent comes from the tensor shape, so
    # query_start_loc stays the two-entry [0, 1] record.  A fresh sequence
    # starts empty: the KV writer positions token k at slot k, so the history
    # must be exactly k long before the call, never k+1.
    v1['paged_state.query_start_loc'].copy_(torch.tensor([0, 1], dtype=torch.int32, device='cuda'))
    v1['paged_state.seq_lens'].fill_(step)
    v1['paged_state.slot_mapping'].fill_(step)
    r1.run_into(*args1, stream=s1.cuda_stream)
    s1.synchronize()
    reference.append(v1[logits1][0].detach().clone())
report['t1_top1'] = [int(x.argmax()) for x in reference]
report['t1_finite'] = [bool(torch.isfinite(x).all()) for x in reference]

# One rdata pool is ~14 GB on a 15.5 GB card, so the candidate artifact cannot
# be resident at the same time as the reference.  Release the reference first.
del args1, v1, r1, s1
torch.cuda.empty_cache()
torch.cuda.synchronize()

# ---- candidate: one multirow call ---------------------------------------
rc, sc, vc, logitsc = build(candidate_path)
vc['input_ids'].copy_(torch.tensor(tokens, dtype=vc['input_ids'].dtype, device='cuda'))
vc['paged_state.query_start_loc'].copy_(torch.tensor([0, steps], dtype=torch.int32, device='cuda'))
# Match reference step 0: seq_lens is the history length before this call.
# Starting at `steps` shifts RoPE positions and leaves empty cache slots in
# front of the candidate chunk, so it compares different attention contexts.
vc['paged_state.seq_lens'].fill_(0)
# slot_mapping is per-sequence (shape (1,)): the multirow call derives the
# per-token slots from the query extent, so the base slot stays 0.
vc['paged_state.slot_mapping'].fill_(0)
argsc = tuple(vc[x['buffer']] for x in rc.external_arguments)
rc.prepare(*argsc)
sc.synchronize()
zero_pools(rc)
rc.run_into(*argsc, stream=sc.cuda_stream)
sc.synchronize()
got = vc[logitsc]

rows = []
for row in range(steps):
    actual = got[row].float()
    expected = reference[row].float()
    diff = (actual - expected).abs()
    nan = int(torch.isnan(actual).sum())
    rows.append(dict(
        row=row,
        nan=nan,
        finite=bool(torch.isfinite(actual).all()),
        maxabsdiff=float(diff.nan_to_num(float('inf')).max()),
        top1=int(actual.argmax()),
        ref_top1=int(expected.argmax()),
        top1_match=bool(int(actual.argmax()) == int(expected.argmax())),
        pass_atol=bool(nan == 0 and bool((diff <= ATOL).all())),
    ))
report['rows'] = rows
report['all_pass'] = all(r['pass_atol'] for r in rows)

print(json.dumps(report, indent=1))

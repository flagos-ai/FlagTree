"""Compare every vocabulary logit as KV state crosses tile/page boundaries."""
import json
import sys
import torch
from triton.flagmega.serving.backend import ArtifactBackend

base = ArtifactBackend.load(sys.argv[1], device="cuda:0", cuda_graph=True, warmups=8)
candidate = ArtifactBackend.load(sys.argv[2], device="cuda:0", cuda_graph=True, warmups=8)
print("loaded", sys.argv[1:3], flush=True)
maximum = 0.0
exact = True
tokens = []
for step in range(2112):
    token = 9707 if step == 0 else base.sample()
    base.append(token)
    candidate.append(token)
    a, b = base.logits, candidate.logits
    difference = float((a-b).abs().max())
    maximum = max(maximum, difference)
    exact &= bool(torch.equal(a, b))
    assert torch.equal(a,b), (step,difference)
    assert base.sample() == candidate.sample(), (step, base.sample(), candidate.sample())
    tokens.append(token)
    if step in (0, 15, 63, 127, 255, 511, 599, 1023, 1535, 2047, 2111):
        print("CHECK", json.dumps({"context":step+1,"max_abs":difference,"exact":bool(torch.equal(a,b))}), flush=True)
print("LONG_CORRECT", json.dumps({"steps":2112,"exact":exact,"max_abs":maximum,"tokens":tokens}),flush=True)
base.close()
candidate.close()

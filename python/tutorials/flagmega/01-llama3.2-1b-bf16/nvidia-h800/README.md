# Llama 3.2 1B BF16 on NVIDIA H800

This example compiles batch-one, single-token decode for an H800. Use a local
Llama 3.2 1B Instruct BF16 safetensors checkpoint containing `config.json` and
the model weights. Obtain the weights from the model publisher under its terms.

Run these commands from the repository root in the environment described in
the [tutorial index](../../README.md).

```sh
export CUDA_VISIBLE_DEVICES=0
CHECKPOINT=/path/to/Llama-3.2-1B-Instruct
WORK_DIR="$(mktemp -d)"

flagmega compile --model "$CHECKPOINT" --full-model --mode decode-1 \
    --target nvidia-sm90 --emit-executable --output "$WORK_DIR/artifact"
flagmega artifact verify "$WORK_DIR/artifact"

python - "$WORK_DIR/input.pt" <<'PYTHON'
import sys

import torch

torch.save(torch.tensor([128000], dtype=torch.int32), sys.argv[1])
PYTHON

flagmega artifact run "$WORK_DIR/artifact" --input "$WORK_DIR/input.pt" \
    --device cuda:0 --output "$WORK_DIR/output.pt"

python - "$WORK_DIR/output.pt" <<'PYTHON'
import sys

import torch

output = torch.load(sys.argv[1], weights_only=True)
assert torch.isfinite(output["logits"]).all()
print("Next token ID:", output["next_token"].item())
PYTHON
```

The saved output contains logits, the greedy next token, and the paged KV cache
after that step.

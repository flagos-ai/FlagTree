# Qwen3 1.7B BF16 on AMD RDNA

This example compiles batch-one decode for RDNA3 (`gfx1100`) or RDNA4
(`gfx1201`). Install the AMD build of FlagTree from the
[tutorial index](../../README.md) and use a local Qwen3 1.7B BF16 safetensors
checkpoint.

```sh
python python/tutorials/flagmega/02-qwen3-1.7b-bf16/amd-rdna/run.py \
    --checkpoint /path/to/Qwen3-1.7B --arch gfx1100 \
    --work-dir "$(mktemp -d)"
```

The script emits a runtime artifact and prints the greedy next token and
whether the logits are finite. Pass `--benchmark --context-lengths 1024,4096`
to time decode after populating the KV cache with generated tokens.

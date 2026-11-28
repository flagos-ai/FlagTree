# FlagMega tutorials

Build FlagTree from the repository root for the device used by the tutorial.
Use a Python environment with the matching PyTorch distribution.

```sh
FLAGTREE_BACKEND=nvidia FLAGTREE_TLE=1 python -m pip install '.[flagmega]'
# For the AMD example, use FLAGTREE_BACKEND=amd in a ROCm environment.
```

| Number | Model | Example |
| --- | --- | --- |
| 01 | Llama 3.2 1B BF16 | [NVIDIA H800](01-llama3.2-1b-bf16/nvidia-h800/README.md) |
| 02 | Qwen3 1.7B BF16 | [AMD RDNA](02-qwen3-1.7b-bf16/amd-rdna/README.md) |
| 03 | Qwen3.5 35B A3B BF16 | |
| 04 | Qwen3.8 27B IQ4 | |
| 05 | GLM 5.3 Flash FP8 | |

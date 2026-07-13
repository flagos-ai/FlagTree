# Proton CANN Liger-Kernel Full Suite

This suite profiles a real open-source Triton operator library:

https://github.com/linkedin/Liger-Kernel

It validates that Proton CANN can profile Triton kernels coming from a public
operator library, not only the small in-repository smoke kernels.

## Scope

The suite uses Liger's low-level operator APIs and runs these cases on Ascend
NPU tensors:

- `liger_rms_norm`
- `liger_layer_norm`
- `liger_fused_add_rms_norm`
- `liger_modulated_rms_norm`
- `liger_poly_norm`
- `liger_dyt`
- `liger_relu_squared`
- `liger_swiglu_mlp`
- `liger_geglu_mlp`
- `liger_softmax`
- `liger_sparsemax`
- `liger_rope`
- `liger_cross_entropy`
- `liger_fused_linear_cross_entropy`
- `liger_kl_div`
- `liger_jsd`
- `liger_tvd`
- `liger_group_norm`
- `liger_fused_linear_jsd`

## Run

Run with the script-managed Liger checkout:

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
python third_party/proton/flagtree_profiler/scripts/cann_profile_test_suite.py \
  --out /tmp/proton_cann_liger_full \
  --with-liger \
  --clean
```

When `--liger-source` is omitted, the script clones Liger-Kernel into
`<out>/liger/Liger-Kernel`. To reuse an existing checkout, pass
`--liger-source /path/to/Liger-Kernel`.

The explicit clone option is still accepted:

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
python third_party/proton/flagtree_profiler/scripts/cann_profile_test_suite.py \
  --out /tmp/proton_cann_liger_full \
  --with-liger \
  --clean
```

The script uses Proton's direct workflow:

```python
sid = proton.start(..., backend="cann", hook="triton", ...)
# run Liger operators
proton.finalize(sid)
```

No user-visible external `msprof python ...` wrapper is required. The CANN
backend still invokes `msprof --export=on` internally during `finalize()` to
convert CANN profiling data into CSV files for import.

## Outputs

The output directory contains:

- `liger_full_profile.meta.json`
- `liger_full_profile.vendor.json`
- `liger_full_profile.timeline.json`
- `liger_full_profile.hatchet`
- `summary.json`

`summary.json` reports case success/failure, CANN association sources, MSTX
ranges, top CANN op types, bandwidth association count, and degradation reasons.

## Notes

Liger-Kernel's low-level operators only require Torch and Triton. On Ascend,
the suite sets `LIGER_KERNEL_IMPL=ascend` and patches Liger's NPU detection to
use `torch.npu.is_available()`, so the full suite does not require installing
the optional `transformers` package.

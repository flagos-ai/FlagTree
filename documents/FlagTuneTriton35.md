# FlagTune on Triton 3.5 / Ascend

This port takes `python/triton/flagtune` and its tests from FlagTree main
`d65b6d70511e4d9aaeb3bf52ae762a96212def0a` onto `triton_v3.5.x`
`394d50493a21dc99bdbfc527b9625dbbb5f327da`.
It includes optional dependency declarations and the shared graph benchmark's
configurable retry count and caller-stream ordering fix. Triton 3.5's compiler,
Autotuner and Ascend driver APIs are preserved.

## Ascend runtime

The Ascend driver reports target backend `npu`, rather than `ascend`.
FlagTune registers this target with vendor `huawei` and PyTorch device type
`npu`. SoC names such as `Ascend910B1` and `Ascend910_9392` are retained in
lowercase as architecture metadata. Device ordinal and product name come from
the active driver's device interface. A product name of `Ascend910B` or
`Huawei Ascend910B` produces platform key `huawei-ascend910b`.

Event timing uses the active Ascend driver's benchmarker and NPU events.
Set `FLAGTUNE_BENCHMARK_MODE=event` to select it explicitly. The inherited
FlagTune default requests replay; Ascend currently warns and falls back to
event timing. It never calls the CUDA graph helper. A strict replay request
(`allow_fallback=False`) fails because no Ascend replay protocol is registered.

Install the Ascend build of this branch in the existing CANN / torch_npu
environment using the repository's Ascend build instructions, adding the
`flagtune` extra for model loading and training:

```sh
# Use the same backend/build environment as your normal Ascend installation.
FLAGTREE_BACKEND=ascend python -m pip install -e '.[flagtune]'
FLAGTUNE_BENCHMARK_MODE=event python -m pytest python/test/flagtune -q
python -m pytest python/test/flagtune/test_ascend_integration.py -q
```

Use `from triton.flagtune import flagtune` as the autotuning decorator.
Without an `op_id` / `variant`, it runs ordinary Triton configuration tuning.
Model-driven prediction additionally requires `FLAGTUNE_ENABLE=1`, a matching
operator/variant/dtype contract, and a model package trained for the exact
Ascend platform key. Porting the runtime does not provide Ascend model weights;
models for other vendors cannot be reused as Ascend models.

The integration test tunes a vector addition kernel with two configurations,
checks fp16/bf16/fp32 results, and exercises the tuning cache. It requires a
working Ascend installation and skips when torch_npu or NPU hardware is absent.
It covers baseline tuning; enabled model prediction additionally needs an
Ascend model package.

## Local validation limits

Development was performed on macOS without torch, torch_npu or Triton's native
extension. CPU tests used the real ported FlagTune code, Triton 3.5 Autotuner,
and benchmark helper, with temporary stand-ins for native/compiler/JIT import
boundaries. Training and model package tests used installed NumPy, PyYAML,
scikit-learn and XGBoost. These checks do not validate NPU kernel compilation,
event timing accuracy, or device execution. Run the integration test above on
Ascend before merging.

# FlagTree Function-Level Autotuner

> **Language / 语言**: [English](autotuner.md) | [中文](autotuner-cn.md)

`flagtree.tune.autotune` tunes the parameters of Python functions. The decorated function can launch kernels that perform distributed communication.

Key features:

- Define candidate parameters as a list of dictionaries, with support for `key_fn` and `prune_fn`.
- Tune locally or jointly within a specified process group.
- Intersect distributed candidate lists, check that effective tuning parameters agree across ranks, and coordinate cache-hit decisions.
- Automatically cache tuning results under the directory specified by the `FLAGTREE_AUTOTUNE_CACHE_DIR` environment variable or the tuner's `cache_dir` argument. The default locations are `~/.flagtree/autotune/local` and `~/.flagtree/autotune/distributed`.

The autotuner does not automatically detect whether a function performs distributed communication. If the function requires multiple ranks to cooperate, explicitly pass the process group through `autotune_pg` to enable distributed tuning.

## 1. Decorator Parameters

```python
def autotune(
    config_space,
    key_fn,
    prune_fn=None,
    *,
    cache_dir=None,
    cache_version: str | None = None,
):
    ...
```

| Parameter | Description |
| --- | --- |
| `config_space` | Required. A list of dictionaries containing tunable parameters. Dictionary keys correspond to keyword arguments of the decorated function. |
| `key_fn` | Required. A function that receives the arguments passed by the caller and returns a hashable cache key. |
| `prune_fn` | Optional. A function that filters candidates: a truthy return value keeps a candidate, and a falsy value discards it. |
| `cache_dir` | Optional. The disk-cache directory. Defaults to the environment variable or the default path; pass `False` to disable disk caching. |
| `cache_version` | Optional. A string or `None`. Distinguishes implementation versions that require retuning. Changing this string forces a new disk-cache entry. |

## 2. Basic Usage

Suppose a host-side Python function, `process_data`, needs to choose two parameters:

- `kernel_config`: a `triton.Config` object specifying the kernel configuration.
- `batch_size`: the number of rows to process in each batch.

The following example only demonstrates argument passing. Fill in the function body with the actual GPU computation.

```python
import triton
from flagtree.tune import autotune, to_hashable


@autotune(
    config_space=[
        {"kernel_config": triton.Config({"BLOCK_SIZE": 128}), "batch_size": 16},
        {"kernel_config": triton.Config({"BLOCK_SIZE": 256}), "batch_size": 32},
    ],
    key_fn=lambda x, **kwargs: to_hashable(x),
)
def process_data(x, *, kernel_config: triton.Config, batch_size: int):
    # Process data in batches of batch_size; configure and launch the kernel with kernel_config.
    ...
```

The autotuner passes the two fields of each candidate dictionary as the function's `kernel_config` and `batch_size` arguments. The `triton.Config` object is passed directly; the function decides how to use it.

## 3. Calling the Decorated Function

After implementing the function body, a normal call only needs `x`, the GPU tensor to process:

```python
result = process_data(x)
```

When no cache entry is available, the autotuner benchmarks the two candidate combinations above and selects the faster one. A valid cache entry is reused if its selected configuration remains in `config_space`; otherwise, the autotuner runs tuning again.

| Call Argument | Default | Behavior |
| --- | --- | --- |
| `autotune` | `True` | Whether to benchmark candidates and use the tuning cache. Configuration selection when set to `False` is described below. |
| `autotune_allow_arg_overwrite` | `False` | Whether the caller may explicitly override candidate parameters. |
| `autotune_pg` | `None` | A non-`None` value enables joint tuning within the specified process group; otherwise, tuning is local. |

These call arguments are reserved by the autotuner and are not forwarded to the original function. All other arguments are passed according to the original function's signature.

### 3.1 Disabling Tuning

```python
result = process_data(x, autotune=False)
```

| Mode | Behavior with `autotune=False` |
| --- | --- |
| Local | Skips `prune_fn` and tuning-cache lookups, and executes the decorated function with the original `config_space[0]`. Argument-override checks still apply. |
| Distributed | Still checks whether argument overrides are allowed and prunes candidates on each rank. It then intersects the pruned candidate sets and selects the first candidate in canonical sorted order. No benchmarking or tuning-cache lookup is performed. |

### 3.2 Explicitly Overriding Candidate Parameters

To fix the batch size at 64 rows, explicitly pass `batch_size=64` and allow overrides:

```python
result = process_data(
    x,
    batch_size=64,
    autotune_allow_arg_overwrite=True,
)
```

The effective parameters of the two candidates are then:

| Configuration in `kernel_config` | Original `batch_size` | Overridden `batch_size` |
| --- | --- | --- |
| `BLOCK_SIZE=128` | `16` | `64` |
| `BLOCK_SIZE=256` | `32` | `64` |

**`batch_size` stays fixed at 64 while `kernel_config` is still tuned.** The final execution with the best configuration also uses `batch_size=64`.

Omitting `autotune_allow_arg_overwrite=True` causes an argument-conflict error.

Distributed calls also require `autotune_allow_arg_overwrite=True` for explicit overrides. If any rank supplies conflicting arguments without allowing overrides, the error is collected during preparation and reported by every rank in the group.

Even when overrides are allowed, all ranks must use the same final tuning-parameter values for each common candidate.

## 4. Distributed Joint Tuning

### Prerequisites

- Initialize the `torch.distributed` process group and select the device for each process before calling the function. The current `tle.get_mem_pool()` implementation handles this setup.
- All ranks in the group must use the same `autotune` setting.
- `autotune_pg` must include the processes that participate in the operator's communication. Typically, you can pass `dist.group.WORLD`.
- Distributed tuning requires a valid `LOCAL_WORLD_SIZE` environment variable. Launching with `torchrun` provides this variable.

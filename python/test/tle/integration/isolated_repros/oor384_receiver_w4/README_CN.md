# OOR384 receiver-w4 隔离复现

这个目录是 `Required: 384, Hardware limit: 256` 的隔离复现副本。它不 import
上层 `triton_tle_megamoe` 目录里的 Python/CUDA 文件，所需本地代码已经复制到
当前目录：

- `repro_receiver_w4.py`
- `triton_userhopper_single_kernel_l1_tldot_smoke.py`
- `triton_tle_ws_userhopper_dispatch_receiver_smoke.py`
- `ws_userhopper_dispatch_receiver_device.cu`
- `ws_userhopper_dispatch_receiver_extern_call.py`
- `ws_userhopper_dispatch_receiver_host.cu`
- `nvcc_flock_wrapper.sh`

## 修改点

目标 kernel：

```text
_single_kernel_dispatch_receiver_l1_l2_expert_wave_tldot_kernel
```

在隔离副本中，它的 TLE WS worker 配置从原始：

```python
tle.gpu.warp_specialize(..., [1, 4], [80, 180])
```

改成：

```python
tle.gpu.warp_specialize(..., [4, 4], [80, 180])
```

含义是：

- receiver worker partition: 4 warps
- compute worker partition: 4 warps

同时，`_receiver_pipe_to_compute_partition` 里加了 `warp0` guard：

```python
tid = tl.inline_asm_elementwise(
    "mov.u32 $0, %tid.x;",
    "=r",
    [],
    dtype=tl.uint32,
    is_pure=True,
    pack=1,
)
warp_id = tid // 32
if warp_id == 0:
    ...
```

也就是说，receiver partition 在资源配置上是 4 warp，但实际只有第一个 warp
执行：

```text
dispatch_reader.wait -> receiver extern -> compute_writer.commit -> dispatch_reader.release
```

其他 3 个 warp 不执行 receiver 逻辑。

## 运行命令

```bash
cd /workspace/megakernel/triton_flagos_support_nvshmem/python/test/tle/integration/isolated_repros/oor384_receiver_w4

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export NVSHMEM_HOME=/path/to/nvshmem
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="${CUDA_HOME}/targets/x86_64-linux/include:$NVSHMEM_HOME/include:${CPATH:-}"

PYTHONNOUSERSITE=1 \
PYTHONPATH=/workspace/megakernel/triton_flagos_support_nvshmem/python:$PWD \
NVCC=$PWD/nvcc_flock_wrapper.sh \
TRITON_CACHE_DIR=/tmp/tle_ws_oor384_receiver_w4_isolated \
${MEGAMOE_PYTHON:-python} \
  repro_receiver_w4.py
```

## 当前验证结果

已验证仍然触发：

```text
triton.runtime.errors.OutOfResources:
out of resource: threads, Required: 384, Hardware limit: 256.
```

这次 `Required:384` 不再能归因于原始 `[1, 4]` 的 receiver 1-warp 不规范配置。
隔离副本中的配置已经是 `[4, 4]`，因此更直接的解释是：

```text
default/dispatch partition: 4 warps
receiver worker partition: 4 warps
compute worker partition:   4 warps
total:                     12 warps = 384 threads
```

而当前 kernel launch / TLE WS backend 的 CTA thread limit 是：

```text
8 warps = 256 threads
```

所以这个 case 说明当前单 CTA 内同时放置 dispatch + receiver + compute 三个
4-warp role 会超过当前线程预算。它不是“receiver 只该用 1 warp 却被错误算成
4 warp”的复现。

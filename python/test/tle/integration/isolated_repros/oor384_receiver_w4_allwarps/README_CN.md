# OOR384 receiver-w4 all-warps 隔离复现

这个目录用于回答一个具体问题：

```text
如果 receiver partition 配置为 4 warps，并且不是只让 warp0 执行，
而是让 4 个 warp 合理分工做 receiver，还会不会触发 Required:384？
```

## 与 receiver_w4 版本的差异

`receiver_w4` 版本中，Triton wrapper 使用 `%tid.x` 计算 `warp_id`，只让
`warp_id == 0` 执行：

```text
dispatch_reader.wait -> receiver extern -> compute_writer.commit -> dispatch_reader.release
```

本目录的 all-warps 版本去掉了 Triton wrapper 里的 `warp_id == 0` guard。
也就是说，receiver partition 的 4 个 warp 都执行 pipe wait / receiver extern /
pipe commit 路径。

同时，`ws_userhopper_dispatch_receiver_device.cu` 中的
`userhopper_ws_receiver_partition` 被改成按 warp 分工：

```cpp
const uint32_t warp_idx = static_cast<uint32_t>(threadIdx.x >> 5);
const uint32_t lane_idx = static_cast<uint32_t>(threadIdx.x & 31);
constexpr uint32_t kReceiverWarps = 4;
if (lane_idx != 0 || warp_idx >= kReceiverWarps || blockIdx.x != 0) {
  return;
}
...
for (uint32_t token_idx_in_expert = warp_idx; token_idx_in_expert < total;
     token_idx_in_expert += kReceiverWarps) {
  ...
}
```

因此它不是 4 个 warp 重复接收同一批 token，而是 4 个 warp 的 lane0 分摊
`token_idx_in_expert`。

## 运行命令

```bash
cd /workspace/megakernel/triton_flagos_support_nvshmem/python/test/tle/integration/isolated_repros/oor384_receiver_w4_allwarps

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export NVSHMEM_HOME=/path/to/nvshmem
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="${CUDA_HOME}/targets/x86_64-linux/include:$NVSHMEM_HOME/include:${CPATH:-}"

PYTHONNOUSERSITE=1 \
PYTHONPATH=/workspace/megakernel/triton_flagos_support_nvshmem/python:$PWD \
NVCC=$PWD/nvcc_flock_wrapper.sh \
TRITON_CACHE_DIR=/tmp/tle_ws_oor384_receiver_w4_allwarps_isolated \
${MEGAMOE_PYTHON:-python} \
  repro_receiver_w4_allwarps.py
```

## 当前验证结果

仍然触发：

```text
triton.runtime.errors.OutOfResources:
out of resource: threads, Required: 384, Hardware limit: 256.
```

结论：这个 OOR 不取决于 receiver 内部是只用 warp0，还是让 4 个 warp 分摊
receiver 工作。错误发生在 kernel handle 初始化阶段，此时真正的 receiver 逻辑
还没有执行。关键仍是 WS role 配置导致最终 metadata 需要：

```text
dispatch/default partition: 4 warps
receiver worker partition: 4 warps
compute worker partition:   4 warps
total:                     12 warps = 384 threads
```

而 CUDA driver 对该 compiled function 返回的
`CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK` 是：

```text
256 threads = 8 warps
```

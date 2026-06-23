# no-NVSHMEM TLE WS 4+4+4 复现

这个目录用于检查 `Required: 384, Hardware limit: 256` 是否可以在完全不使用
NVSHMEM/raw extern 的情况下复现。

本目录只使用：

- 单进程
- 单 GPU
- 普通 CUDA tensor
- `tle.gpu.warp_specialize`
- 普通 `tl.store` / `tl.load`

不使用：

- `mpirun`
- `NVSHMEM_HOME`
- raw dialect extern
- `.cu` device/host helper

## 运行命令

```bash
cd /workspace/megakernel/triton_flagos_support_nvshmem/python/test/tle/integration/isolated_repros/no_nvshmem_ws_444

PYTHONNOUSERSITE=1 \
PYTHONDONTWRITEBYTECODE=1 \
MEGAMOE_TORCH_SITE_PACKAGES=/path/to/site-packages \
TRITON_CACHE_DIR=/tmp/tle_ws_444_no_nvshmem \
${PYTHON_BIN:-python} repro_ws_444_no_nvshmem.py
```

注意：如果使用 editable install 的 FlagTree/Triton 环境，不要额外把
`/path/to/FlagTree/python` 放进 `PYTHONPATH`，否则可能绕过 editable backend
finder，导致 `triton.backends.*` 或 `triton.experimental.tle` 解析异常。

## 验证结果

已验证这个 case 会触发：

```text
triton.runtime.errors.OutOfResources:
out of resource: threads, Required: 384, Hardware limit: 256.
```

结论：这个 thread-budget OOR 不依赖 NVSHMEM/raw extern。单进程、单 GPU、
普通 tensor、无 raw dialect extern 的 TLE WS 4+4+4 role 结构已经足够复现。

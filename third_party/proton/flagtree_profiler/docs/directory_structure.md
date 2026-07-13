# FlagTree Profiler 目录结构

FlagTree Profiler 的主体代码集中在 `third_party/proton/flagtree_profiler/`。因为它复用 Proton 的用户 API 和生命周期，仍有少量接入点保留在 `third_party/proton/` 原有位置。

## 主体目录

```text
third_party/proton/flagtree_profiler/
  README.md
  __init__.py

  csrc/
    include/
      Driver/Ascend/
        AscendApi.h
      Profiler/Vendor/
        Adapter.h
        CannAdapter.h
        CannProfiler.h
        Mode.h
      TraceDataIO/
        TraceWriter.h
    lib/
      Driver/Ascend/
        AscendApi.cpp
      Profiler/Vendor/
        Adapter.cpp
        CannAdapter.cpp
        CannProfiler.cpp
        Mode.cpp

  scripts/
    cann_profile_test_suite.py
    cann_flaggems_profile_suite.py
    cann_liger_profile_suite.py
    cann_native_acl_mstx_validate.py
    cann_operator_profile_suite.py
    cann_post_import_msprof.py
    cann_real_msprof_validate.py
    cann_real_npu_workload.py

  test/
    __init__.py
    test_cann_smoke.py

  docs/
    directory_structure.md
    proton_cann_acceptance_status.md
    proton_cann_flaggems_suite.md
    proton_cann_liger_full_suite.md
    proton_vendor_adapter_minimal_patch.md
```

## 主体目录职责

- `csrc/include/Profiler/Vendor/`：vendor backend 的 C++ 接口和 CANN 后端声明。
- `csrc/lib/Profiler/Vendor/`：CANN profiling、mode 解析、CSV 导入、metric 关联等核心实现。
- `csrc/include/Driver/Ascend/` 和 `csrc/lib/Driver/Ascend/`：昇腾 runtime/device discovery shim。
- `csrc/include/TraceDataIO/`：trace 输出辅助结构。
- `scripts/`：真实 NPU workload、CANN post-import、12 个自定义 Triton 算子 suite、Liger-Kernel 真实开源库验证 suite、FlagGems 公开 Triton 算子库全量评估 suite。
- `test/`：FlagTree Profiler 的 CANN 自动化测试。
- `docs/`：FlagTree Profiler 相关文档集中位置。

## Proton 主目录中的接入点

这些文件仍在 Proton 原目录中，因为它们是 Proton 的公共 API、生命周期或数据模型，不适合完全移入 `flagtree_profiler`。

```text
third_party/proton/CMakeLists.txt
```

把 `flagtree_profiler/csrc/lib/*.cpp` 编进 `libproton.so`，并加入 `flagtree_profiler/csrc/include` include path。

```text
third_party/proton/proton/hook.py
```

`hook="triton"` 的 Python 接入点。它兼容 Triton/FlagTree Ascend launcher 的 hook 参数，并在 kernel launch 前后进入/退出 Proton scope。

```text
third_party/proton/proton/proton.py
third_party/proton/proton/profile.py
```

Proton Python API 层，负责把 `backend`、`mode`、`hook` 等参数传到 C++ session。

```text
third_party/proton/csrc/include/Session/Session.h
third_party/proton/csrc/lib/Session/Session.cpp
```

`proton.start()` / `proton.finalize()` 生命周期。这里创建 vendor profiler，停止 profiling，并触发 vendor artifact 导入。

```text
third_party/proton/csrc/include/Data/
third_party/proton/csrc/lib/Data/
```

Proton 数据模型和序列化。FlagTree Profiler 复用这里输出 `meta.json`、`timeline.json`、`vendor.json` 和 `hatchet`。

```text
third_party/proton/csrc/include/Profiler/Profiler.h
```

Profiler 基类接口。CANN backend 通过这个接口接入 Proton profiler 生命周期。

```text
third_party/proton/csrc/include/Driver/Device.h
third_party/proton/csrc/lib/Driver/Device.cpp
```

设备枚举/发现接入点。昇腾 device discovery 通过 `flagtree_profiler/csrc/include/Driver/Ascend/AscendApi.h` 和对应实现提供。

```text
third_party/proton/test/test_cann_smoke.py
```

兼容旧测试路径的 shim，实际测试实现已经移动到：

```text
third_party/proton/flagtree_profiler/test/test_cann_smoke.py
```

## 顶层 docs 中的变更

原先放在仓库顶层 `docs/` 下的 Proton/CANN 文档已经移动到：

```text
third_party/proton/flagtree_profiler/docs/
```

这样顶层 `docs/` 保持为 FlagTree 项目级文档，Profiler 专项文档集中在 `flagtree_profiler/docs/`。

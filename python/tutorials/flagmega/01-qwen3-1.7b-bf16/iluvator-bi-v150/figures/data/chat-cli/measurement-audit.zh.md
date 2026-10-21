# CLI decode 对比的采集审计

此前图中 CLI 的 32/128-token decode 为 8.048/8.119 ms，而 FlagMega-vLLM 为
7.236/7.303 ms。旧 CLI 的三个轮次在同一进程连续执行，先把四档输入全部预热；
vLLM 每个轮次使用新进程，并在每组请求前预热。CLI 的 token-scan prefill 长时间
占满 GPU，测量顺序与设备状态不同。图表合并只检查产物、依赖与 token，未检查
进程生命周期和每组开始时的设备条件。这是本次比较方法的缺陷。

## 实验定位

同一最终 C++ 产物、物理 GPU 8、原始输入与完整 64-token 输出，保留模型规模、
输入长度、KV 容量、采样与计时边界。没有改写编译器或生成代码。

`profile_cli_soak.measured.py` 在同一 CLI 进程中先测 32-token 请求，然后连续执行
24 次 2048-token 请求，再测同样的短请求，最后停止 GPU 工作 40 秒后重测。
40 秒等待发生在请求计时之外，仅用于定位设备状态影响。

| 状态 | host-visible decode 中位数 | 图执行均值 | 图外均值 |
|---|---:|---:|---:|
| 持续长请求前 | 6.243 ms | 6.134 ms | 0.100 ms |
| 连续 24 次长请求后 | 8.850 ms | 8.430 ms | 0.101 ms |
| 同进程冷却 40 秒后 | 6.632 ms | 6.528 ms | 0.100 ms |

设备事件括住完整模型与 greedy argmax 图；图外包括 token 上传、主机 token 返回、
同步返回开销与增量文本解码。同步等待模型完成的时间属于图执行，不能再重复算作
图外同步开销。图外差异未解释 2 ms 以上的漂移。

`telemetry.csv.gz` 记录升温与动态频率变化：连续长 prefill 时温度接近 90–92°C，
SM 从 1600 MHz 降到约 1400–1550 MHz；HBM 时钟仍为 1600 MHz。冷却后图执行
恢复，进程和 CUDA Graph 均未重建，说明这次反转主要是设备热态执行变化。
不能仅根据频率把全部差异线性归因于 DVFS；驱动不支持该卡的单独锁频。

独立同卡交替复测 `CLI → FlagMega-vLLM → CLI`：vLLM 的短输入中位数为
7.221/7.292 ms，末次 CLI 的 32-token 中位数为约 6.21 ms，趋势恢复。
这些两档诊断不代替正式四档测量。未经温度控制的完整复测与旧批次都保留在
`superseded-continuous/` 中，不能筛选快样本进入最终图表。

## 修正与回归

位置：`benchmark_chat_cli.py::main` 的进程内多轮与全局预热，
`render_chat_cli_results.py::add_flagos_comparison` 缺少采集生命周期验证。
修正增加全局轮次索引，每轮使用新进程，每组先达到相同起始温度上限 70°C，再
完成该组预热和全部五次测量。`benchmark_conditions.py::scenario_start` 在请求
计时外等待并保存原始温度、频率、显存与利用率；vLLM 使用同一个 helper。
完整请求里的真实 token-scan prefill 和每步同步仍全部计时。

最终三个实现在同卡串行轮换，四档输入、三轮、每轮五次请求，各实现每档
945 个 decode 间隔，保留所有样本。起始温度控制是此图的明确实验条件；
持续长 prefill 的热态退化仍作为诊断结果保留，不声称已修复硬件/驱动问题。

回归 `test_chat_comparison_rejects_shared_process_rounds` 修正前失败
（未抛出异常），修正后通过；不同起始条件也拒绝合并。CPU 测试隐藏 GPU，
硬件测试在性能采集结束后单独运行。原始失败、通过日志及所有诊断保留。

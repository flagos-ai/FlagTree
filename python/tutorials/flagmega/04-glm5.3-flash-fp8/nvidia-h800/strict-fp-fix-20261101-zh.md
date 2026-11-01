# 严格浮点修复与完整 TP4 复验（2026-11-01）

当前源码的完整数值及硬件验收通过。合并后第二个 token 为 `646`、参考为 `284` 的失配，来自 KDA 和 DSA 模板对隐式 FMA 的依赖。现在显式保持原生舍入顺序，`floating_point_contract=strict`、`enable_fp_fusion=False` 保持；模型、26 program/228 调用、block/warps、状态 IO 优化、并行 producer/有序 consumer、双 stream 和 CUDA Graph 保持。

本轮没有重跑 Native/Flag 的 320 正式请求对照；历史 6.4–6.8ms 表格仍属于合并前源码。本轮四上下文各一次 warmup 的 TPOT 仅作诊断。

## 根因、复现与修正

`runtime/module.py::_prepare_external` 在合并提交 `53843322b` 后正确传递 strict 合同，`runtime/nvidia_launch.py::compilation_options` 因此关闭隐式融合；模板此前未显式表达原生计算必须使用的 FMA。这项 strict 处理保留。

- `kda_gated_recurrent_core/local.py.jinja` 的 recurrent decode：原生 K128 点积每线程单独舍入第二个 K4 乘积，再执行三次 FMA，最后以 16/8/4/2/1 butterfly 归约；state 外积也只有一次舍入。修正为显式非 FTZ `mul.rn.f32`、`tl.fma`，固定 subtotal 的 32 部分和布局，再恢复返回向量布局。仅加 FMA 后，融合布局仍把归约改成线程内四组加法及 4/2/1 shuffle；实际 PTX 和分阶段零差异诊断定位了这个原因。返回向量布局还避免与原 coalesced state IO 的 K4 hint 冲突，没有删 hint 或绕过编译器检查。
- `dsa_paged_attention/platforms/nvidia/sm90/online_mma.py.jinja::__online_probability`：原生指数先单独舍入 maximum×scale，再用 FMA 计算 score×scale−maximum_scaled；strict 下原先两乘一减改变了指数、normalizer 及 BF16 输出。现在显式保留这两处边界及 online normalizer 的 FMA。`native_latent_probability_template.py::probability_template` 从原递推表达式仅替换 producer 提供的 tile sum，使长 consumer 保留同一舍入合同。

四卡逐入口隔离先定位 `program_006` 的最早 KDA 状态差异。KDA 修正后，完整输入上仅关闭 006/017 的隐式融合已精确相等，但全 strict 仍失败；二分剩余入口定位 `program_011`，补齐 DSA 后才通过完整 Gold。没有关闭优化、恢复旧浮点选项、手改生成源码、缩小模型或放宽原断言。

最小复现与 fail-before/pass-after：

| 回归 | 输入与实际结果 |
| --- | --- |
| KDA 原生归约/state | seed117、2 head、K128/V128、非零 FP32 初态，rate/log_rate、平面/8-lane；独立 one-warp 参考。旧 strict 有8733/32768 state差异，4 failed/7 passed；修正后13 passed，含 coalesced-state-IO 邻域。 |
| DSA 原生概率 | seed0、16 head/D512、两 token、仅feature0非零、key256/258；QK乘积精确，对照独立原生内核。旧 strict 的 slots/compact 各3/8192 BF16输出不同，2 failed/2 passed；修正后4 passed。 |
| 邻域 | KDA/convolution 79 passed；DSA及producer/consumer合同65 passed。旧head-replica测试钩子同步到现有prepare API，全部原输出与保护区断言保留。 |

```sh
conda run -n flagtree pytest python/test/flagmega/codegen/triton/kernels/kda_gated_recurrent_core/test_native_reduction_contract.py python/test/flagmega/codegen/triton/kernels/dsa_paged_attention/test_native_probability_contract.py -q --import-mode=importlib
```

需要已正常安装 NVIDIA/TLE 编译器及 SM90 GPU。完整输入、实际 ABI/常量/属性/选项和旧编译冲突保存在本地证据目录，不读取或复制 Core/cubin/权重载荷作手工检查；PTX/LLIR 来自正常 compiled.asm 文本 API。

## 完整复验及性能范围

原四张 H800、45层/E288/top8/H4096/cache4096/TP4/batch1/8token；正常 final-IR artifact API、正常只读缓存及一份完整模型池。当前源码 `52fc40f2ab04934793c91c4fac3742e9e1e28fd2249b707272759d276fae991d`，语义 `7a073b51c9d72f7383aa0a5763697ee53917de427451b6f57841a4468c43a6c1`。

| 检查 | 实际结果 |
| --- | --- |
| 正常完整发布 | 退出0，115.698s；228份完整kernel spec、ABI、descriptor/offset table与失败源码逐项相等 |
| 正常 strict CPU 编译 | 8 worker/finalize全部退出0，78.764s；011冷编译66.195s，其余25入口命中缓存 |
| 原完整 Gold | 退出0，292.777s；四rank各100完整请求、各792次后续逐字节状态比较 |
| 原状态与上下文 | 1216 decode及4864 ctx32 prefill Gold签名比较；32/128/1024/2048原token、finite、cache-tail检查通过 |
| 执行与身份 | 实际26 GPU key=CPU key；228调用零spill；双stream、prefill/decode Graph；源码冻结、Core stat与各rank实际映射通过 |
| 硬件独立核验 | 原四卡请求窗口实测1980MHz、独占、无thermal；821条窗口观测 |
| KDA局部 B/F/F/B | 128样本全保留，55.152543→55.641703µs（+0.8869%） |
| DSA局部 B/F/F/B | 1980MHz、128样本全保留，12.367318→12.407217µs（+0.3226%） |

局部修复代价保留，两个微测分别为2-head K128/V128 KDA和16-head/D512两token DSA，不代表完整入口或整模TPOT。四上下文单次warmup最慢rank TPOT：`{'32': 6.485914570345942, '128': 6.5531888602501045, '1024': 6.697459429103349, '2048': 6.891612429171801}` ms；没有新Native对照、正式分布或性能通过声明。

上一轮相同修正源码的 Gold 数值通过、472.327s，但独立硬件检查发现整个计时窗口实测1830MHz，旧样本保留并拒绝作为1980MHz结果。GPU0相同Graph负载、仅切换锁频模式，模式0实测1830、模式1实测1980；本机驱动580.167.08需实测核对，命令成功不能替代NVML记录。最终以reset后模式1复验，原1980MHz门槛及原数值断言保持。

```sh
nvidia-smi -i 1,2,3,4 -rgc
nvidia-smi -i 1,2,3,4 -lgc 1980,1980 -m 1
```

以上为本机原TP4的验收设置，其他环境仍需核对实际频率。正常模型装载仍为分钟级，源码生成、冷编译和装载时间分别记录，装载阶段嵌套耗时不重复相加。

证据：[中文迭代日志](.local/iteration-log-zh.md)、[修复证据目录](.local/fix-strict-fp-20261101-a4a795ec6/)。关键回执为 `native-state-strict-fail-before.log`、`native-state-final-Root-regression.log`、`DSA-native-probability-fail-before.log`、`DSA-all-final.log`、`fixed-final-codegen-launch-contract.json`、`fixed-KDA-butterfly-proof.json`、`fixed-MLA-probability-arithmetic-proof.json`、`prewarm-v7/input-proof.json`、`gold-v3/exit.json`及`gold-v3/independent-audit.json`。

范围限制：本轮没有修改原生C++或其他后端；warp_k仍遵循已有SM90/K128合同，非warp_k state FMA及邻近prefill/归约路径有回归覆盖。原始IR fresh的7项替换/6项合同、完整物理catalog、TLE inline argument hints与全量CPU收集失败仍按[旧验收记录](reacceptance-20261101-zh.md)保留；正常final-IR GPU复验不替代这些检查。历史生成源码快照与正式图表保持。`reports/flagmega`不存在。

# main c8cc 上的优化与验收（2026-11-03）

当前已 rebase 到 main `c8cc6023a45aa875bda85c34e63008b0c6c037aa`。
84 个 feature 补丁的 range-diff 全部相同；原分支与未提交修改的备份保留。
本轮完整模型 decode TPOT 已通过严格 vLLM 门槛；完整请求含 prefill 仍较慢。

新 Core20 通过正常 CMake 构建，135 个构建目标耗时 113.147 秒。
输出到独立目录，使用正常 Python importer 与 SourceLoader；没有手动读取、复制或修改
旧 Core、checkpoint 或 FP8 payload。主干新增的 swizzled-dot 行为限定 PPU，
不将它记为 H800 性能优化。

## 当前优化与验证

完整模型仍为 45 层、E288/top8、H4096、cache4096、TP4/batch1、生成8token。
共有26个独立program源码、228次调用，使用按program声明的4/8warps和两条stream。
CUDA Graph、完整state与greedy token检查保持；编译通过与GPU数值通过分别记录。

- 并行完整 pool 排名在257池以上使用完整512键、原padding及稳定tie规则；
  一般稳定索引压缩使用2048项分段前缀与3项尾部，保留任意invalid孔洞和position过滤。
- indexer 的sequence更新放在最后一次grid前，同一次grid发布全部index/query/sequence。
  typed producer/consumer与指针合同验证后消除重复caller grid。
- 长PV保留原K64浮点累加顺序，完整循环对与尾部tile分别执行；短序列保留原循环。
- LMhead采用4warps、原完整4096列归约及全部有效vocab输出；mHC完整计算保留，
  最终4096元素归一化由8个CTA各处理512元素。
- 正常SourceLoader的recipe作用域隔离与声明的关键路径stream调度保持完整ABI、DAG、
  lifetime、资源与输入身份验证。

v22b正常生成及CPU runtime退出0，生成120.351秒、外层131.469秒。
新Core20上26入口均零spill，预热80.728秒；同Core的上一版预热73.552秒。
只有program011的源码AST改变，其余25个相同。完整TP4实际26 GPU key与新Core20的CPU key一致，
228调用零spill；每版各100请求、全state/greedy8/四上下文、两stream和Graph通过。
真实外层退出0、771.644秒；独立审计退出0、2.629秒。

扩展CPU回归209 passed/12 skipped。首次混合运行因lifetime测试激活整个recipe，
将合法的fence-mask共享hook留给后续原生SDK测试而失败；修正测试hook生命周期后原组合通过。
原生fence位置断言与完整recipe的独立进程安装检查均保留，未修改模型或编译器语义。
新main的swizzled-dot与4项相邻MLIR检查通过。

## 因果位置与可运行回归

| 位置、函数及触发条件 | 因果修改与范围检查 |
| --- | --- |
| `joint_dsa_absorbed_pool_scores.py::wrapper`，完整indexer末尾及紧随其后的caller grid | 原末尾grid之后才更新sequence，caller grid再发布它。将唯一CTA0的sequence更新移到最后grid前，由同一次grid发布index/query/sequence；`indexer_complete_publication.py::remove_redundant_caller_grid`验证typed producer/consumer、全部指针和四次内部grid合同，才移除重复caller grid。原rank捕获grid和完整索引选择保留。 |
| `paired_pv_loop.py::transform`，compact count>=1024的K64 PV循环 | 完整128项循环对与最后不足128项的原K64 tile分别执行，保持每个tile的原数学正文、顺序、遮罩与两个同步grid。count<1024保留原循环。GPU136边界用例、96实际key通过，2048/2051项微测分别降低4.44%/2.05%。 |
| `parallel_dsa_pool_sort.py::selection`，257–512池的完整排名 | 原阈值512让中等池继续走较慢排名分支；并行分支仍处理完整512键、原invalid padding与稳定tie规则。462个GPU用例在4/8warps均逐字节一致，未丢弃任何池或top8。 |
| `latent_segmented_compaction.py::transform`，任意invalid孔洞及position过滤 | 完整2048项分段前缀加3项尾部，保持稳定原顺序、4096容量和原workspace合同。136个GPU用例、80实际key通过；微测约降低2.87%，整模单请求未重现，不当作整模收益。 |
| `python/triton/flagmega/ir/program_order.py::_critical_path_stream_indices`，重复非对称fork | 原最小链覆盖只优化stream数量，长依赖链可能反复跨stream。最大匹配保持最少stream和原DAG，代价只用于保留关键路径与直接邻接；codegen/runtime按同一声明策略验证。未声明策略的旧artifact继续按原v1验证；未知策略拒绝。 |
| `test_concurrent_call_lifetimes.py`，与原生SDK入口共享测试混合运行 | 测试曾激活整个recipe并留下合法fence-mask共享hook。只激活被测lifetime hook并逐用例恢复；完整recipe安装仍在独立子进程检查。原组合208 passed/12 skipped/1 failed，修正后209 passed/12 skipped；未修改原生fence语义或断言。 |

最小复现与拒绝错误合同的回归已放入源码：
[indexer发布](local_optimizations/test_indexer_complete_publication.py)、
[PV循环覆盖](local_optimizations/test_paired_pv_loop.py)、
[完整pool排序](local_optimizations/test_parallel_dsa_pool_sort.py)、
[稳定索引压缩](local_optimizations/test_latent_segmented_compaction.py)、
[关键路径与旧artifact合同](../../../../test/flagmega/runtime/test_critical_program_chains.py)。
例如从本教程目录执行以下CPU回归；GPU完整验证使用下文实际26key/228调用和Gold证据。

```sh
conda run -n flagtree pytest -q \
  local_optimizations/test_indexer_complete_publication.py \
  local_optimizations/test_paired_pv_loop.py \
  local_optimizations/test_parallel_dsa_pool_sort.py \
  local_optimizations/test_latent_segmented_compaction.py
```

indexer新发布合同和PV循环合同的首次回归分别退出1，2.80/3.11秒；
PV首次生成因误把未渲染Jinja边界当作literal而在CPU阶段失败，保留原失败。
精确匹配未渲染边界后，7项相关文件28个回归通过；没有绕过模板验证。
indexer GPU390个用例同时检查完整2051索引与advance后Base，各CTA consumer snapshot相等，
包含eager/Graph、NaN/Inf/tie与padding边界；8实际key零spill。
发布微测降低12.85–15%，属于算子收益，未把它当作已证明的整模增益。
这两项算子微测先在Core19完成；Core20上重新执行了本页完整模型与实际key验收。

## 本轮完整八段结果

| 上下文 | FlagMega TPOT | vLLM TPOT | 降低 | 四Flag对四Native最小余量 |
| --- | ---: | ---: | ---: | ---: |
| 32 | 6.174020 ms | 6.764799 ms | 8.733% | 561.876 μs |
| 128 | 6.255618 ms | 6.770960 ms | 7.611% | 489.858 μs |
| 1024 | 6.404630 ms | 6.821416 ms | 6.110% | 399.302 μs |
| 2048 | 6.485827 ms | 6.906239 ms | 6.087% | 366.811 μs |

| 段 | ctx32 | ctx128 | ctx1024 | ctx2048 |
| --- | ---: | ---: | ---: | ---: |
| A_N0 | 6.737184 | 6.773185 | 6.822133 | 6.919133 |
| A_F0 | 6.175171 | 6.255430 | 6.406806 | 6.487501 |
| A_F1 | 6.175308 | 6.259561 | 6.408074 | 6.489021 |
| A_N1 | 6.770337 | 6.781908 | 6.965185 | 6.909568 |
| B_F0 | 6.167364 | 6.251212 | 6.398678 | 6.483485 |
| B_N0 | 6.745226 | 6.749419 | 6.807376 | 6.855832 |
| B_N1 | 6.776375 | 6.773144 | 6.810165 | 6.911380 |
| B_F1 | 6.169726 | 6.256107 | 6.406109 | 6.484416 |

以上逐段数值单位为ms，每项取10个正式请求的中位数；每个Flag请求先取四rank最慢wall TPOT。
双方版本、Graph与batch设置未改，原版vLLM 0.31.0 / Torch 2.13.0+cu130 / CUDA 13.0。
合并中位数没有覆盖逐段失败。全部384条标量请求、逐段中位数和实际退出回执见
[本轮公开结果](figures/summary-20261103.json)；完整rank报告及独立审计见
[本地原始审计](.local/beat-vllm-20261103-77ed634c5/formal-publication-pair-main-v22b/independent-audit.json)。

完整请求含prefill的中位数如下，decode胜出不代表这一指标胜出：

| 上下文 | FlagMega完整请求 | vLLM完整请求 |
| --- | ---: | ---: |
| 32 | 242.840 ms | 173.685 ms |
| 128 | 844.319 ms | 173.881 ms |
| 1024 | 6583.170 ms | 288.879 ms |
| 2048 | 13231.348 ms | 526.540 ms |

## 本轮正常生成入口

本轮输入为已保存的完整final IR，正常入口是
[emit_verified_candidate.py](emit_verified_candidate.py)。源码hash为
`7dd00d8a1c5b9c1c9d50d4ce13151b4327bc0ef9d6ea1b21c2846a92b706295b`，
语义hash为`9f0de10350a5ad33851d22880f6a7aaf31b7756f92099e417b70fd35ca57e087`。
入口严格核对完整45层模型、原readonly数据身份、26入口与228调用，并通过正常CPU runtime。
需要当前main正常构建的NVIDIA/TLE环境、原checkpoint及完整readonly cache。

```sh
conda activate flagtree
export GLM_TUTORIAL="$PWD/python/tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800"
python "$GLM_TUTORIAL/emit_verified_candidate.py" \
  --final-ir /path/to/complete-final-ir.py \
  --checkpoint /path/to/original-checkpoint \
  --integrity-report /path/to/checkpoint-digests.json \
  --rdata-cache /path/to/readonly-cache \
  --output "$GLM_TUTORIAL/.local/current-main-candidate"
```

使用与本轮相同的完整输入才要求相同的源码与语义hash。
`optimize.py`及根目录`generated_kernels.py`的fresh快照属于历史交付合同，
本轮没有放宽该合同，也没有用显式final-IR生成替代fresh原始输入性能验收。

## 性能门槛与已有失败

本轮在停止额外CPU编译和GPU微测的条件下完成 NFFN/FNNF 八段对照：
每上下文每段2次warmup与10次正式请求，保留全部384原始请求、320正式请求，未删除outlier。
四个Flag段在每个上下文均低于四个Native段，逐顺序严格门槛也通过。
正式控制器真实退出0，2525.095秒；独立审计退出0，1.536秒，重新核对4030份源码输入。
完整45层模型、Graph、原GPU1–4 UUID、1980MHz/独占/无thermal、26实际key与CPU key一致、
228调用零spill全部通过，decode/prefill Gold签名分别为4864/19456。

此前v19c在Core19上的完整八段数值、硬件与实际编译key审计通过，性能门槛失败：
ctx1024最快Native为6.384869ms，Flag各段为6.441–6.463ms；
ctx2048最快Native为6.436596ms，Flag各段为6.473–6.485ms。
本轮Flag长上下文仍未超过这两个历史最快Native中位数；跨轮Native变化的根因尚未确定，
不能宣称已建立对历史最佳结果的稳定优势。后续较慢Native段及快速首段均保留，没有挑选样本或删除outlier。
合并中位数约快5.65%不能据此宣布通过。

局部indexer发布和PV配对分别减少约1.0–1.2μs及0.49–1.04μs，属于算子微测结果。
v21的索引压缩微测收益未在一次完整模型诊断请求中重现，未当作模型收益。
v22b与v21同Core配对的ctx2048单请求为6.506/6.491ms，仍未重现局部收益；
该诊断不替代正式多轮性能验收。
router新分组、更多PV消费者和多种MMA布局的负结果均保留，未接入当前源码。

TPOT取完整decode wall的四rank最慢值，包含state与token交接，排除prefill及首个生成token；
Native使用原版vLLM首个到最后一个生成token的TPOT。
完整请求含prefill的结果单独记录，decode胜出不等于端到端胜出。

每轮因果修正、正负性能变化和原始证据见[中文迭代日志](.local/iteration-log-zh.md)。

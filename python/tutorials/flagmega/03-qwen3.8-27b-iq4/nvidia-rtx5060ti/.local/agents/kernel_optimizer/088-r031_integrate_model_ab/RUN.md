# R031：R27 GLU + R17 GDN PREP

本目录是独立集成主体；分支 `r031/r27-r17-model-ab`。`integration-source.patch` 是相对 pinned R27 source-tree 的 15 个原生接入修改，不是生成源码、PTX 或 cubin 补丁。`source-closure.json` 固定所有库源码、宿主脚本、原生编译器及上游资格收据。R16 未被选择。

新主体：`build-attempt2/artifact`，策略 `flagmega.r031-model-policy/v1`，GDN ABI `r031.gdn.native-model/v1`，包版本 `v11-native-model-r031`。原生 `load` / `ModelContext` 按该路径显式选择主体。

冻结回退：`../048-r004_v6_model_integration/package-attempt1`，策略 `r004_column_v6`。`ab_worker.py` 的 `paths['r004']` 已显式使用此路径；部署调用方将 artifact 路径选回该路径即可回退。没有修改冻结 revision、旧策略、旧源码、旧签名或旧 cubin。源路径导致重新编译的 debug ELF 节不同，机器码、常量和启动节全部相同，证据为 `fallback-machine-audit.json`。

以下命令从本目录执行；环境使用真实的 `/mnt/home-nas` 路径。GPU 命令独占 GPU0，计时不使用 ncu。

```sh
export PYTHONPATH=/mnt/home-nas/work/repo/flagtree/python
export CUDA_VISIBLE_DEVICES=0
task_python=/home/sunnycase/miniconda3/envs/flagtree/bin/python

# 仅从已保存证据重算统计、核对逐行 SHA256；不启动 GPU。
"$task_python" verify_delivery.py

# 显式新输出目录，重新核对当前集成主体的 MODEL_TOKEN10 和资源。
flock /tmp/flagmega-gpu0.lock "$task_python" bootstrap.py qualify_model.py model-gates-reproduction

# 单个新的独立进程：三主体 × 两场景 × 三轮，完整 NUM/TOKEN 门禁。
flock /tmp/flagmega-gpu0.lock "$task_python" bootstrap.py ab_worker.py ab-reproduction 0
```

复现目录必须尚不存在；该单进程复现不覆盖正式三进程证据，也不能单独形成新的整模型性能结论。正式三进程为 `ab-p3/ab-p4/ab-p5`，全部原始 timing、失败的 `ab-p1/ab-p2` 及三 seed 的全行 SHA256 均保留。`analyze_ab.py` 的正式汇总读取这五个既有目录，包含不完整进程的零剔除敏感性审计；不填补缺失测量。

原始轨迹和全行复核结果见 `delivery-verification.json`；冻结证据的 97,909 个文件复核见 `protected-verification.json`。主交付为 `REPORT.md`、`integration-ab-receipt.json`，R27/R17 各有独立 subject receipt。

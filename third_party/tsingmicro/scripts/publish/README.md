# Tsingmicor triton 版本说明

## 版本包说明

### 文件

- triton-3.3.0-cp310-cp310-linux_x86_64.whl
- flag_gems-3.0-cp310-cp310-linux_x86_64.whl
- flaggems_tests
- flaggems_benchmark
- pack
- scripts
  - install.sh
  - run_tsingmicro.sh
  - base_run.sh
  - offline_python_deps.sh
  - run_flaggems_on_multicards.sh
  - requirements_ts.txt
- README.md
- code_version.txt


### 依赖组件

- llvm: [制品库下载链接](http://172.50.1.66:8082/artifactory/tx8-generic-dev/triton/tools/llvm-a66376b0-ubuntu-x64.tar.gz)
- pytorch2.7等whl离线安装包: [制品库下载链接](http://172.50.1.66:8082/artifactory/tx8-generic-dev/triton/offline_pkgs/offline_pkgs_v5.3.0.tar.gz)
- docker: hub.tsingmicro.com/tx8/ubuntu/v5.7.0.0524:tsingmicro_release
	- tx8_deps: [制品库下载链接](http://172.50.1.66:8082/artifactory/tx8-generic-dev/triton/tx8_depends/tx8_depends_dev_20260507_104051.tar.gz)
	- torch_txda: [制品库下载链接](http://172.50.1.66:8082/artifactory/tx8-generic-dev/torch_txda/torch_txda-0.1.0%2B20260416.b8f53e8a-cp310-cp310-linux_x86_64.whl)，版本包pack目录下已包含。
    - txops: [制品库下载链接](http://172.50.1.66:8082/artifactory/tx8-generic-dev/torch_txda/temp/txops-0.1.0%2B20260508.60287151-py3-none-any.whl)，版本包pack目录下已包含。

注意: tx8_deps、torch_txda、txops迭代较快, 每个版本最终配套的以Triton项目发布为准.

## 版本运行说明

1. 安装

```shell
# 下载依赖组件(llvm, offline_pkgs, tx8_deps)到triton版本包目录
# 将该目录映射到docker容器内(容器内能访问), 在容器内该目录下执行以下安装命令
bash ./scripts/install.sh venv
```
或
```shell
# 由于环境可能多种多样，install.sh无法适配多种场景，可以选择手动安装，步骤如下
# 1. 下载依赖组件(llvm, offline_pkgs, tx8_deps)到triton版本包目录
# 2. tar解压llvm, offline_pkgs, tx8_deps到triton版本包目录
# 3. 将triton版本包目录映射到docker容器内, 进入docker容器内该目录下
# 4. 创建python虚拟环境, 激活虚拟环境
    python3 -m venv .venv --prompt triton
    source .venv/bin/activate
	# 如若提示未安装python3.10-venv执行以下命令安装后再进行创建
	apt update
	apt install python3.10-venv -y
# 5. 安装pytorch2.7等依赖
    bash ./scripts/offline_python_deps.sh -i -r ./scripts/requirements_ts.txt -d ./offline_pkgs
# 6. 安装txops、torch_txda
    pip3 install ./pack/txops*.whl
    pip3 install ./pack/torch_txda*.whl
# 7. 安装triton
    pip3 install ./triton*.whl
# 8. 安装flaggems
    pip3 install ./flag_gems*.whl
```

2. 运行简单case,检测环境是否OK

```shell
# 运行无异常,case全部success才说明环境部署安装OK
bash ./scripts/run_tsingmicro.sh pytest ./flaggems_tests/test_unary_pointwise_ops.py::test_accuracy_abs --ref cpu --mode quick
```

3. 批量运行已支持的flaggems算子

```shell
python3 ./flaggems_tests/test_flag_gems_all.py --test_set ci_ops --device 0
```
4. 多卡运行已支持的flaggems算子
```shell
bash ./scripts/run_flaggems_on_multicards.sh ci_ops 3 1 0
```

5. benchmark flaggems算子性能数据
```shell
# 运行前设置以下环境变量, 并确保Tsm_profiler_xxx_x86_64_installer.run已install。
export TSM_PROFILER_EN=1
# 运行以下命令
bash ./scripts/run_tsingmicro.sh pytest ./flaggems_benchmark/ -k test_general_unary_pointwise_perf[abs-abs-dtypes0] --mode txda --level core --metrics latency --dtypes float16
# 运行后屏幕log会输出flaggems benchmark性能
```

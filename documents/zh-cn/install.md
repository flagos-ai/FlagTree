[English](../en/install.md)

## 从源代码安装

安装依赖（注意使用正确的 Python 3.x 版本）：

```shell
apt install zlib1g zlib1g-dev libxml2 libxml2-dev  # ubuntu
cd python; python3 -m pip install -r requirements.txt
```

构建安装（网络畅通环境下推荐使用）：

```shell
cd python
export FLAGTREE_BACKEND=<backend>
python3 -m pip install . --no-build-isolation -v
python3 -m pip install show flagtree
cd; python3 -c 'import triton; print(triton.__path__)'
```

> [!Tip]
> 自动下载依赖库的速度可能受限于网络环境，编译前可自行下载至缓存目录 `~/.flagtree`。
> 缓存目录可通过环境变量 `FLAGTREE_CACHE_DIR` 修改。
> 无需自行设置 `LLVM_BUILD_DIR` 等环境变量。

## 非源码安装
<!--Custom anchor ID-->
<div id="no-source-install"></div>

如果不希望从源码安装，可以直接拉取安装 Python Wheel 包（支持部分后端）。

先安装 PyTorch，然后执行下列命令：

```shell
python3 -m pip uninstall -y triton
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple --trusted-host=https://resource.flagos.net"
```

| 后端     | 安装命令                                                   | Triton 版本 | 支持的 Python 版本 |
| -------- | ---------------------------------------------------------- | ----------- | ------------------ |
| nvidia   | python3 -m pip install flagtree==0.3.0rc1 $RES             | 3.1         | 3.10, 3.11, 3.12   |
| nvidia   | python3 -m pip install flagtree==0.3.0rc1+3.2 $RES         | 3.2         | 3.10, 3.11, 3.12   |
| nvidia   | python3 -m pip install flagtree==0.3.0rc1+3.3 $RES         | 3.3         | 3.10, 3.11, 3.12   |
| iluvatar | python3 -m pip install flagtree==0.3.0rc2+iluvatar3.1 $RES | 3.1         | 3.10               |
| mthreads | python3 -m pip install flagtree==0.3.0rc3+mthreads3.1 $RES | 3.1         | 3.10               |
| ascend   | python3 -m pip install flagtree==0.3.0rc1+ascend3.2 $RES   | 3.2         | 3.11               |
| hcu      | python3 -m pip install flagtree==0.3.0rc2+hcu3.0 $RES      | 3.0         | 3.10               |
| enflame  | python3 -m pip install flagtree==0.3.0rc1+enflame3.3 $RES  | 3.3         | 3.10               |

其中 Flagtree 版本号为对应的 GIT 标签。

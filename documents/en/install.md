[中文](../zh-cn/install.md)

## Installation

### Install from source

Install dependencies (ensure you use the correct python3.x version):

```shell
apt install zlib1g zlib1g-dev libxml2 libxml2-dev  # ubuntu
cd python; python3 -m pip install -r requirements.txt
```

Build and install:

```shell
cd python
export FLAGTREE_BACKEND=<backend>
python3 -m pip install . --no-build-isolation -v
python3 -m pip show flagtree
cd; python3 -c 'import triton; print(triton.__path__)'
```

> [!Tip]
> Automatic dependency library downloads may be limited by network conditions.
> You can manually download the dependent libraries to the cache directory `~/.flagtree`.
> The cache directory can be customized using the `FLAGTREE_CACHE_DIR` environment variable.
> You don't need to manually set LLVM environment variables such as `LLVM_BUILD_DIR`.

### Install from Wheel

In addition to build from source, you can also directly pull and install the Python Wheels files
(supports some backends).

First install PyTorch, then execute the following commands:

```shell
python3 -m pip uninstall -y triton
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple --trusted-host=https://resource.flagos.net"
```

| Backend  | Install Command                                            | Triton Version | Python version    |
| -------- | ---------------------------------------------------------- | -------------- | ----------------- |
| nvidia   | python3 -m pip install flagtree==0.3.0rc1 $RES             | 3.1            | 3.10, 3.11, 3.12  |
| nvidia   | python3 -m pip install flagtree==0.3.0rc1+3.2 $RES         | 3.2            | 3.10, 3.11, 3.12  |
| nvidia   | python3 -m pip install flagtree==0.3.0rc1+3.3 $RES         | 3.3            | 3.10, 3.11, 3.12  |
| iluvatar | python3 -m pip install flagtree==0.3.0rc2+iluvatar3.1 $RES | 3.1            | 3.10              |
| mthreads | python3 -m pip install flagtree==0.3.0rc3+mthreads3.1 $RES | 3.1            | 3.10              |
| ascend   | python3 -m pip install flagtree==0.3.0rc1+ascend3.2 $RES   | 3.2            | 3.11              |
| hcu      | python3 -m pip install flagtree==0.3.0rc2+hcu3.0 $RES      | 3.0            | 3.10              |
| enflame  | python3 -m pip install flagtree==0.3.0rc1+enflame3.3 $RES  | 3.3            | 3.10              |


The FlagTree version string are the GIT tags.

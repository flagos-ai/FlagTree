# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import platform
from pathlib import Path

_CUDA_ARCHIVE_BASE_URL = "https://developer.download.nvidia.com/compute/cuda"
_TILEIR_TOOLCHAIN_VERSION = "13.4.59"


def _cuda_archive_platform():
    system = {"Linux": "linux", "Darwin": "linux"}[platform.system()]
    arch = {"arm64": "sbsa", "aarch64": "sbsa"}.get(platform.machine(), platform.machine())
    return system, arch


def register_cache(cache, flagtree_backend, check_env, set_llvm_env):
    if flagtree_backend != "tileir":
        return

    system, arch = _cuda_archive_platform()
    cache_root = Path(cache.sub_dirs[flagtree_backend])
    install_root = "third_party/nvidia/backend/tileir_cuda"

    bundle_tileiras = not check_env("TRITON_TILEIRAS_PATH")
    cache.store(
        file="cuda_tileiras",
        condition=bundle_tileiras,
        url=(f"{_CUDA_ARCHIVE_BASE_URL}/redist/cuda_tileiras/{system}-{arch}/"
             f"cuda_tileiras-{system}-{arch}-{_TILEIR_TOOLCHAIN_VERSION}-archive.tar.xz"),
        version=_TILEIR_TOOLCHAIN_VERSION,
    )
    cache.store(
        files=("tileiras", ),
        condition=bundle_tileiras,
        copy_src_path=cache_root / "cuda_tileiras" / "bin",
        copy_dst_path=f"{install_root}/bin",
    )

    cache.store(
        file="cuda_nvcc",
        condition=True,
        url=(f"{_CUDA_ARCHIVE_BASE_URL}/redist/cuda_nvcc/{system}-{arch}/"
             f"cuda_nvcc-{system}-{arch}-{_TILEIR_TOOLCHAIN_VERSION}-archive.tar.xz"),
        version=_TILEIR_TOOLCHAIN_VERSION,
    )
    cache.store(
        files=("ptxas", ),
        condition=True,
        copy_src_path=cache_root / "cuda_nvcc" / "bin",
        copy_dst_path=f"{install_root}/bin",
    )

    cache.store(
        file="libnvvm",
        condition=True,
        url=(f"{_CUDA_ARCHIVE_BASE_URL}/redist/libnvvm/{system}-{arch}/"
             f"libnvvm-{system}-{arch}-{_TILEIR_TOOLCHAIN_VERSION}-archive.tar.xz"),
        version=_TILEIR_TOOLCHAIN_VERSION,
    )
    cache.store(
        files=("lib64", "libdevice"),
        condition=True,
        copy_src_path=cache_root / "libnvvm" / "nvvm",
        copy_dst_path=f"{install_root}/nvvm",
    )

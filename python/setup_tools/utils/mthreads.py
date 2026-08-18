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


def skip_package_dir(package):
    return package == "triton" or package.startswith("triton.")


def get_package_dir():
    return {}


def register_cache(cache, flagtree_backend, check_env, set_llvm_env):
    is_mthreads = "mthreads" == flagtree_backend
    cache.store(
        file="mthreads-llvm22",
        condition=is_mthreads,
        url="https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/mthreads-llvm22-x64_v0.5.1.tar.gz",
        pre_hook=lambda: check_env("LLVM_SYSPATH"),
        post_hook=set_llvm_env,
    )
    cache.store(
        file="mthreads_local_binary",
        condition=is_mthreads,
        url="https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/mthreads_local_binary_v0.6.0.tar.gz",
    )
    cache.store(
        files=("ld.lld", "llc"),
        condition=is_mthreads,
        copy_src_path=f"{cache.dir_path}/{flagtree_backend}/mthreads_local_binary",
        copy_dst_path=f"third_party/{flagtree_backend}/backend/bin",
    )

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

import os
import shutil
from pathlib import Path


def register_cache(cache, flagtree_backend, check_env, set_llvm_env):
    is_metax = "metax" == flagtree_backend
    cache.store(
        file="metax-llvm19",
        condition=is_metax,
        url="https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/metax-llvm19-3.8.0.6-x86_64_v0.6.0.tar.gz",
        pre_hook=lambda: check_env("LLVM_SYSPATH"),
        post_hook=set_llvm_env,
    )
    cache.store(
        file="metaxTritonPlugin.so",
        condition=is_metax and not os.environ.get("FLAGTREE_PLUGIN"),
        url="https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/metaxTritonPlugin-cpython3.12-x86_64_v0.6.1.tar.gz",
        copy_dst_path=f"third_party/{flagtree_backend}",
        md5_digest="afb7ab8f",
    )


def install_extension(*args, **kargs):
    return

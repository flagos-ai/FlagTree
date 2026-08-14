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


def register_cache(cache, flagtree_backend, check_env, set_llvm_env):
    cache.store(
        file="llvm-f6ded0be-ubuntu-x64",
        condition=("thrive" == flagtree_backend),
        url="https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-f6ded0be-ubuntu-x64.tar.gz",
        pre_hook=lambda: check_env("LLVM_SYSPATH"),
        post_hook=set_llvm_env,
    )


def get_backend_cmake_args(*args, **kargs):
    build_ext = kargs['build_ext']
    src_ext_path = build_ext.get_ext_fullpath("triton")
    src_ext_path = os.path.abspath(os.path.dirname(src_ext_path))

    cmake_args = [
        "-DCMAKE_INSTALL_PREFIX=" + src_ext_path,
    ]

    return cmake_args

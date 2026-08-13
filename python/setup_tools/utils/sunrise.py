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
    is_sunrise = "sunrise" == flagtree_backend

    def configure_llvm(path):
        set_llvm_env(path)
        sunrise_cp_bc_files(path)

    cache.store(
        file="sunrise_llvm22_dev_release",
        condition=is_sunrise,
        url="https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/llvm-9027ac27-triton-v3.6.x.tar.gz",
        pre_hook=lambda: check_env("LLVM_SYSPATH"),
        post_hook=configure_llvm,
    )
    cache.store(
        file="sunriseTritonPlugin.so",
        condition=is_sunrise and not os.environ.get("FLAGTREE_PLUGIN"),
        url="https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/sunriseTritonPlugin_v0.6.0.4.tar.gz",
        md5_digest="dd543bcc",
    )


# sunrise
def sunrise_cp_bc_files(path):
    # mkdir -p third_party/sunrise/backend/lib
    lib_dir = Path("third_party/sunrise/backend/lib")
    os.makedirs(lib_dir, exist_ok=True)
    # cp ${LLVM_SYSPATH}/stpu/bitcode/*.bc third_party/sunrise/backend/lib
    bc_dir = Path(path) / "stpu" / "bitcode"
    for bc_file in bc_dir.glob("*.bc"):
        shutil.copy(bc_file, lib_dir)

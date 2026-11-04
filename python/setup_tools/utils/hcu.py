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


def _remove(p):
    if os.path.islink(p) or os.path.isfile(p):
        os.unlink(p)
    elif os.path.isdir(p):
        shutil.rmtree(p)


def copy_tree_overwrite(src, dst):
    os.makedirs(dst, exist_ok=True)
    with os.scandir(src) as it:
        for entry in it:
            s, d = entry.path, os.path.join(dst, entry.name)
            if entry.is_symlink():
                if os.path.lexists(d):
                    _remove(d)
                os.symlink(os.readlink(s), d)
            elif entry.is_dir():
                if os.path.lexists(d) and (os.path.islink(d) or not os.path.isdir(d)):
                    _remove(d)
                copy_tree_overwrite(s, d)
            else:
                if os.path.lexists(d):
                    _remove(d)
                shutil.copy2(s, d)


def _install_after_download(cache, flagtree_backend, file, dst_root):
    base = cache.sub_dirs[flagtree_backend] if flagtree_backend else cache.dir_path
    cached = Path(base) / file
    state = {"fresh": False}

    def pre_hook():
        state["fresh"] = not cached.exists()

    def post_hook(src):
        if not state["fresh"]:
            return
        dst = Path(dst_root) / file
        os.makedirs(dst_root, exist_ok=True)
        print(f"copying {src} to {dst}")
        try:
            if os.path.isdir(src):
                copy_tree_overwrite(src, dst)
            else:
                shutil.copy(src, dst)
        except Exception:
            if os.path.isdir(src):
                shutil.rmtree(src, ignore_errors=True)
            else:
                os.unlink(src)
            raise

    return pre_hook, post_hook


def register_cache(cache, flagtree_backend, check_env, set_llvm_env):
    is_hcu = "hcu" == flagtree_backend
    cache.store(
        file="hcu-llvm22-188a510-glibc2.35-glibcxx3.4.30-ubuntu-x86_64",
        condition=is_hcu,
        url=("https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/"
             "hcu-llvm22-188a510-glibc2.35-glibcxx3.4.30-ubuntu-x86_64.tar.gz"),
        pre_hook=lambda: check_env("LLVM_SYSPATH"),
        post_hook=set_llvm_env,
    )

    pre, post = _install_after_download(cache, flagtree_backend, "aillvm", "/opt/dtk")
    cache.store(
        file="aillvm",
        condition=is_hcu,
        url=("https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/"
             "aillvm_2609221416_g5f9d5e.tar.gz"),
        pre_hook=pre,
        post_hook=post
    )

    cache.store(
        file="libdushmem_device_gfx936.bc",
        condition=is_hcu,
        url="https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/"
            "libdushmem_device_gfx936_bc.tar.gz",
        copy_dst_path="/opt/dtk/dushmem/lib/",
        md5_digest="f6260239")


def install_extension(*args, **kargs):
    return

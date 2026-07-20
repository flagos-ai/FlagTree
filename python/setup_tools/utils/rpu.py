# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
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


def _repo_root():
    return Path(__file__).resolve().parents[3]


def _link_rpu_libtriton_into_main_package():
    root = _repo_root()
    src = root / "third_party" / "rpu" / "python" / "triton" / "_C" / "libtriton.so"
    dst = root / "python" / "triton" / "_C" / "libtriton.so"
    if not src.exists():
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        try:
            if dst.resolve() == src.resolve():
                return
        except FileNotFoundError:
            pass
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    rel_src = os.path.relpath(src, dst.parent)
    try:
        os.symlink(rel_src, dst)
    except OSError:
        shutil.copy2(src, dst)


def skip_package_dir(package):
    return True


def get_resources_url(resource_name):
    return None


def get_resources_hash(resource_name):
    return None


def install_extension(*args, **kargs):
    _link_rpu_libtriton_into_main_package()


def post_install():
    _link_rpu_libtriton_into_main_package()

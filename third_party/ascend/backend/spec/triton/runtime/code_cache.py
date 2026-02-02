# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
# Copyright © 2024 BAAI. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Modifications:
# - 2025-06-03:
#   - init version: e9c7aa71832eb2f897a49ce787e42d5377404a72
#

import functools
import os
import shutil
from pathlib import Path


@functools.lru_cache(maxsize=None)  # this is the same as functools.cache in Python 3.9+
def cache_dir_path() -> Path:
    """Return the cache directory for generated files in flaggems."""
    _cache_dir = os.environ.get("FLAGGEMS_CACHE_DIR")
    if _cache_dir is None:
        _cache_dir = Path.home() / ".flaggems"
    else:
        _cache_dir = Path(_cache_dir)
    return _cache_dir


def cache_dir() -> Path:
    """Return cache directory for generated files in flaggems. Create it if it does not exist."""
    _cache_dir = cache_dir_path()
    os.makedirs(_cache_dir, exist_ok=True)
    return _cache_dir


def code_cache_dir() -> Path:
    _code_cache_dir = cache_dir() / "code_cache"
    os.makedirs(_code_cache_dir, exist_ok=True)
    return _code_cache_dir


def config_cache_dir() -> Path:
    _config_cache_dir = cache_dir() / "config_cache"
    os.makedirs(_config_cache_dir, exist_ok=True)
    return _config_cache_dir


def clear_cache():
    """Clear the cache directory for code cache."""
    _cache_dir = cache_dir_path()
    shutil.rmtree(_cache_dir)

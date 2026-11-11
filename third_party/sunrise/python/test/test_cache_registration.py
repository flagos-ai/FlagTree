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
"""Alternating toolchain revisions must retain their own cached package."""

import ast
import functools
import hashlib
import os
import re
from pathlib import Path
import shutil
import types

import pytest

_ROOT = Path(__file__).resolve().parents[4]


def _cache_class():
    # setup_helper has build-time import side effects. Load the real cache class
    # alone so this host regression cannot download compiler/device packages.
    source = _ROOT / "python/setup_tools/setup_helper.py"
    tree = ast.parse(source.read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FlagTreeCache")
    namespace = dict(__file__=str(source), Path=Path, os=os, functools=functools, hashlib=hashlib, shutil=shutil,
                     flagtree_backend="sunrise")
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("first_revision", ["71bd243f", "ff67ef55"])
def test_cached_llvm_revision_switch_and_return(tmp_path, monkeypatch, first_revision):
    source = _ROOT / "python/setup_tools/utils/sunrise.py"
    source_text = source.read_text()
    source_revision = re.search(r"llvm-([0-9a-f]+)-triton-v3\.6\.x", source_text).group(1)
    monkeypatch.setenv("FLAGTREE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("FLAGTREE_PLUGIN", "1")
    namespace = _cache_class()
    downloaded = []

    class TextPackageDownloader:

        def download(self, url, path, file_name):
            package = Path(path) / file_name
            package.mkdir()
            (package / "package-url.txt").write_text(url)
            downloaded.append(url)

    namespace["downloader"] = TextPackageDownloader()
    cache_type = namespace["FlagTreeCache"]
    other = "ff67ef55" if first_revision == "71bd243f" else "71bd243f"
    paths = []
    for revision in (first_revision, other, first_revision):
        package = f"llvm-{revision}-triton-v3.6.x"
        # Changing the package declaration models another PR/branch using the
        # same backend cache with a different LLVM ABI.
        config = types.ModuleType("sunrise_cache_config")
        code = source_text.replace(source_revision, revision)
        exec(compile(code, str(source), "exec"), config.__dict__)
        monkeypatch.setattr(config, "sunrise_cp_bc_files", lambda path: None)
        cache = cache_type()
        selected = []
        config.register_cache(cache, "sunrise", lambda name: False, selected.append)
        assert len(selected) == 1
        url = (selected[0] / "package-url.txt").read_text()
        assert url.endswith(f"/{package}.tar.gz")
        paths.append(selected[0])
    assert paths[0] == paths[2]
    assert paths[0] != paths[1]
    assert len(downloaded) == 2


@pytest.mark.parametrize("backend,external_llvm", [("nvidia", False), ("sunrise", True)])
def test_unselected_or_external_llvm_does_not_download(tmp_path, monkeypatch, backend, external_llvm):
    source = _ROOT / "python/setup_tools/utils/sunrise.py"
    config = types.ModuleType("sunrise_cache_config")
    exec(compile(source.read_text(), str(source), "exec"), config.__dict__)
    monkeypatch.setenv("FLAGTREE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("FLAGTREE_PLUGIN", "1")
    namespace = _cache_class()

    class UnexpectedDownloader:

        def download(self, **kwargs):
            pytest.fail("unselected or explicitly overridden LLVM must not download")

    namespace["downloader"] = UnexpectedDownloader()
    cache = namespace["FlagTreeCache"]()
    selected = []
    config.register_cache(cache, backend, lambda name: external_llvm, selected.append)
    assert not selected
    assert not cache.cache_files

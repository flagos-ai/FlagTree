# Copyright 2026 FlagOS Contributors
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

import importlib.util
import sys
from pathlib import Path


def _load_cluster_gemm_module():
    repo_root = Path(__file__).resolve().parents[4]
    mod_path = repo_root / "python" / "tutorials" / "tle" / "04-cluster-gemm.py"
    spec = importlib.util.spec_from_file_location("tle_cluster_gemm_tutorial", mod_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cluster_gemm_tutorial_skips_on_pre_sm90_cuda(monkeypatch, capsys):
    mod = _load_cluster_gemm_module()
    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(mod.torch.cuda, "get_device_capability", lambda: (8, 0))

    assert mod._cluster_remote_support_skip_reason() == "cluster+remote path requires sm90+ (Hopper or newer)"

    mod.main(["--m", "16", "--n", "16", "--k", "16", "--no-autotune"])

    captured = capsys.readouterr()
    assert "SKIP: cluster+remote path requires sm90+ (Hopper or newer)" in captured.out

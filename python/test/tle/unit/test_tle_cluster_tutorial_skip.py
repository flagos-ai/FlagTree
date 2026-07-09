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

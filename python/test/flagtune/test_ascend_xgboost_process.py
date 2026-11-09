"""Exercise the production dependency loader in fresh Ascend processes."""
import os
import subprocess
import sys

import pytest

from triton.flagtune._dependencies import _needs_xgboost_isolation


@pytest.mark.parametrize("order", ["torch-first", "flagtune-first"])
def test_ascend_xgboost_native_load_order(order):
    if not _needs_xgboost_isolation():
        pytest.skip("requires Linux Ascend runtime")
    code = '''
import sys
if sys.argv[1] == "torch-first":
    import torch
    tensor = torch.empty(1)
from triton.flagtune._dependencies import require_xgboost
xgb = require_xgboost("subprocess regression")
import torch
import numpy as np
assert require_xgboost("repeated load") is xgb
model = xgb.XGBRanker(n_estimators=2, n_jobs=1)
x = np.array([[0., 1.], [1., 0.]], dtype=np.float32)
model.fit(x, np.array([0., 1.]), group=[2])
assert np.isfinite(model.predict(x)).all()
payload = model.get_booster().save_raw(raw_format="json")
loaded = xgb.XGBRanker()
loaded.load_model(payload)
np.testing.assert_allclose(model.predict(x), loaded.predict(x))
print("training, prediction, reload passed")
'''
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    result = subprocess.run([sys.executable, "-c", code, order], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "reload passed" in result.stdout


def test_flagtune_import_keeps_xgboost_lazy():
    code = '''
import sys
import triton.flagtune
from triton.flagtune import _dependencies
assert "xgboost" not in sys.modules
assert _dependencies._XGBOOST_LIBRARY is None
print("lazy import passed")
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_ascend_foreign_xgboost_import_fails_before_native_use():
    if not _needs_xgboost_isolation():
        pytest.skip("requires Linux Ascend runtime")
    code = '''
import torch
import xgboost
from triton.flagtune._dependencies import require_xgboost, FlagTuneDependencyError
try:
    require_xgboost("foreign import regression")
except FlagTuneDependencyError as exc:
    assert "restart" in str(exc)
    print("actionable error passed")
else:
    raise AssertionError("an already loaded library cannot be isolated retroactively")
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "actionable error passed" in result.stdout

"""Exercise the production dependency loader in fresh Ascend processes."""
import os
import subprocess
import sys

import pytest

from triton.flagtune._dependencies import _needs_xgboost_isolation


@pytest.mark.parametrize("order", ["torch-first", "flagtune-first", "foreign-first", "foreign-used-first"])
def test_ascend_xgboost_native_load_order(order):
    if not _needs_xgboost_isolation():
        pytest.skip("requires Linux Ascend runtime")
    code = '''
import sys
if sys.argv[1] == "foreign-used-first":
    import xgboost
    import numpy as np
    foreign_model = xgboost.XGBRanker(n_estimators=2, n_jobs=1)
    foreign_model.fit(np.array([[0.,1.],[1.,0.]]), np.array([0.,1.]), group=[2])
if sys.argv[1] != "flagtune-first":
    import torch
    tensor = torch.empty(1)
if sys.argv[1] == "foreign-first":
    import xgboost
foreign = sys.modules.get("xgboost")
from triton.flagtune._dependencies import require_xgboost
xgb = require_xgboost("subprocess regression")
import torch
import numpy as np
assert require_xgboost("repeated load") is xgb
if foreign is not None:
    assert sys.modules["xgboost"] is foreign
    assert xgb is not foreign
model = xgb.XGBRanker(n_estimators=2, n_jobs=1)
x = np.array([[0., 1.], [1., 0.]], dtype=np.float32)
model.fit(x, np.array([0., 1.]), group=[2])
assert np.isfinite(model.predict(x)).all()
payload = model.get_booster().save_raw(raw_format="json")
loaded = xgb.XGBRanker()
loaded.load_model(payload)
np.testing.assert_allclose(model.predict(x), loaded.predict(x))
if sys.argv[1] == "foreign-used-first":
    assert np.isfinite(foreign_model.predict(x)).all()
print("training, prediction, reload passed")
'''
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    result = subprocess.run([sys.executable, "-c", code, order], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "reload passed" in result.stdout
    assert "Exception ignored" not in result.stderr


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

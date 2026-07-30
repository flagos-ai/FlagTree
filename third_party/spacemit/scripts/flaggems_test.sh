!/bin/bash

set -e
python3 -m pytest tests/test_v1_ops.py --ref=cpu
python3 -m pytest tests/test_v2_ops.py --ref=cpu
python3 -m pytest tests/test_v3_ops.py --ref=cpu
python3 tests/model_resnet18_test.py

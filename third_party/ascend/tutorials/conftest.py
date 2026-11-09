# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import pytest


def pytest_collect_file(parent, file_path):
    if file_path.suffix == ".py" and file_path.name[0].isdigit() and file_path.name != "__init__.py":
        return pytest.Module.from_parent(parent, path=file_path)
    return None

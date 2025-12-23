[中文版](../zh-cn/CONTRIBUTING_cn.md)

## Contributor Guide

Thank you for your interest in FlagTree!
We use GitHub to host code, manage issues, and handle pull requests.
Before contributing, please read the following guideline.

### Bug Reports

Please use GitHub Issues to report bugs. When reporting a bug, include:

- a concise summary
- steps to reproduce the bug
- specific and accurate descriptions
- sample code if possible (this is particularly helpful)

### Code Contributions

When submitting a pull request, contributors should describe the changes made and the rationale behind them.
If possible, provide corresponding tests.
Pull requests require approval from __ONE__ team member before merging and must pass all continuous integration checks.

### Code Formatting

We use [`pre-commit`](https://pre-commit.com/) for code formatting checks:

```shell
python3 -m pip install pre-commit
cd ${YOUR_CODE_DIR}/flagtree
pre-commit install
pre-commit
```

### Unit Tests

We use [PyTest](https://pytest.org) to drive unit tests.
After installation, you can run unit tests in the backend directory:

```shell
cd third_party/<backend>/python/test/unit
python3 -m pytest -s
```

### Backend Integration

Please contact the core development team for backend integration.
<!--TODO(Qiming): Add MAINTAINERS.md -->

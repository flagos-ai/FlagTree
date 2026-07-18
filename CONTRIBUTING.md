<!--
 Copyright 2026 FlagOS Contributors

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 -->

[[中文版](./CONTRIBUTING_cn.md)|English]

# FlagTree Contributor Guide

Thank you for your interest in FlagTree! We use GitHub to host code, manage issues, and handle pull requests. Before contributing, please read the following guidelines.

## Bug Reports

Please use GitHub Issues to report bugs. When reporting a bug, include:
- A concise summary
- Steps to reproduce
- Specific and accurate descriptions
- Example code if possible (this is particularly helpful)

## Code Contributions

When submitting a pull request, contributors should describe the changes made and the rationale behind them.
If possible, provide corresponding tests and add them to `.github/workflow/`.
Pull requests require approval from at least __ONE__ team member before merging and must pass all continuous integration checks.
The best Pull Request has 1,000 review comments, discussions, and revisions; the next one has 100.

### Code Formatting

We use pre-commit for code formatting checks:

```shell
python3 -m pip install pre-commit
cd ${YOUR_CODE_DIR}/FlagTree
pre-commit install
pre-commit
```

### Unit Tests

After installation, you can run unit tests in the backend directory:
```shell
cd third_party/backendxxx/python/test/unit
python3 -m pytest -s
```

### Backend Integration

Please contact the core development team for backend integration matters.

## License

FlagTree is licensed under the [MIT license](/LICENSE).

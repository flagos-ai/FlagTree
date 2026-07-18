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

# Installing python3-flagtree-nvidia

After `apt install python3-flagtree-nvidia` (Debian/Ubuntu) or `dnf install python3-flagtree-nvidia` (Fedora),
install the ML runtime separately — it is intentionally **not** declared as a
hard Depends/Requires because the distro versions are CPU-only (torch) or
too old (triton) for GPU workloads.

This package itself *provides* `python3-triton`, so no separate triton install
is needed. Torch is not a hard dependency, but most FlagTree consumers will
want a matching GPU build:

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.0+cu128
```

A `venv` is recommended to isolate pip-installed packages from the system
Python:

```bash
python3 -m venv ~/.venv/flagos
source ~/.venv/flagos/bin/activate
# then the pip install lines above
```

Note: the distro `python3-torch` package (CPU-only build) is intentionally
not pulled in — FlagOS workloads need a GPU build, which PyTorch upstream
distributes via PyPI (per-CUDA-version wheels), not as a `.deb` / `.rpm`.

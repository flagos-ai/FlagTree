# Installing python3-flagtree-&lt;backend&gt;

One package is published per FlagTree backend — `python3-flagtree-nvidia`,
`python3-flagtree-metax`, and so on; `packaging/backends.conf` lists them.
Install the one matching the hardware, and only that one: they all occupy the
`triton` Python namespace and conflict with each other.

After `apt install python3-flagtree-<backend>` (Debian/Ubuntu) or
`dnf install python3-flagtree-<backend>` (Fedora), install the ML runtime
separately — it is intentionally **not** declared as a hard Depends/Requires
because the distro versions are CPU-only (torch) or too old (triton) for GPU
workloads. The chip vendor's own runtime has to be installed the vendor's way;
the package deliberately does not depend on vendor libraries it cannot name.

This package itself *provides* `python3-triton`, so no separate triton install
is needed. Torch is not a hard dependency, but most FlagTree consumers will
want a matching GPU build:

```bash
# nvidia; other backends install the torch build their vendor ships
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

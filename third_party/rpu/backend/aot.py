import json
import os
import subprocess
import tempfile
from pathlib import Path


class RPUAOTKernel:

    def __init__(self, artifact_path, kernel_name="kernel", launcher=None, metadata=None):
        self.artifact_path = str(Path(artifact_path).resolve())
        self.kernel_name = kernel_name
        self.launcher = launcher or os.getenv("RPU_AOT_LAUNCHER")
        self.metadata = dict(metadata or {})

    def _manifest(self, grid, args):
        return {
            "artifact_path": self.artifact_path,
            "kernel_name": self.kernel_name,
            "grid": list(grid) if isinstance(grid, tuple) else [grid],
            "args": list(args),
            "metadata": self.metadata,
        }

    def run(self, grid, *args):
        if not self.launcher:
            raise RuntimeError("RPU_AOT_LAUNCHER is not set and no launcher was provided")
        manifest = self._manifest(grid, args)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(manifest, handle, sort_keys=True)
            manifest_path = handle.name
        try:
            result = subprocess.run([self.launcher, manifest_path], check=False, text=True, capture_output=True)
            if result.returncode != 0:
                details = [
                    f"RPU AOT launcher failed: {self.launcher}",
                    f"exit code {result.returncode}",
                ]
                if result.stdout:
                    details.append(f"stdout:\n{result.stdout.rstrip()}")
                if result.stderr:
                    details.append(f"stderr:\n{result.stderr.rstrip()}")
                raise RuntimeError("\n".join(details))
            return result
        finally:
            os.unlink(manifest_path)

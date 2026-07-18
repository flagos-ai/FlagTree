from pathlib import Path


def _read_flagtree_backend() -> str:
    backend_file = Path(__file__).parent / "FLAGTREE_BACKEND"
    try:
        return backend_file.read_text().strip()
    except (FileNotFoundError, IOError):
        return ""


FLAGTREE_BACKEND: str = _read_flagtree_backend()

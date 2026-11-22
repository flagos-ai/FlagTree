"""Normal original-GGUF build; --candidate r001-v2.1 installs pinned E32 residency."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compile import main

if __name__ == "__main__":
    main()

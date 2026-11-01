"""Run the second complete comparison order with the shared controller."""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.argv.append("--reverse")
    runpy.run_path(str(Path(__file__).with_name("compare_native.py")), run_name="__main__")

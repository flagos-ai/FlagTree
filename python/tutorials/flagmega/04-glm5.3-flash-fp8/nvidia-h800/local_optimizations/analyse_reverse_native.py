"""Summarize the second order using the shared scenario-aware analysis."""
import runpy
from pathlib import Path

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('analyse_native.py')), run_name='__main__')

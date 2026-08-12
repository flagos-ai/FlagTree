"""
CPU Proton Profiler API for spine-triton backend.

This module provides a clean Python interface for CPU kernel profiling.

Usage:
    from triton.backends.spacemit.proton import profiler

    # Method 1: Context manager (recommended)
    with profiler.profile():
        kernel[grid](...)
    # Results are automatically dumped

    # Method 2: Manual control
    profiler.reset()
    kernel[grid](...)
    profiler.dump()

    # Method 3: Get results as dict
    results = profiler.get_results()

Environment Variables:
    PROTON_OUTPUT: Output file path
        - <name>.json: Chrome Trace format (view in chrome://tracing or perfetto.dev)
        - <name>.hatchet: Hatchet format for Proton tools
        - (not set): Console output

    PROTON_VERBOSE: Set to "1" for detailed per-thread output (console only)
"""

import os
from contextlib import contextmanager
from typing import Optional


class CpuProtonProfiler:
    """CPU Proton Profiler for spine-triton RISC-V backend."""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._lib = None
        self._initialized = True

    def _ensure_lib(self):
        """Lazily load the runtime library."""
        if self._lib is not None:
            return True

        try:
            from triton.backends.spacemit import env
            self._lib = env.libtritonruntime
            return True
        except Exception as e:
            print(f"Warning: Could not load spine-triton runtime: {e}")
            return False

    def reset(self) -> bool:
        """
        Reset all profiling data.

        Call this before starting a new profiling session to clear
        any previously recorded data.

        Returns:
            True if successful, False otherwise.
        """
        if not self._ensure_lib():
            return False
        try:
            self._lib.proton_reset()
            return True
        except Exception as e:
            print(f"Warning: proton_reset failed: {e}")
            return False

    def dump(self, output_path: Optional[str] = None) -> bool:
        """
        Dump profiling results.

        Args:
            output_path: Optional output file path. If provided, overrides
                        PROTON_OUTPUT environment variable.
                        - .json: Chrome Trace format
                        - .hatchet: Hatchet format
                        - None: Console output

        Returns:
            True if successful, False otherwise.
        """
        if not self._ensure_lib():
            return False

        # Temporarily set PROTON_OUTPUT if path provided
        old_env = os.environ.get("PROTON_OUTPUT")
        if output_path:
            os.environ["PROTON_OUTPUT"] = output_path

        try:
            self._lib.proton_dump()
            return True
        except Exception as e:
            print(f"Warning: proton_dump failed: {e}")
            return False
        finally:
            # Restore environment
            if output_path:
                if old_env is not None:
                    os.environ["PROTON_OUTPUT"] = old_env
                else:
                    os.environ.pop("PROTON_OUTPUT", None)

    @contextmanager
    def profile(self, output_path: Optional[str] = None, auto_dump: bool = True):
        """
        Context manager for profiling.

        Usage:
            with profiler.profile():
                kernel[grid](...)

            # Or with custom output:
            with profiler.profile("trace.json"):
                kernel[grid](...)

        Args:
            output_path: Optional output file path for results.
            auto_dump: If True, automatically dump results on exit.

        Yields:
            The profiler instance.
        """
        self.reset()
        try:
            yield self
        finally:
            if auto_dump:
                self.dump(output_path)

    def is_available(self) -> bool:
        """Check if the profiler is available."""
        return self._ensure_lib()


# Global profiler instance
profiler = CpuProtonProfiler()


# Convenience functions
def reset() -> bool:
    """Reset profiling data. Shortcut for profiler.reset()."""
    return profiler.reset()


def dump(output_path: Optional[str] = None) -> bool:
    """Dump profiling results. Shortcut for profiler.dump()."""
    return profiler.dump(output_path)


def profile(output_path: Optional[str] = None, auto_dump: bool = True):
    """Context manager for profiling. Shortcut for profiler.profile()."""
    return profiler.profile(output_path, auto_dump)


def is_available() -> bool:
    """Check if profiler is available. Shortcut for profiler.is_available()."""
    return profiler.is_available()

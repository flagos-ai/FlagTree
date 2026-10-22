"""Frontend-test hooks for the spacemit (CPU / RISC-V) backend.

Mirrors the hooks the amd, iluvatar, mthreads and ppu spec packages install for
``triton._filecheck``; see also ``triton/spec/*/triton/_filecheck.py``.
"""

from triton.backends.compiler import GPUTarget


def spec_get_stub_target() -> GPUTarget:
    """Target the AST -> TTIR tests should compile for.

    The CPU backend cannot be reached through ``triton.runtime.driver.active``:
    ``CPUDriver.is_active()`` returns False unconditionally, so the driver is
    only ever available after an explicit
    ``triton.runtime.driver.set_active(CPUDriver())`` - which its own tests do
    but a frontend test cannot know about.  Ask the driver directly instead, the
    same way its RPC mode does.  The import stays inside the function because
    this module is loaded from ``triton/__init__.py`` via
    ``spec_call("triton_extend_globals", ...)``, before the backend is importable.
    """
    try:
        from triton.backends.spacemit.driver import CPUDriver
        return CPUDriver().get_current_target()
    except Exception:
        # No usable driver here: fall back to the generic stub target.
        return None

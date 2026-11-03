# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Keep the tutorial's process-wide compiler hooks within one artifact's lifetime."""

import atexit
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
import sys
from threading import RLock

_lock = RLock()


def _hook_targets():
    from jinja2 import Environment
    from triton.flagmega.codegen.triton import function_programs, program_dependencies, renderer, tir_package
    from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
    from triton.flagmega.passes.tir.bufferize import program_lifetimes

    return (
        (Environment, "get_template"),
        (TritonTemplateRegistry, "prepare"),
        (TritonTemplateRegistry, "render"),
        (tir_package, "describe_tir_package"),
        (function_programs, "project_function_programs"),
        (function_programs, "_actual_bindings"),
        (program_dependencies, "verify_program_dependencies"),
        (program_lifetimes, "program_use_ends"),
    ), renderer.renderer_registry


def _source_cells(functions):
    """O-proj extends an inner renderer's captured source in place."""
    pending = list(functions)
    seen = set()
    cells = {}
    while pending:
        function = pending.pop()
        if id(function) in seen:
            continue
        seen.add(id(function))
        for cell in getattr(function, "__closure__", ()) or ():
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if isinstance(value, str):
                cells[id(cell)] = (cell, value)
            elif callable(value):
                pending.append(value)
    return tuple(cells.values())


@contextmanager
def activated_artifact_recipe(trial):
    """Activate the frozen recipe, then restore hooks even after an exception.

    Tutorial recipes include absolute ``local_optimizations`` imports and
    import-time hooks. A fresh package namespace is therefore required for
    every artifact, along with restoration of all compiler hooks they own.
    The scope must cover lazy preparation and every use of that runtime.
    Process-global rendering is serialized by the lock.
    """
    trial = Path(trial).resolve()
    if not (trial / "local_optimizations" / "kernel_recipe.py").is_file():
        raise ValueError(f"Missing frozen artifact recipe: {trial}")
    with _lock:
        targets, registry = _hook_targets()
        hooks = [(owner, name, getattr(owner, name)) for owner, name in targets]
        specs = dict(registry._specs)
        sources = _source_cells([value for _, _, value in hooks])
        package = "local_optimizations"

        def owned(name):
            return name == package or name.startswith(package + ".")

        modules = {name: module for name, module in sys.modules.items() if owned(name)}
        paths = list(sys.path)
        recipe = None
        try:
            for name in modules:
                del sys.modules[name]
            sys.path.insert(0, str(trial))
            recipe = import_module(package + ".kernel_recipe")
            recipe.activate()
            yield recipe
        finally:
            try:
                if recipe is not None:
                    owner = getattr(recipe, "_owner_context", None)
                    if owner is not None:
                        owner.__exit__(None, None, None)
                active_package = sys.modules.get(package)
                contexts = getattr(active_package, "_trial_contexts", None)
                if contexts is not None:
                    atexit.unregister(contexts.close)
                    contexts.close()
            finally:
                for cell, source in sources:
                    cell.cell_contents = source
                for owner, name, value in hooks:
                    setattr(owner, name, value)
                registry._specs.clear()
                registry._specs.update(specs)
                for name in tuple(sys.modules):
                    if owned(name):
                        del sys.modules[name]
                sys.modules.update(modules)
                sys.path[:] = paths

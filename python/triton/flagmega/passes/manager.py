# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Deterministic pass manager with nncase-style pass dump ownership."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, replace
from typing import Callable, Protocol

from triton.flagmega.diagnostics.dump import DumpFlags, DumpScope, Dumper, FunctionDump
from triton.flagmega.ir import IRModule, verify_module
from triton.flagmega.passes.analysis import AnalysisManager, AnalysisProvider
from triton.flagmega.passes.context import RunPassContext, pass_context
from triton.flagmega.passes.function_pass import FunctionPass, FunctionTraversalOrder
from triton.flagmega.passes.prim_function_pass import PrimFunctionPass


class ModulePass(Protocol):
    name: str
    preserves: frozenset[str]

    def run(self, module: IRModule) -> IRModule: ...


@dataclass(frozen=True)
class FunctionalPass:
    name: str
    transform: Callable[[IRModule], IRModule]
    preserves: frozenset[str] = frozenset()

    def run(self, module: IRModule) -> IRModule:
        return self.transform(module)


@dataclass(frozen=True)
class PassResult:
    module: IRModule
    executed: tuple[str, ...]
    invalidated_analyses: tuple[str, ...]
    executions: tuple[PassExecution, ...] = ()
    before_dump: str | None = None
    after_dump: str | None = None
    before_functions: tuple[FunctionDump, ...] = ()
    after_functions: tuple[FunctionDump, ...] = ()


@dataclass(frozen=True)
class PassExecution:
    index: int
    name: str
    input_semantic_hash: str
    output_semantic_hash: str
    elapsed_ms: float
    before_dump: str | None
    after_dump: str | None
    before_functions: tuple[FunctionDump, ...] = ()
    after_functions: tuple[FunctionDump, ...] = ()

    def to_data(self) -> dict[str, object]:
        return {
            "index": self.index,
            "name": self.name,
            "input_semantic_hash": self.input_semantic_hash,
            "output_semantic_hash": self.output_semantic_hash,
            "elapsed_ms": self.elapsed_ms,
            "before_dump": self.before_dump,
            "after_dump": self.after_dump,
            "before_functions": [item.to_data() for item in self.before_functions],
            "after_functions": [item.to_data() for item in self.after_functions],
        }


class PassManager:
    def __init__(self, name: str = "PassManager", *, dumper: Dumper | None = None) -> None:
        self.name = name
        self._dumper = dumper
        self._passes: list[ModulePass] = []
        self._analyses = AnalysisManager()
        self._open_egraph_session = None
        self._frozen = False

    def add(self, module_pass: ModulePass) -> PassManager:
        if self._frozen:
            raise RuntimeError(f"PassManager {self.name!r} is frozen after its first run.")
        # Match nncase PassManager: consecutive EGraph passes share one graph,
        # with construct/extract lifecycle passes inserted automatically.
        from triton.flagmega.egraph import EGraphSession
        from triton.flagmega.passes.rewriter import EGraphConstructPass, EGraphRulesPass

        if isinstance(module_pass, EGraphRulesPass):
            if self._open_egraph_session is None:
                self._open_egraph_session = EGraphSession(
                    node_limit=module_pass.node_limit,
                    class_limit=module_pass.class_limit,
                )
                self._passes.append(EGraphConstructPass(self._open_egraph_session))
            elif (
                self._open_egraph_session.node_limit != module_pass.node_limit
                or self._open_egraph_session.class_limit != module_pass.class_limit
            ):
                raise ValueError("Contiguous EGraphRulesPass instances must use identical graph limits.")
            self._passes.append(module_pass.bind(self._open_egraph_session))
            return self
        self._close_egraph_group()
        self._passes.append(module_pass)
        return self

    def seed_analysis(self, name: str) -> None:
        self._analyses.seed(name)

    def register_analysis(self, provider: AnalysisProvider) -> PassManager:
        if self._frozen:
            raise RuntimeError(f"PassManager {self.name!r} is frozen after its first run.")
        self._analyses.register(provider)
        return self

    def run(self, module: IRModule) -> PassResult:
        if not self._frozen:
            self._close_egraph_group()
        self._frozen = True
        current = verify_module(module)
        executed: list[str] = []
        invalidated: set[str] = set()
        executions: list[PassExecution] = []
        root = self._dumper or DumpScope.current()
        with DumpScope(root):
            manager_before = root.dump_module(module, "Before", category=DumpFlags.PASS_IR)
            for index, module_pass in enumerate(self._passes):
                pass_dumper = DumpScope.current().create_sub_dumper(
                    f"{index:02d}_{_path_name(module_pass.name)}")
                with DumpScope(pass_dumper):
                    before = pass_dumper.dump_module(current, "Before", category=DumpFlags.PASS_IR)
                    input_hash = current.semantic_hash
                    started = time.perf_counter()
                    context = RunPassContext(
                        manager_name=self.name,
                        pass_name=module_pass.name,
                        pass_index=index,
                        module=current,
                        analyses=self._analyses,
                        dumper=pass_dumper,
                    )
                    if isinstance(module_pass, PrimFunctionPass):
                        current = self._run_prim_function_pass(module_pass, current, context)
                    elif isinstance(module_pass, FunctionPass):
                        current = self._run_function_pass(module_pass, current, context)
                    else:
                        with pass_context(context):
                            current = verify_module(module_pass.run(current))
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    after = pass_dumper.dump_module(current, "After", category=DumpFlags.PASS_IR)
                executed.append(module_pass.name)
                removed = self._analyses.finish_pass(
                    input_hash=input_hash,
                    output_hash=current.semantic_hash,
                    preserves=module_pass.preserves,
                )
                invalidated.update(removed)
                executions.append(PassExecution(
                    index=index,
                    name=module_pass.name,
                    input_semantic_hash=input_hash,
                    output_semantic_hash=current.semantic_hash,
                    elapsed_ms=elapsed_ms,
                    before_dump=None if before is None else str(before.directory),
                    after_dump=None if after is None else str(after.directory),
                    before_functions=() if before is None else before.functions,
                    after_functions=() if after is None else after.functions,
                ))
            manager_after = root.dump_module(current, "After", category=DumpFlags.PASS_IR)
        return PassResult(
            current,
            tuple(executed),
            tuple(sorted(invalidated)),
            tuple(executions),
            None if manager_before is None else str(manager_before.directory),
            None if manager_after is None else str(manager_after.directory),
            () if manager_before is None else manager_before.functions,
            () if manager_after is None else manager_after.functions,
        )

    def _run_function_pass(
        self,
        function_pass: FunctionPass,
        module: IRModule,
        context: RunPassContext,
    ) -> IRModule:
        from triton.flagmega.passes.functions import callee_first_functions

        functions = (
            callee_first_functions(module)
            if function_pass.traversal_order is FunctionTraversalOrder.CALLEE_FIRST
            else module.functions
        )
        current = module
        for function in functions:
            if function.name not in current.function_map:
                raise RuntimeError(
                    f"Function pass {function_pass.name!r} removed pending function "
                    f"@{function.name}; use a ModulePass for function-set changes."
                )
            scoped = replace(
                context,
                module=current,
                function=function.name,
            )
            with pass_context(scoped):
                current = verify_module(function_pass.run_function(
                    current.function_map[function.name],
                    current,
                    scoped,
                ))
        return current

    def _run_prim_function_pass(
        self,
        function_pass: PrimFunctionPass,
        module: IRModule,
        context: RunPassContext,
    ) -> IRModule:
        current = module
        names = tuple(function.name for function in module.prim_functions)
        for name in names:
            if name not in current.prim_function_map:
                raise RuntimeError(
                    f"PrimFunction pass {function_pass.name!r} removed pending @{name}; "
                    "use a ModulePass for function-set changes."
                )
            scoped = replace(context, module=current, function=name)
            with pass_context(scoped):
                rewritten = function_pass.run_prim_function(
                    current.prim_function_map[name], current, scoped
                )
                functions = tuple(
                    rewritten if value.name == name else value
                    for value in current.prim_functions
                )
                current = verify_module(replace(current, prim_functions=functions))
        return current

    def _close_egraph_group(self) -> None:
        if self._open_egraph_session is None:
            return
        from triton.flagmega.passes.rewriter import EGraphExtractPass

        self._passes.append(EGraphExtractPass(self._open_egraph_session))
        self._open_egraph_session = None


def _path_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return value or "pass"

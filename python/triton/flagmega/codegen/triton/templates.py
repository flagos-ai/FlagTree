# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Deterministic Jinja template lookup for selected semantic TIR kernels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateError

from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import IRModule


_SNAKE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class KernelTemplateSpec:
    """Template identity derived from a selected TIR candidate."""

    kernel: str
    variant: str
    platform: str | None = None
    architecture: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("kernel", "variant", "platform", "architecture"):
            value = getattr(self, field_name)
            if value is not None and not _SNAKE_NAME.fullmatch(value):
                raise CodegenError(
                    f"Triton template {field_name} must be snake_case, got {value!r}."
                )
        if self.architecture is not None and self.platform is None:
            raise CodegenError("A Triton architecture specialization requires a platform.")


@dataclass(frozen=True)
class RenderedKernelTemplate:
    spec: KernelTemplateSpec
    template: str
    source: str


class TritonTemplateRegistry:
    """Resolve per-kernel implementations with deterministic specialization fallback."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else Path(__file__).parent
        self.environment = Environment(
            loader=FileSystemLoader(str(self.root)),
            undefined=StrictUndefined,
            extensions=("jinja2.ext.do",),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            autoescape=False,
        )

    def candidates(self, spec: KernelTemplateSpec) -> tuple[str, ...]:
        # Constructing another value applies validation even when callers use
        # dataclasses.replace or deserialize an object without __init__.
        spec = KernelTemplateSpec(
            spec.kernel,
            spec.variant,
            spec.platform,
            spec.architecture,
        )
        base = f"kernels/{spec.kernel}"
        values: list[str] = []
        if spec.platform is not None and spec.architecture is not None:
            values.append(
                f"{base}/platforms/{spec.platform}/{spec.architecture}/{spec.variant}.py.jinja"
            )
        if spec.platform is not None:
            values.append(f"{base}/platforms/{spec.platform}/{spec.variant}.py.jinja")
        values.append(f"{base}/{spec.variant}.py.jinja")
        return tuple(values)

    def resolve(self, spec: KernelTemplateSpec) -> str:
        candidates = self.candidates(spec)
        for relative in candidates:
            if (self.root / relative).is_file():
                return relative
        raise CodegenError(
            f"No Triton template implements {spec.kernel}/{spec.variant}; "
            f"searched {list(candidates)}."
        )

    def render_kernel(
        self,
        spec: KernelTemplateSpec,
        context: Mapping[str, object],
    ) -> RenderedKernelTemplate:
        template_name = self.resolve(spec)
        return RenderedKernelTemplate(
            spec,
            template_name,
            self.render(template_name, context),
        )

    def render(self, template_name: str, context: Mapping[str, object]) -> str:
        try:
            return self.environment.get_template(template_name).render(**context)
        except TemplateError as error:
            raise CodegenError(
                f"Failed to render Triton template {template_name}: {error}"
            ) from error


def module_template_target(module: IRModule) -> tuple[str, str]:
    """Read the concrete target-owned template namespace from selected TIR."""

    value = module.metadata.get("codegen_template_target")
    if not isinstance(value, Mapping):
        raise CodegenError(
            "Selected TIR has no target-owned codegen_template_target metadata.",
            stage=module.stage,
        )
    platform = str(value.get("platform", ""))
    architecture = str(value.get("architecture", ""))
    # Reuse the public identity validator instead of accepting renderer-local
    # spelling or target-name parsing conventions.
    KernelTemplateSpec("identity", "identity", platform, architecture)
    return platform, architecture


__all__ = [
    "KernelTemplateSpec",
    "RenderedKernelTemplate",
    "TritonTemplateRegistry",
    "module_template_target",
]

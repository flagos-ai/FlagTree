---
name: tle-developer
description: Self-contained orchestration skill for writing high-performance TLE kernels and shipping TLE feature changes with reproducible validation.
---

# TLE Developer

## Mission
Use this skill to execute TLE work end-to-end:
intake -> implementation -> validation -> artifacts -> merge decision.

## Scope
Use for:
1. Writing or optimizing TLE kernels.
2. Implementing TLE API/verifier/lowering/pipeline features.
3. Debugging correctness, performance, and regression issues.

## Self-Contained Policy
1. Do not rely on documentation outside this skill folder.
2. Put all detailed guidance in `references/`.
3. Keep this file as orchestration-only (no duplicated deep details).

## Required Input
Start every task with:
```text
Goal:
Non-goal:
Acceptance:
Impact scope (optional):
```

## Mandatory Read Order
1. `references/tle-sources.md`
2. `references/workflow-templates.md`

## Operating Contract
1. Treat `references/tle-sources.md` as the technical source of truth for:
   - quickstart,
   - current TLE semantics contract,
   - kernel patterns,
   - feature-development file map,
   - debug/perf procedures.
2. Treat `references/workflow-templates.md` as the source of truth for:
   - intake,
   - validation matrix,
   - performance record,
   - fix summary,
   - lessons entry,
   - merge package.

## Non-Negotiable Guardrails
1. Never assume a specific python environment name.
2. Never assume a fixed build script name.
3. If native Triton files are modified for TLE-specific behavior, use compile-time guards like `#ifdef __TLE__` / `#endif`.
4. Do not use comment marker blocks (`// begin flagtree tle` / `// end flagtree tle`) as a policy mechanism.
5. Do not add `__TLE__` guards inside `third_party/tle` unless that subtree explicitly requires it.

## Required Outputs Per Task
1. Validation commands and outcomes.
2. Fix Summary (when fixing bugs or regressions).
3. Lessons Entry (for fixes and optimization work).
4. Merge Decision Package (changed layers, risks, follow-ups).

## Completion Checklist
1. Acceptance criteria mapped to tests.
2. Changes validated with reproducible commands.
3. Artifacts filled from templates.
4. Residual risks explicitly stated.

## References
1. `references/tle-sources.md`
2. `references/workflow-templates.md`

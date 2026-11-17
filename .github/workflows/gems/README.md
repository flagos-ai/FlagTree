# FlagGems CI skip lists

Per-backend skip lists for the FlagGems CI, one directory per FlagGems
release:

```
gems/<flaggems-tag>/skip_list/
├── default.yaml      used unless the backend has its own
└── <backend>.yaml    overrides default.yaml when present
```
FlagGems-vllm uses the same layout in a sibling directory,
`gems-vllm/<flaggems-vllm-tag>/skip_list/`, selected by `repo-kind:
flaggems-vllm`. The two never share a directory, so their tag namespaces
cannot collide.

## Why only the skip lists live here

Everything else the FlagGems CI needs moved into the actions that run it:

| file | shipped with |
|---|---|
| `ci/ci_run_tests.py` | `.github/actions/run-flaggems-test/assets/` |
| `gems/<tag>/conftest.py`, `gems-vllm/<tag>/conftest.py` | `.github/actions/run-flaggems-test/assets/` |
| `compare_test_result.py` | `.github/actions/compare-flaggems-test/scripts/` |
| `collect_gems_record.py`, `flaggems_runs.schema.json` | `.github/actions/collect-gems-record/scripts/` |

Those are read through `$GITHUB_ACTION_PATH`, which resolves under the ref in
`uses: ...@main` rather than the branch being tested, so one copy on `main`
serves every branch. Files read from `$GITHUB_WORKSPACE` would need a copy per
branch instead, and keeping them in step across branches was the problem this
layout solves.

The skip lists stay in the checkout because they are not duplication: each
branch carries the lists for the backends it builds, and the contents differ
per backend. A backend only needs a file when it differs from `default.yaml`.

## Adding a FlagGems version

The directory name is the `flaggems-tag` input of
`template-gems-{test,baseline}.yml` verbatim, so no mapping is maintained. A
new release needs two things: `assets/gems/<tag>/conftest.py` in
`run-flaggems-test`, and `<tag>/skip_list/default.yaml` here. Old and new can
coexist, so backends move over one at a time.

A FlagGems tag without a skip list fails. A FlagGems tag without a
`conftest.py` asset falls back to the clone's own `conftest.py`, which works
only when that release already implements `--first-parameter-only`.

A FlagGems-vllm release needs `assets/gems-vllm/<tag>/conftest.py` and
`gems-vllm/<tag>/skip_list/default.yaml`. Both are mandatory: FlagGems-vllm's
own `conftest.py` does not implement `--first-parameter-only`, so there is no
fallback.

## Why the skip lists matter beyond skipping

Skipped operators change the denominator of the pass rate, so keeping these in
git is what makes a change in pass rate attributable: a compiler change, or an
operator added to a skip list. `record.json` carries the list that each run
used for the same reason.

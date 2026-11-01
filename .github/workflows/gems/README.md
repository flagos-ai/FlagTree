# FlagGems CI assets

Everything the FlagGems CI runs needs beyond the FlagGems repository itself.
Previously these lived only on the runners under `/root/flaggems_ci/script/`,
outside version control.

```
gems/
├── ci/                     run in place from here, never copied
│   ├── ci_run_tests.py           takes --flaggems-path
│   └── compare_test_result.py
└── <flaggems-tag>/         one directory per FlagGems release
    ├── conftest.py         copied to <clone>/tests/conftest.py
    └── skip_list/
        ├── default.yaml    used unless the backend has its own
        └── <backend>.yaml  overrides default.yaml when present
```

The layout follows how the files actually vary:

| file | varies by backend | varies by FlagGems version |
|---|---|---|
| `ci/*` | no | no |
| `conftest.py` | no | yes |
| `skip_list` | yes | yes |

## What gets copied, and what does not

Only `conftest.py`, because pytest will only pick up a conftest from the tests
directory it is collecting.

The scripts under `ci/` take every path as an argument, so they run straight
out of this directory. `ci_run_tests.py` is told where the clone is via
`--flaggems-path`; without that flag it falls back to assuming it lives in
`<FlagGems>/tools/ci/`, which is how it used to be deployed.

The selected skip list is written to the run's own output directory rather than
into the clone, so it survives cleanup and records which list the run actually
used.

## Adding a FlagGems version

The directory name is the `flaggems-tag` input of
`template-gems-{test,baseline}.yml` verbatim, so no mapping has to be
maintained. Copy the previous version's directory, adjust the contents, then
bump `flaggems-tag` in the per-backend callers. Old and new can coexist, which
makes it possible to move backends over one at a time.

A run whose `flaggems-tag` has no directory here **fails** rather than falling
back. Running with FlagGems' own `conftest.py` would collect no operator
markers, leaving a `summary.json` that looks plausible but is incomplete —
worse than an obvious failure.

## Per-backend skip lists

`default.yaml` applies unless `<backend>.yaml` exists, so a backend only needs
a file when it genuinely differs. Skipped operators change the denominator of
the pass rate, so keeping these in git is what makes the numbers in the CI
dashboard explainable: a jump in pass rate can be traced to a compiler change
or to an operator being added to a skip list.

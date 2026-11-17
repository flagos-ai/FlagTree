# Explicit SME copy stride

This frontend extension builds on the explicit TLE copy/group API in #1538
and the Iluvatar SME builder/transport implementation in #1559. It is kept
separate so the NVIDIA PR does not introduce a backend-specific stride knob.
The backend implementation and geometry validation remain owned by #1559.

`tle.gpu.copy(..., input_stride=...)` supplies the SME source row stride in
elements. The default `None` preserves ordinary copies. Other backends reject
an explicit stride instead of ignoring it. Synchronous copies forward the
stride to Iluvatar's SME load slot; explicit asynchronous copies forward its
IR value to the strided builder supplied by #1559.

The stride is validated before copy-path dispatch. Descriptor copies,
shared-to-global copies, non-shared destinations, and CommonIR cannot consume
it and are rejected before IR emission. A constexpr-wrapped `None` is treated
as the ordinary default.

The shared NVIDIA legalization pass is not built in Iluvatar's configuration,
which sets FLAGTREE_TLE=OFF and FLAGTREE_ILUVATAR_TLE=ON. Its unused Iluvatar
byte-width branch is therefore removed from #1538 rather than copied here.
Iluvatar continues to use its own SME lowering and wait-count conversion.

## Regression checks

```sh
pytest -q third_party/iluvatar/python/test/unit/tle/test_tle_sme_copy_frontend.py
```

On the parent NVIDIA-only API, six stride cases fail and the ordinary-copy
control passes. With this extension all seven host cases pass. They cover
constant, constexpr and runtime strides, default dispatch, unsupported
backends, and rejection when the strided builder capability is absent.

`test_sme_stride_reaches_native_tle_builder` checks native TTIR on the Iluvatar
compiler after the #1559 backend implementation is present. It is skipped on
NVIDIA. This PR remains draft until its backend dependency is independently
validated; host tests do not establish device correctness or performance.

## Dispatch regression (2026-11-17)

`gpu/core.py::copy` originally checked stride ownership only inside `normcopy`.
Descriptor and CommonIR dispatch returned earlier, and the shared-to-global
branch did not consume the normalized stride. Those paths silently discarded
the request. Checking for `None` before constexpr unwrapping also selected the
strided builder for `tl.constexpr(None)`.

Eight added regression cases fail on `7d5594d9`: four descriptor/backend-mode
combinations, two CommonIR async-mode combinations, shared-to-global dispatch,
and constexpr `None`. Validation now occurs before selecting any transport,
after unwrapping the optional stride. All 18 host cases pass, including three
synchronous SME load-forwarding controls and the existing async cases. The
Iluvatar native TTIR case remains skipped on the available NVIDIA compiler.

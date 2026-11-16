# Global data publication to the async proxy

`tle.gpu.fence_proxy_async_global()` emits a global-memory fence between the
generic and async proxies on NVIDIA SM90 or newer with PTX ISA 8.0 or newer.
Other targets are rejected. The operation has observable read/write effects,
survives canonicalization and CSE, and lowers to `fence.proxy.async.global`.

The fence acts in the issuing thread. Cross-thread synchronization remains a
separate requirement. For a reader-side fence, acquire the publication first:

```python
ready_handoff.wait(epoch)
tle.gpu.fence_proxy_async_global()
# Subsequent TMA reads may now observe the acquired generic global writes.
```

Every actual TMA issuer must execute the fence, or be ordered after a thread
that does by an appropriate synchronization protocol. Executing it in all
lanes of the producer warp before electing the TMA issuer is sufficient.
An existing grid publication followed by a CTA handoff can supply the
cross-thread happens-before chain; the fence does not replace either step.

`tensor_map_fenceproxy_acquire` acquires descriptor bytes, so it is separately
required for a host-written device-resident tensor map. It does not publish
the tensor payload. Similarly, TMA completion provides the opposite
async-to-generic ordering and does not publish preceding generic writes to
a subsequent async read.

The compiler does not automatically infer mutable data publication from
opaque descriptor pointers. A caller using this API must establish the
publication chain and choose the correct location. Writer-side placement
with a subsequent publication is also possible; this API does not impose
reader-side placement on all protocols.

The contract follows the PTX ISA sections on [async proxy operations](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#async-proxy)
and [memory and proxy fences](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar).
CPU and device regressions are in `python/test/tle/unit/test_tle_global_proxy_fence.py`;
capability and non-elision checks are in
`third_party/tle/test/GPU/test_tle_global_proxy_fence.mlir`.

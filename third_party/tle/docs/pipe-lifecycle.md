# Reusable pipe identity and lifecycle

A pipe is a runtime instance, even when it is fieldless or shares backing
storage with another pipe. The NVIDIA pipe IR carries an SSA identity from
creation through writer/reader operations. Diagnostic names do not change the
structural type or JIT function specialization. The logical ring capacity stays
exact; only the internal NVWS token tensor is padded to a power of two.

A cyclic writer acquires and commits data epochs, then closes the next epoch.
Close publishes an elected-thread arrival using the established data count.
It must not replace that count with the writer partition width. Empty streams
derive the count after all lifecycle operations have been analyzed.

After close and the final reader release, every participant can call
wait_drained once. It rendezvous before whole-storage reuse. The lowering checks
writer and declared reader participation and rejects subsequent pipe use.

## Root causes and regressions

- In TleLowerPipeToNvws.cpp, getPipeKey previously relied on diagnostic names and
  field pointers. Distinct fieldless instances could collide, and renaming
  specialized otherwise equivalent endpoint functions. SSA identity separates
  instances, while type equality and mangling omit the diagnostic name.
- Writer-close handling previously overwrote the proven data arrival count.
  Sub-warp payloads need their actual data count; close now preserves that count.
  test_tle_pipe_close_participant_count.mlir covers the 16-thread case.
- analyzePipeDrains originally counted lexical drain sites without checking their
  control placement. A conditional drain can omit arrivals, while a loop can
  arrive repeatedly on the one-shot epoch. recordDrainArrival now requires the
  boundary outside conditional and loop regions within its async task.
  test_tle_pipe_drain_conditional.mlir fails before this correction and passes
  afterward for both branch and zero-trip-loop inputs.
- The shared frontend identity argument was incompatible with the existing
  MUSA native field-only builder. Its backend-owned pipe value/endpoint adapter
  retains that ABI, including flatten/unflatten behavior and name-based types.
  The test_musa_native_pipe_uses_existing_field_only_signature host regression
  exercises acquire/commit/close/wait/release and endpoint transport.

Runtime regression commands:

~~~bash
pytest python/test/tle/integration/test_tle_pipe_non_power_two_capacity.py -q
pytest python/test/tle/integration/test_tle_pipe_drain.py -q
pytest python/test/tle/unit/test_tle.py -k musa_native_pipe -q
~~~

The CUDA regressions cover capacities 1/2/3/5, zero payloads, close tags,
fieldless edges, TMA handoffs, and helper calls. MUSA retains its existing
field-only contract; this PR does not add native MUSA identity or drain support.
All new runtime tests declare required TLE primitives via the shared capability
marker.

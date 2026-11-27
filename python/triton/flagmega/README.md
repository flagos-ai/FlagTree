# FlagMega

FlagMega compiles a model into executable Triton kernels and a runtime artifact.
It imports model weights and configuration from a checkpoint, rewrites the
model graph, selects distribution and kernel implementations for the target,
and plans the runtime buffers. Triton Language Extensions provide the
synchronization and memory operations used by generated kernels.

The compiler saves editable Python IR at stage boundaries. Use
`flagmega inspect` to examine a checkpoint, `flagmega stage` to run one pass
group, and `flagmega resume` to continue from saved IR. An executable artifact
contains generated kernel source, a manifest, and the packed readonly weights.
Use `flagmega artifact verify` to check the artifact before loading it on the
target device.

Install the dependencies with the `flagmega` package extra. Install PyTorch
with GPU support for the target to execute an artifact.

See the [tutorial index](../../tutorials/flagmega/README.md) for installation
commands and model examples.

The [FlagMega developer skill](../../../skills/flagmega-developer/SKILL.md)
covers profiling, optimization, and validation workflows.

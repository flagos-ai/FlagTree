# Triton-TileIR Backend User Guide

## Build Instructions

To build and install the Triton-TileIR backend, simply run:

```bash
pip install .
```

## Running

Before using the backend, ensure you have CTK 13.1 installed and set the following environment variable:

```bash
export ENABLE_TILE=1
```

## Known Limitations

- Some tests that are not supported by CudaTile are not yet automatically skipped; as a result, you may see failures in certain unit tests.

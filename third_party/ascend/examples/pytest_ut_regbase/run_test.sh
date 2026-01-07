#!/bin/bash

TRITON_ASCEND_ARCH=Ascend310B4 \
TRITON_ENABLE_TASKQUEUE=0 \
pytest -n 7 --dist=load \
.

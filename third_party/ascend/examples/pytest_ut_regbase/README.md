# Test Guide

When you already set up bisheng-triton in this branch `regbase`, you can run the script `run_test.sh` to run the unit tests in this directory.

Notes:

- It seems that 310B4 can run max 7 threads at the same time. So the `-n` option of pytest should be no more than 7. Otherwise, the output of your kernel maybe empty.

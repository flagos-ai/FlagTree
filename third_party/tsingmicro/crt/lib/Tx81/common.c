//===----------------------- common.c -------------------------------------===//
//
//
//===----------------------------------------------------------------------===//
//
// Implement common helper functions in this file.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

// WORKAROUND for undefined symbols in libkcorert.a
int main(int argc, char **argv) { return 0; }

int get_app_version() { return 1; }

int nvram_get_val() { return 1; }

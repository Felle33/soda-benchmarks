// RUN: not env LCS_COMPUTE_LANES=32 bash %S/../../../../../scripts/tosa_to_lcs.sh %s %t --mode kernel 2>&1 | FileCheck %s --check-prefix=ENV
// RUN: not bash %S/../../../../../scripts/tosa_to_lcs.sh %s %t --mode kernel --hardware-family conv 2>&1 | FileCheck %s --check-prefix=MODE
// ENV: LCS_COMPUTE_LANES is no longer a compiler option
// MODE: A hardware family is only valid for network engine extraction
module {}

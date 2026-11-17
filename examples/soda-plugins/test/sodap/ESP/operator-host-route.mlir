// RUN: bash %S/../../../../../scripts/lcs/check_host_mode.sh kernel
// RUN: not bash %S/../../../../../scripts/lcs/check_host_mode.sh operators 2>&1 | FileCheck %s
// CHECK: Operator mode must use sodap-tosa-network-host; refusing sodap-launches-to-esp
module {}

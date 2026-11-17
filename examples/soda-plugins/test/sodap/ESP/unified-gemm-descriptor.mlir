// RUN: mkdir -p %t.dir
// RUN: c++ -std=c++17 -O2 -ffp-contract=off -DSODAP_GEMM_TRACE -DSODAP_OP_EXTERNAL_EXECUTOR -I %S/../../../include %S/Inputs/test_unified_descriptor.cpp %S/../../../lib/sodap/ExecutionEngine/ReusableGemmRuntime.cpp %S/../../../lib/sodap/ExecutionEngine/ReusableOperatorRuntime.cpp -o %t
// RUN: %t %t.dir | FileCheck %s
// CHECK: Unified descriptor byte identity PASS

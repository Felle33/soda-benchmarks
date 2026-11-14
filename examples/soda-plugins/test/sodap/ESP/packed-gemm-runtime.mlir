// RUN: c++ -std=c++17 -O2 -ffp-contract=off -I %S/../../../include %S/Inputs/test_packed_gemm.cpp %S/../../../lib/sodap/ExecutionEngine/ReusableGemmRuntime.cpp -o %t
// RUN: %t | FileCheck %s
// CHECK: Packed reusable GEMM runtime PASS

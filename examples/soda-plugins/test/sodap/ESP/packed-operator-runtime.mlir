// RUN: c++ -std=c++17 -O2 -ffp-contract=off -I %S/../../../include %S/Inputs/test_packed_operator.cpp %S/../../../lib/sodap/ExecutionEngine/ReusableOperatorRuntime.cpp -o %t
// RUN: %t | FileCheck %s
// CHECK: Packed convolution/add/GEMM/layout runtime PASS

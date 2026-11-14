// RUN: c++ -std=c++17 -O2 -Wall -Wextra -I %S/../../../include %S/Inputs/test_packed_layout.cpp -o %t
// RUN: %t | FileCheck %s
// CHECK: packed layout PASS

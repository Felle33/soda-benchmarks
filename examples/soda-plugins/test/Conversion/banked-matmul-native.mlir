// RUN: python3 %S/Inputs/banked_matmul_native.py %sodap_libs/SODAPlugin%shlibext %t
// REQUIRES: panda
// Executes the transformed LLVM against scalar C for full blocks and tails,
// nonzero initial accumulators, repeated calls, and physical padding guards.
module {}

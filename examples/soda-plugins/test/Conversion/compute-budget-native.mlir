// RUN: python3 %S/Inputs/compute_budget_native.py %sodap_libs/SODAPlugin%shlibext %t
// REQUIRES: panda
// Compare independent ordinary and banked LLVM execution with partial tiles,
// shared intermediates, vector normalization, batched contraction and canaries.
module {}

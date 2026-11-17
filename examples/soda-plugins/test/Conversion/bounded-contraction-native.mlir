// RUN: python3 %S/Inputs/bounded_contraction_native.py %sodap_libs/SODAPlugin%shlibext %t.dir
// Poisoned padding must not enter the reduction, and neither initialization
// nor the broadcast-bias epilogue may overwrite outside the actual M/N extent.

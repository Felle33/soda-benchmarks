// RUN: mkdir -p %t
// RUN: mlir-opt %s --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-emit-hls-configuration{lanes=16 directory=%t})" | FileCheck %s --check-prefix=IR
// RUN: cat %t/bambu_fp_constraints.txt | FileCheck %s --check-prefix=CONSTRAINT
// RUN: cat %t/compute_config.json | FileCheck %s --check-prefix=CONFIG
// RUN: not mlir-opt %s --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-emit-hls-configuration{lanes=257 directory=%t})" 2>&1 | FileCheck %s --check-prefix=INVALID
// CONSTRAINT: -C=__float_mule8m23b_127nih=16,__float_adde8m23b_127nih=16
// CONFIG: "fp_units_per_task":16
// INVALID: HLS core unit budget must be 1..256
// IR: module {
// IR-NEXT: }
module {}

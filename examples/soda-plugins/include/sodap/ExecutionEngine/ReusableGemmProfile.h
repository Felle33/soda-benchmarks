// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef SODAP_REUSABLE_GEMM_PROFILE_H
#define SODAP_REUSABLE_GEMM_PROFILE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Optional observer, enabled by SODAP_GEMM_PROFILE. Events bracket a complete
// invocation, including descriptor programming, fences and completion polling.
// Observers must not print or allocate in the measured interval.
void esp_gemm_profile_event(uint32_t begin, uint32_t invocation);
#ifdef __cplusplus
}
#endif
#endif

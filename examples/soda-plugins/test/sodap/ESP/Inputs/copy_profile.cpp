#include <cstdint>
// Native lifetime tests exercise region counters without a board timer hook.
extern "C" void esp_op_profile_event(uint32_t, uint32_t) {}

#ifndef FUZZY_MATCH_FFI_BRIDGE_H
#define FUZZY_MATCH_FFI_BRIDGE_H

#include <stdint.h>

#if defined(_WIN32)
#define FUZZY_API __declspec(dllexport)
#else
#define FUZZY_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

FUZZY_API double fuzzy_score(const char *a, const char *b);
FUZZY_API uint8_t fuzzy_match(const char *a, const char *b, double threshold);

#ifdef __cplusplus
}
#endif

#endif
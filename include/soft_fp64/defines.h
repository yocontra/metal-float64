#pragma once

// Attribute macros shared across the soft-fp64 surface.
//
// SPDX-License-Identifier: MIT

#if defined(__clang__) || defined(__GNUC__)
#define SF64_ALWAYS_INLINE __attribute__((always_inline)) inline
#define SF64_NOINLINE __attribute__((noinline))
#define SF64_EXPORT __attribute__((visibility("default")))
#elif defined(_MSC_VER)
#define SF64_ALWAYS_INLINE __forceinline
#define SF64_NOINLINE __declspec(noinline)
#define SF64_EXPORT __declspec(dllexport)
#else
#define SF64_ALWAYS_INLINE inline
#define SF64_NOINLINE
#define SF64_EXPORT
#endif

// Entry points consumed by AdaptiveCpp's MSL emitter are `extern "C"` with a
// predefined symbol prefix. Consumers that want to hook into AdaptiveCpp set
// this to the emitter's expected attribute (e.g. HIPSYCL_SSCP_BUILTIN) in
// their own build; stand-alone builds fall back to `extern "C"` + always-inline.
#ifndef SF64_ABI
#define SF64_ABI extern "C" SF64_ALWAYS_INLINE
#endif

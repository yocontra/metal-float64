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

// Disable function-body optimizations. Used on the bit-twiddle ABI entries
// (sf64_fabs / sf64_copysign / sf64_neg / sf64_fcmp) where clang -O3's
// InstCombine pattern-matches the integer ops back into llvm.fabs.f64 /
// llvm.copysign.f64 / fneg double / fcmp <pred> double. AdaptiveCpp's MSL
// emitter then routes those intrinsics back through __acpp_sscp_*_f64
// wrappers whose bodies forward to sf64_*, producing infinite mutual
// recursion that hangs the AGX command buffer (kIOGPUCommandBufferCallback
// ErrorHang). Apply only to the four cycle-risk bodies — bodies are 1–10
// lines of bit ops, so the optimizer cost is negligible. Clang-only:
// GCC builds don't feed the Metal pipeline.
#if defined(__clang__)
#define SF64_NO_OPT __attribute__((optnone))
#else
#define SF64_NO_OPT
#endif

// Entry points consumed by AdaptiveCpp's MSL emitter are `extern "C"` with a
// predefined symbol prefix. Consumers that want to hook into AdaptiveCpp set
// this to the emitter's expected attribute (e.g. HIPSYCL_SSCP_BUILTIN) in
// their own build; stand-alone builds fall back to `extern "C"` + always-inline.
#ifndef SF64_ABI
#define SF64_ABI extern "C" SF64_ALWAYS_INLINE
#endif

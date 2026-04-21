// Minimal platform header for Berkeley SoftFloat-3e and TestFloat-3e on
// macOS-arm64 / clang. Mirrors `build/Linux-x86_64-GCC/platform.h` from the
// upstream tree — the only host dependency is a little-endian byte order
// and an `inline` keyword that works under C99+.
//
// Referenced by the vendored .c sources via `-I` on the include path.
//
// SPDX-License-Identifier: BSD-3-Clause (matches upstream)
#pragma once

// All our supported hosts (x86_64, aarch64) are little-endian. Upstream
// keys raw bit-pattern construction off this macro in several headers.
#define LITTLEENDIAN 1

// clang honours C99 `inline` under __GNUC_STDC_INLINE__.
#ifdef __GNUC_STDC_INLINE__
#define INLINE inline
#else
#define INLINE extern inline
#endif

// No THREAD_LOCAL needed — the test process is single-threaded.
#define THREAD_LOCAL

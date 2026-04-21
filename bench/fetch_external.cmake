# Detect vendored comparison libraries under bench/external/.
#
# We never FetchContent these at configure time (that would gate every
# clean build on network access). Instead a shell helper `fetch_external.sh`
# clones them once into bench/external/. This file detects what's there.

set(_ext "${CMAKE_CURRENT_SOURCE_DIR}/external")

# The comparison libraries include a C subtree (Berkeley SoftFloat 3e is
# pure C). The root project declares LANGUAGES CXX only, so enable C here
# — same scope as the comparison bench itself, zero cost when the vendor
# tree is absent.
enable_language(C)

# --- Berkeley SoftFloat 3e --------------------------------------------------
set(SOFT_FP64_BENCH_HAVE_SOFTFLOAT OFF)
set(_sf_src  "${_ext}/softfloat/source")
set(_sf_inc  "${_ext}/softfloat/source/include")
set(_sf_spec "${_ext}/softfloat/source/8086")
if(EXISTS "${_sf_src}/f64_add.c" AND EXISTS "${_sf_inc}/softfloat.h")
    set(SOFT_FP64_BENCH_HAVE_SOFTFLOAT ON)
    message(STATUS "bench: Berkeley SoftFloat 3e detected at ${_ext}/softfloat")

    # Compile the portable core. The 8086/ spec includes the rounding mode,
    # tininess detection, and `specialize.h` that SoftFloat needs to build.
    file(GLOB _softfloat_sources
        "${_sf_src}/s_*.c"
        "${_sf_src}/f64_*.c"
        "${_sf_src}/i32_to_f64.c"
        "${_sf_src}/i64_to_f64.c"
        "${_sf_src}/ui32_to_f64.c"
        "${_sf_src}/ui64_to_f64.c"
        "${_sf_src}/softfloat_state.c")
    # We only need the f64 and support helpers for f64. Drop every file
    # that touches extF80 / f128 / f16 / f32 / f64M — these pull in
    # upstream bugs (incompatible function-pointer warnings under AppleClang)
    # and add build time for no benefit to a f64-only microbench.
    list(FILTER _softfloat_sources EXCLUDE REGEX "s_commonNaNToF(16|32|128|RawF16|RawF32|RawF128).*\\.c$")
    list(FILTER _softfloat_sources EXCLUDE REGEX "/f(16|32|128|128M|64M)_.*\\.c$")
    list(FILTER _softfloat_sources EXCLUDE REGEX "/s_.*([Ee]xt[Ff]80|[Ff]128M|[Ff]128|[Ff]16|[Ff]32).*\\.c$")
    # `s_*M.c` = multi-precision helpers used only by f128M / extF80M paths
    # we've already excluded above. They reference `softfloat_*M` functions
    # that only exist in those excluded TUs, so they fail to link.
    list(FILTER _softfloat_sources EXCLUDE REGEX "/s_[A-Za-z]+M\\.c$")

    file(GLOB _softfloat_8086 "${_sf_spec}/*.c")
    # Only keep f64 (and shared) specializations from the 8086 spec dir.
    # Filenames mix case (`s_f128M...` lowercase, `s_commonNaNToF128M`
    # uppercase) — match both.
    list(FILTER _softfloat_8086 EXCLUDE REGEX "[Ee]xt[Ff]80|[Ff]128M|[Ff]128|[Ff]16|[Ff]32")

    add_library(softfloat3e_bench STATIC
        ${_softfloat_sources}
        ${_softfloat_8086})
    # Upstream SoftFloat source files `#include "platform.h"` where
    # `platform.h` is a per-target knob file that lives under
    # build/<Target>/. The bench build uses the Linux-x86_64-GCC knobs on
    # every host (LITTLEENDIAN=1, SOFTFLOAT_BUILTIN_CLZ, INTRINSIC_INT128)
    # because the bench runners (macOS/AArch64, Linux/x86_64) share those
    # invariants — SoftFloat has no AArch64-specific tuning and the
    # GCC-builtin-clz path works under AppleClang too.
    target_include_directories(softfloat3e_bench PUBLIC
        "${_sf_inc}"
        "${_sf_spec}"
        "${_ext}/softfloat/build/Linux-x86_64-GCC")
    target_compile_options(softfloat3e_bench PRIVATE
        -O3 -w  # Upstream ships with gcc-specific warnings; silence them.
        -DLITTLEENDIAN=1
        -DINLINE_LEVEL=5
        -DSOFTFLOAT_FAST_INT64
        -DSOFTFLOAT_FAST_DIV64TO32)
    set(SOFT_FP64_BENCH_SOFTFLOAT_INCLUDE "${_sf_inc}" PARENT_SCOPE)
endif()

# --- ckormanyos/soft_double -------------------------------------------------
set(SOFT_FP64_BENCH_HAVE_SOFT_DOUBLE OFF)
set(_sd_hdr "${_ext}/soft_double/math/softfloat/soft_double.h")
if(EXISTS "${_sd_hdr}")
    set(SOFT_FP64_BENCH_HAVE_SOFT_DOUBLE ON)
    set(SOFT_FP64_BENCH_SOFT_DOUBLE_INCLUDE "${_ext}/soft_double")
    message(STATUS "bench: ckormanyos/soft_double detected at ${_ext}/soft_double")
endif()

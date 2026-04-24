# libFuzzer targets for soft-fp64.
#
# The root CMakeLists gates entry via `option(SOFT_FP64_BUILD_FUZZ)` (OFF
# by default) and an `EXISTS` check, so default developer builds are
# unaffected.
#
# libFuzzer is shipped with clang; guard on compiler ID and surface a
# helpful error if the user forgot `-DCMAKE_CXX_COMPILER=clang++`.
#
# The exhaustive f32<->f64 round-trip test is a plain executable wired
# to ctest via `add_test(...)` further down — it does NOT depend on
# libFuzzer/sanitizers, so it is always built when the fuzz subdir is
# added.

# ---- Exhaustive f32<->f64 round-trip (no fuzzer, plain ctest) ----------
#
# Always built when this subdir is active. Uses only the host compiler
# (no libFuzzer / ASan dependency). Runs as ctest `test_exhaustive_f32`.

add_executable(exhaustive_f32_f64_roundtrip
    exhaustive_f32_f64_roundtrip.cpp)
target_link_libraries(exhaustive_f32_f64_roundtrip PRIVATE soft_fp64::soft_fp64)
target_compile_options(exhaustive_f32_f64_roundtrip PRIVATE
    -O2
    -ffp-contract=off
    -fno-math-errno
    -fno-trapping-math)

# Register as a ctest.  Time budget: well under 10 minutes on M-series.
# Mark as a long-running test so devs can exclude it via `-LE long`.
add_test(NAME test_exhaustive_f32_roundtrip
         COMMAND exhaustive_f32_f64_roundtrip)
set_tests_properties(test_exhaustive_f32_roundtrip PROPERTIES
    TIMEOUT 900
    LABELS "long;exhaustive")

# ---- libFuzzer per-op targets ------------------------------------------
#
# libFuzzer is built-in to clang; requires `-fsanitize=fuzzer,address,undefined`
# at both compile and link time.  If the active compiler isn't clang we skip
# these silently — the option may still be ON because the user also wants the
# exhaustive round-trip test registered.

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
   OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")

    set(_sf64_fuzz_flags
        -O1
        -g
        -fsanitize=fuzzer,address,undefined
        -fno-omit-frame-pointer
        # Match main-library FP strictness so fused/contract paths can't
        # hide divergences behind host FPU contraction.
        -ffp-contract=off
        -fno-math-errno
        -fno-trapping-math)

    set(_sf64_fuzz_link
        -fsanitize=fuzzer,address,undefined)

    function(_sf64_add_fuzz_target name source)
        add_executable(${name} ${source})
        target_link_libraries(${name} PRIVATE soft_fp64::soft_fp64)
        target_compile_options(${name} PRIVATE ${_sf64_fuzz_flags})
        target_link_options(${name} PRIVATE ${_sf64_fuzz_link})
        # Fuzz binaries are not useful as install artifacts; keep them out of
        # any `cmake --install` step.
        set_target_properties(${name} PROPERTIES EXCLUDE_FROM_ALL FALSE)
    endfunction()

    _sf64_add_fuzz_target(fuzz_arithmetic    fuzz_arithmetic.cpp)
    _sf64_add_fuzz_target(fuzz_sqrt_fma      fuzz_sqrt_fma.cpp)
    _sf64_add_fuzz_target(fuzz_transcendental fuzz_transcendental.cpp)
    _sf64_add_fuzz_target(fuzz_fcmp          fuzz_fcmp.cpp)
    _sf64_add_fuzz_target(fuzz_remainder     fuzz_remainder.cpp)
    _sf64_add_fuzz_target(fuzz_narrow_int    fuzz_narrow_int.cpp)

    # fuzz_rounding_modes links MPFR as the bit-exact oracle for the
    # sf64_*_r surface. Mirror the pkg-config / find_library discovery
    # from tests/mpfr/CMakeLists.txt so the target is skipped cleanly
    # when MPFR is absent (keeps the default fuzz build workable on
    # minimal hosts).
    find_package(PkgConfig QUIET)
    set(_fuzz_mpfr_found FALSE)
    set(_fuzz_mpfr_include_dirs "")
    set(_fuzz_mpfr_libraries "")
    if(PkgConfig_FOUND)
        pkg_check_modules(FUZZ_MPFR_PC QUIET mpfr)
        if(FUZZ_MPFR_PC_FOUND)
            set(_fuzz_mpfr_found TRUE)
            set(_fuzz_mpfr_include_dirs ${FUZZ_MPFR_PC_INCLUDE_DIRS})
            set(_fuzz_mpfr_libraries ${FUZZ_MPFR_PC_LINK_LIBRARIES})
            if(NOT _fuzz_mpfr_libraries)
                set(_fuzz_mpfr_libraries ${FUZZ_MPFR_PC_LIBRARIES})
            endif()
            pkg_check_modules(FUZZ_GMP_PC QUIET gmp)
            if(FUZZ_GMP_PC_FOUND)
                list(APPEND _fuzz_mpfr_include_dirs ${FUZZ_GMP_PC_INCLUDE_DIRS})
            endif()
        endif()
    endif()
    if(NOT _fuzz_mpfr_found)
        find_library(FUZZ_MPFR_LIB mpfr)
        find_path(FUZZ_MPFR_INCLUDE mpfr.h)
        if(FUZZ_MPFR_LIB AND FUZZ_MPFR_INCLUDE)
            set(_fuzz_mpfr_found TRUE)
            set(_fuzz_mpfr_include_dirs ${FUZZ_MPFR_INCLUDE})
            set(_fuzz_mpfr_libraries ${FUZZ_MPFR_LIB})
            find_library(FUZZ_GMP_LIB gmp)
            find_path(FUZZ_GMP_INCLUDE gmp.h)
            if(FUZZ_GMP_LIB)
                list(APPEND _fuzz_mpfr_libraries ${FUZZ_GMP_LIB})
            endif()
            if(FUZZ_GMP_INCLUDE)
                list(APPEND _fuzz_mpfr_include_dirs ${FUZZ_GMP_INCLUDE})
            endif()
        endif()
    endif()
    if(_fuzz_mpfr_found)
        _sf64_add_fuzz_target(fuzz_rounding_modes fuzz_rounding_modes.cpp)
        target_include_directories(fuzz_rounding_modes PRIVATE ${_fuzz_mpfr_include_dirs})
        target_link_libraries(fuzz_rounding_modes PRIVATE ${_fuzz_mpfr_libraries})
        message(STATUS "soft_fp64: fuzz_rounding_modes enabled (MPFR=${_fuzz_mpfr_include_dirs})")
    else()
        message(STATUS "soft_fp64: fuzz_rounding_modes skipped (MPFR not found)")
    endif()

    message(STATUS "soft_fp64: libFuzzer targets enabled "
                   "(fuzz_arithmetic, fuzz_sqrt_fma, fuzz_transcendental, "
                   "fuzz_fcmp, fuzz_remainder, fuzz_narrow_int, fuzz_rounding_modes)")
else()
    message(WARNING
        "SOFT_FP64_BUILD_FUZZ=ON but the active C++ compiler is "
        "'${CMAKE_CXX_COMPILER_ID}'. libFuzzer targets require clang. "
        "Reconfigure with -DCMAKE_CXX_COMPILER=clang++ to build fuzz_*.\n"
        "The exhaustive_f32_f64_roundtrip test is still enabled.")
endif()

# MPFR arbitrary-precision differential oracle.
#
# Gracefully skips configuration if MPFR is not installed on the host. The
# build does not hard-fail; an informational STATUS message is emitted.

set(_mpfr_found FALSE)
set(_mpfr_include_dirs "")
set(_mpfr_libraries "")

# Prefer pkg-config (which `brew install mpfr` populates via
# /opt/homebrew/Cellar/mpfr/*/lib/pkgconfig/mpfr.pc).
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(MPFR_PC QUIET mpfr)
    if(MPFR_PC_FOUND)
        set(_mpfr_found TRUE)
        set(_mpfr_include_dirs ${MPFR_PC_INCLUDE_DIRS})
        # Use the linker-friendly variant so -L/-l flags are honored.
        set(_mpfr_libraries ${MPFR_PC_LINK_LIBRARIES})
        if(NOT _mpfr_libraries)
            set(_mpfr_libraries ${MPFR_PC_LIBRARIES})
        endif()
        # MPFR's pkg-config on Homebrew does not expose GMP's include dir,
        # but mpfr.h transitively pulls in <gmp.h>. Resolve GMP too.
        pkg_check_modules(GMP_PC QUIET gmp)
        if(GMP_PC_FOUND)
            list(APPEND _mpfr_include_dirs ${GMP_PC_INCLUDE_DIRS})
        endif()
    endif()
endif()

# Fallback to plain find_library / find_path when pkg-config is unavailable.
if(NOT _mpfr_found)
    find_library(MPFR_LIB mpfr)
    find_path(MPFR_INCLUDE mpfr.h)
    if(MPFR_LIB AND MPFR_INCLUDE)
        set(_mpfr_found TRUE)
        set(_mpfr_include_dirs ${MPFR_INCLUDE})
        set(_mpfr_libraries ${MPFR_LIB})
        find_library(GMP_LIB gmp)
        find_path(GMP_INCLUDE gmp.h)
        if(GMP_LIB)
            list(APPEND _mpfr_libraries ${GMP_LIB})
        endif()
        if(GMP_INCLUDE)
            list(APPEND _mpfr_include_dirs ${GMP_INCLUDE})
        endif()
    endif()
endif()

if(NOT _mpfr_found)
    message(STATUS "MPFR not found — skipping test_mpfr_diff "
                   "(install with `brew install mpfr` on macOS, "
                   "`apt install libmpfr-dev` on Debian/Ubuntu).")
    return()
endif()

set(_src "${CMAKE_CURRENT_SOURCE_DIR}/test_mpfr_diff.cpp")
if(NOT EXISTS ${_src})
    message(STATUS "test_mpfr_diff.cpp not present — configure no-op.")
    return()
endif()

add_executable(test_mpfr_diff test_mpfr_diff.cpp)
target_include_directories(test_mpfr_diff PRIVATE ${_mpfr_include_dirs})
target_link_libraries(test_mpfr_diff
    PRIVATE
        soft_fp64::soft_fp64
        ${_mpfr_libraries}
)
# Matching the other sf64 tests: forbid host FMA contraction and libm
# errno side-effects. Soft-fp64 correctness depends on these.
target_compile_options(test_mpfr_diff PRIVATE -ffp-contract=off -fno-math-errno)

add_test(NAME test_mpfr_diff COMMAND test_mpfr_diff)
# MPFR sweeps at precision 200 are not fast — give the driver room.
set_tests_properties(test_mpfr_diff PROPERTIES TIMEOUT 300)

message(STATUS "test_mpfr_diff registered "
               "(MPFR include=${_mpfr_include_dirs})")

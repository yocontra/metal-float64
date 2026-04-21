# Berkeley TestFloat-3 oracle for soft-fp64.
#
# This subdirectory is lazily pulled in by `tests/CMakeLists.txt` once this
# file exists. It builds:
#
#   softfloat_oracle     — static lib from vendored berkeley-softfloat-3
#   testfloat_gen        — vendor generator binary (prints hex test vectors
#                          to stdout for a given op name + rounding mode)
#   run_testfloat        — our runner: spawns `testfloat_gen` via popen,
#                          reads vectors, calls sf64_* and compares bit-
#                          exact. Single ctest target drives the whole
#                          oracle suite.
#
# The vendored sources live under `vendor/` and are unmodified — see
# their COPYING.txt files for license (BSD-3 style, SPDX: BSD-3-Clause).

cmake_minimum_required(VERSION 3.20)

# The vendored tree is plain C.
enable_language(C)

set(SF64_SOFTFLOAT_ROOT  ${CMAKE_CURRENT_SOURCE_DIR}/vendor/berkeley-softfloat-3)
set(SF64_TESTFLOAT_ROOT  ${CMAKE_CURRENT_SOURCE_DIR}/vendor/berkeley-testfloat-3)
set(SF64_SOFTFLOAT_SRC   ${SF64_SOFTFLOAT_ROOT}/source)
set(SF64_TESTFLOAT_SRC   ${SF64_TESTFLOAT_ROOT}/source)

if(NOT EXISTS ${SF64_SOFTFLOAT_SRC} OR NOT EXISTS ${SF64_TESTFLOAT_SRC})
    message(STATUS "TestFloat oracle: vendored sources missing — skipping")
    return()
endif()

# ---- softfloat_oracle static library ------------------------------------
#
# Use the same layout as berkeley-softfloat-3/build/Linux-x86_64-GCC:
#   - Compile every `source/*.c`
#   - PLUS every `source/8086-SSE/*.c` (NaN handling / raiseFlags
#     specialization — only one specialize dir picked, per upstream).
#   - Flags per Makefile: SOFTFLOAT_FAST_INT64, SOFTFLOAT_ROUND_ODD,
#     INLINE_LEVEL=5, SOFTFLOAT_FAST_DIV32TO16, SOFTFLOAT_FAST_DIV64TO32.

file(GLOB SF64_SOFTFLOAT_C_TOP CONFIGURE_DEPENDS ${SF64_SOFTFLOAT_SRC}/*.c)
file(GLOB SF64_SOFTFLOAT_C_SPEC CONFIGURE_DEPENDS ${SF64_SOFTFLOAT_SRC}/8086-SSE/*.c)

# These slow-path multi-word primitive helpers are NOT compiled under
# SOFTFLOAT_FAST_INT64 (see OBJS_ALL in berkeley-softfloat-3's upstream
# Makefile — they are present in `source/` but never objected). Including
# them triggers "undeclared identifier" errors because their declarations
# live behind `#ifndef SOFTFLOAT_FAST_INT64` in internals.h.
set(SF64_SOFTFLOAT_SLOWPATH_EXCLUDE
    ${SF64_SOFTFLOAT_SRC}/s_addCarryM.c
    ${SF64_SOFTFLOAT_SRC}/s_addComplCarryM.c
    ${SF64_SOFTFLOAT_SRC}/s_addExtF80M.c
    ${SF64_SOFTFLOAT_SRC}/s_addF128M.c
    ${SF64_SOFTFLOAT_SRC}/s_addM.c
    ${SF64_SOFTFLOAT_SRC}/s_compare128M.c
    ${SF64_SOFTFLOAT_SRC}/s_compare96M.c
    ${SF64_SOFTFLOAT_SRC}/s_compareNonnormExtF80M.c
    ${SF64_SOFTFLOAT_SRC}/s_invalidExtF80M.c
    ${SF64_SOFTFLOAT_SRC}/s_invalidF128M.c
    ${SF64_SOFTFLOAT_SRC}/s_isNaNF128M.c
    ${SF64_SOFTFLOAT_SRC}/s_mul128MTo256M.c
    ${SF64_SOFTFLOAT_SRC}/s_mul64To128M.c
    ${SF64_SOFTFLOAT_SRC}/s_mulAddF128M.c
    ${SF64_SOFTFLOAT_SRC}/s_negXM.c
    ${SF64_SOFTFLOAT_SRC}/s_normExtF80SigM.c
    ${SF64_SOFTFLOAT_SRC}/s_normRoundPackMToExtF80M.c
    ${SF64_SOFTFLOAT_SRC}/s_normRoundPackMToF128M.c
    ${SF64_SOFTFLOAT_SRC}/s_normSubnormalF128SigM.c
    ${SF64_SOFTFLOAT_SRC}/s_remStepMBy32.c
    ${SF64_SOFTFLOAT_SRC}/s_roundMToI64.c
    ${SF64_SOFTFLOAT_SRC}/s_roundMToUI64.c
    ${SF64_SOFTFLOAT_SRC}/s_roundPackMToExtF80M.c
    ${SF64_SOFTFLOAT_SRC}/s_roundPackMToF128M.c
    ${SF64_SOFTFLOAT_SRC}/s_shiftLeftM.c
    ${SF64_SOFTFLOAT_SRC}/s_shiftNormSigF128M.c
    ${SF64_SOFTFLOAT_SRC}/s_shiftRightJamM.c
    ${SF64_SOFTFLOAT_SRC}/s_shiftRightM.c
    ${SF64_SOFTFLOAT_SRC}/s_shortShiftLeft64To96M.c
    ${SF64_SOFTFLOAT_SRC}/s_shortShiftLeftM.c
    ${SF64_SOFTFLOAT_SRC}/s_shortShiftRightExtendM.c
    ${SF64_SOFTFLOAT_SRC}/s_shortShiftRightJamM.c
    ${SF64_SOFTFLOAT_SRC}/s_shortShiftRightM.c
    ${SF64_SOFTFLOAT_SRC}/s_sub1XM.c
    ${SF64_SOFTFLOAT_SRC}/s_subM.c
    ${SF64_SOFTFLOAT_SRC}/s_tryPropagateNaNExtF80M.c
    ${SF64_SOFTFLOAT_SRC}/s_tryPropagateNaNF128M.c)
list(REMOVE_ITEM SF64_SOFTFLOAT_C_TOP ${SF64_SOFTFLOAT_SLOWPATH_EXCLUDE})

# Same story in the specialize dir — the multi-word M variants rely on
# SOFTFLOAT_FAST_INT64 being disabled.
set(SF64_SOFTFLOAT_SPEC_EXCLUDE
    ${SF64_SOFTFLOAT_SRC}/8086-SSE/s_commonNaNToExtF80M.c
    ${SF64_SOFTFLOAT_SRC}/8086-SSE/s_commonNaNToF128M.c
    ${SF64_SOFTFLOAT_SRC}/8086-SSE/s_extF80MToCommonNaN.c
    ${SF64_SOFTFLOAT_SRC}/8086-SSE/s_f128MToCommonNaN.c
    ${SF64_SOFTFLOAT_SRC}/8086-SSE/s_propagateNaNExtF80M.c
    ${SF64_SOFTFLOAT_SRC}/8086-SSE/s_propagateNaNF128M.c)
list(REMOVE_ITEM SF64_SOFTFLOAT_C_SPEC ${SF64_SOFTFLOAT_SPEC_EXCLUDE})

add_library(softfloat_oracle STATIC ${SF64_SOFTFLOAT_C_TOP} ${SF64_SOFTFLOAT_C_SPEC})
target_include_directories(softfloat_oracle PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${SF64_SOFTFLOAT_SRC}/8086-SSE
    ${SF64_SOFTFLOAT_SRC}/include)
target_compile_definitions(softfloat_oracle PUBLIC
    SOFTFLOAT_FAST_INT64
    SOFTFLOAT_ROUND_ODD
    INLINE_LEVEL=5
    SOFTFLOAT_FAST_DIV32TO16
    SOFTFLOAT_FAST_DIV64TO32)
target_compile_options(softfloat_oracle PRIVATE
    -Werror-implicit-function-declaration
    -Wno-unused-but-set-variable
    -Wno-parentheses
    -Wno-empty-body
    -Wno-pointer-sign
    -Wno-tautological-constant-out-of-range-compare
    -O2)
set_target_properties(softfloat_oracle PROPERTIES
    C_STANDARD 11
    POSITION_INDEPENDENT_CODE ON)

# ---- testfloat_gen generator binary -------------------------------------
#
# OBJS_LIB per berkeley-testfloat-3/build/Linux-x86_64-GCC/Makefile covers
# every `source/*.c` *except* the program `main()` files:
#     testfloat.c, testsoftfloat.c, timesoftfloat.c, testfloat_ver.c,
#     slowfloat.c, verLoops.c, subjfloat_functions.c, subjfloat (subj-C)
# We add back testfloat_gen.c + genLoops.c (the generator driver).

file(GLOB SF64_TESTFLOAT_ALL_C CONFIGURE_DEPENDS ${SF64_TESTFLOAT_SRC}/*.c)
set(SF64_TESTFLOAT_EXCLUDE
    ${SF64_TESTFLOAT_SRC}/testfloat.c
    ${SF64_TESTFLOAT_SRC}/testsoftfloat.c
    ${SF64_TESTFLOAT_SRC}/timesoftfloat.c
    ${SF64_TESTFLOAT_SRC}/testfloat_ver.c
    ${SF64_TESTFLOAT_SRC}/slowfloat.c
    ${SF64_TESTFLOAT_SRC}/verLoops.c
    ${SF64_TESTFLOAT_SRC}/subjfloat_functions.c)
list(REMOVE_ITEM SF64_TESTFLOAT_ALL_C ${SF64_TESTFLOAT_EXCLUDE})

add_executable(testfloat_gen ${SF64_TESTFLOAT_ALL_C})
target_include_directories(testfloat_gen PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${SF64_TESTFLOAT_SRC}
    ${SF64_TESTFLOAT_SRC}/subj-C
    ${SF64_SOFTFLOAT_SRC}/include)
target_compile_definitions(testfloat_gen PRIVATE
    FLOAT16 BFLOAT16 FLOAT64 EXTFLOAT80 FLOAT128
    FLOAT_ROUND_ODD LONG_DOUBLE_IS_EXTFLOAT80)
target_compile_options(testfloat_gen PRIVATE
    -Werror-implicit-function-declaration
    -Wno-unused-but-set-variable
    -Wno-parentheses
    -Wno-empty-body
    -Wno-unused-function
    -Wno-pointer-sign
    -Wno-tautological-constant-out-of-range-compare
    -O2)
target_link_libraries(testfloat_gen PRIVATE softfloat_oracle)
set_target_properties(testfloat_gen PROPERTIES C_STANDARD 11)

# ---- run_testfloat: our sf64_* oracle runner ----------------------------

add_executable(run_testfloat run_testfloat.cpp)
target_link_libraries(run_testfloat PRIVATE soft_fp64::soft_fp64)
target_compile_options(run_testfloat PRIVATE -ffp-contract=off -fno-math-errno)
# Pass absolute path to testfloat_gen via a macro so popen() finds it
# regardless of ctest's working directory.
target_compile_definitions(run_testfloat PRIVATE
    "SF64_TESTFLOAT_GEN_PATH=\"$<TARGET_FILE:testfloat_gen>\"")
add_dependencies(run_testfloat testfloat_gen)

add_test(NAME test_testfloat COMMAND run_testfloat)
set_tests_properties(test_testfloat PROPERTIES TIMEOUT 600)

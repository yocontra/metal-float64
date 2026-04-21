# Report-only harness for sf64_* input regions parked outside the shipped
# correctness claim. The binary prints observed ULP against libm and always
# exits 0 — it is NOT a ctest gate, on purpose.
#
# Ctest still runs it so the output is captured in every CI log and a
# precision regression shows up as a max_ulp spike, but it will never turn
# CI red by itself. Making any sweep here gate on a numeric bound requires
# moving it into tests/test_transcendental_1ulp.cpp or tests/mpfr/.
#
# To promote a sweep back to the gated tree:
#   1. Fix the underlying algorithm (DD stitching, Temme asymptotics, ...).
#   2. Verify the sweep lands inside a spec tier (U10/U35/GAMMA).
#   3. Move the sweep to the gated tree with the correct tier.
#   4. Delete the corresponding block from experimental_precision.cpp.

add_executable(experimental_precision experimental_precision.cpp)
target_link_libraries(experimental_precision PRIVATE soft_fp64::soft_fp64)
target_compile_options(experimental_precision PRIVATE -ffp-contract=off -fno-math-errno)
add_test(NAME experimental_precision COMMAND experimental_precision)
set_tests_properties(experimental_precision PROPERTIES LABELS "experimental")

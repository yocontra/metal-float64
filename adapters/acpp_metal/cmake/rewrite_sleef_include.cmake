# Helper: copy a sleef header into the staged dir with the relative
# upstream include path rewritten so the file resolves standalone in a
# flat directory.
#
# Invoked as:
#   cmake -DINPUT=<src> -DOUTPUT=<dst> -P rewrite_sleef_include.cmake
#
# Replaces both:
#   #include "../../include/soft_fp64/soft_f64.h"   →  #include "soft_fp64/soft_f64.h"
#   #include "../../include/soft_fp64/defines.h"    →  #include "soft_fp64/defines.h"
#
# The replacement is anchored on the leading "../../include/" literal so
# we cannot accidentally eat an unrelated path component.
#
# SPDX-License-Identifier: MIT

if(NOT INPUT OR NOT OUTPUT)
    message(FATAL_ERROR "rewrite_sleef_include.cmake requires -DINPUT=<src> -DOUTPUT=<dst>")
endif()

file(READ "${INPUT}" content)
string(REPLACE "../../include/soft_fp64/" "soft_fp64/" content "${content}")
# Sibling-escape rewrite for the private fenv shim: in the upstream
# tree, some SLEEF TUs reach `src/internal_fenv.h` via `../internal_fenv.h`.
# In the flat staged layout the header sits next to the .cpp.
string(REPLACE "../internal_fenv.h" "internal_fenv.h" content "${content}")
file(WRITE "${OUTPUT}" "${content}")

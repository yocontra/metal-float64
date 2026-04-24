# Helper: stage src/internal_fenv.h into the flat adapter tree with a
# fixed SOFT_FP64_FENV_MODE value baked in via an include of the
# companion `sf64_fenv_mode.h` header.
#
# Invoked as:
#   cmake -DINPUT=<src>/internal_fenv.h -DOUTPUT=<dst>/internal_fenv.h \
#         -P stage_internal_fenv.cmake
#
# Rationale. AdaptiveCpp's Metal libkernel globs the staged directory and
# compiles each TU with no extra -D flags under our control. The upstream
# `internal_fenv.h` falls back to `SOFT_FP64_FENV_MODE=1` when the macro
# is not externally defined — which means a core build configured with
# `-DSOFT_FP64_FENV=disabled` would still compile the Metal bitcode with
# the tls-mode code path unless we actively pin the staged copy.
#
# The fix is a two-line prepend: a `#include "sf64_fenv_mode.h"` placed
# before the original `#ifndef SOFT_FP64_FENV_MODE` guard so the
# sibling-staged config header wins, while the original guard remains
# intact as a safety net. No other bytes of `internal_fenv.h` change.
#
# SPDX-License-Identifier: MIT

if(NOT INPUT OR NOT OUTPUT)
    message(FATAL_ERROR "stage_internal_fenv.cmake requires -DINPUT=<src> -DOUTPUT=<dst>")
endif()

file(READ "${INPUT}" content)

# Prepend an include of the staged sf64_fenv_mode.h so the adapter's
# configured value wins over the in-header default. Placed immediately
# after the #pragma once so it is seen before any later `#ifndef
# SOFT_FP64_FENV_MODE` fallback.
set(INJECT "// -- injected by adapters/acpp_metal staging: pin SOFT_FP64_FENV_MODE --\n#include \"sf64_fenv_mode.h\"\n")
string(REPLACE "#pragma once" "#pragma once\n\n${INJECT}" content "${content}")

file(WRITE "${OUTPUT}" "${content}")

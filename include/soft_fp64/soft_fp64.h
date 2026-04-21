#pragma once

// soft-fp64 — portable soft-float IEEE-754 double-precision math.
//
// Bit-exact double arithmetic, conversion, and transcendentals implemented
// entirely on 32/64-bit integer bit operations. Intended for any target
// without hardware fp64: Apple GPUs, mobile / tile-based GPUs, WebGPU-class
// devices, embedded DSPs, FPGAs, custom accelerators.
//
// This umbrella header pulls in the public-facing scaffolding. The actual
// soft-fp64 entry points live in dedicated headers; include them directly if
// you only need a subset.
//
// SPDX-License-Identifier: MIT

#include "defines.h"
#include "double.h"
#include "soft_f64.h"
#include "vec.h"

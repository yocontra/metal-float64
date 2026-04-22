#pragma once

/**
 * @file rounding_mode.h
 * @brief IEEE-754 rounding-mode enum consumed by the `sf64_*_r` surface.
 *
 * Values match the numeric codes used by SoftFloat / Berkeley TestFloat 3e
 * (`softfloat_round_*`) so test harnesses do not need a translation table,
 * and bit-assignments are stable: `sf64_rounding_mode` is part of the ABI.
 *
 * The default `sf64_*` entry points are RNE-only; the `_r` variants take an
 * explicit mode and implement all five IEEE-754 rounding attributes.
 *
 * SPDX-License-Identifier: MIT
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief IEEE-754 rounding mode, as consumed by every `sf64_*_r` entry point.
 */
typedef enum sf64_rounding_mode {
    SF64_RNE = 0, /**< round to nearest, ties to even  (IEEE-754 default) */
    SF64_RTZ = 1, /**< round toward zero                                  */
    SF64_RUP = 2, /**< round toward +infinity                             */
    SF64_RDN = 3, /**< round toward -infinity                             */
    SF64_RNA = 4, /**< round to nearest, ties away from zero              */
} sf64_rounding_mode;

#ifdef __cplusplus
} /* extern "C" */
#endif

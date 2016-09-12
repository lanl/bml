#ifndef __BML_NORMALIZE_ELLPACK_H
#define __BML_NORMALIZE_ELLPACK_H

#include "bml_types_ellpack.h"

void bml_normalize_ellpack(
    bml_matrix_ellpack_t * A,
    const double mineval,
    const double maxeval);

void bml_normalize_ellpack_single_real(
    bml_matrix_ellpack_t * A,
    const double mineval,
    const double maxeval);

void bml_normalize_ellpack_double_real(
    bml_matrix_ellpack_t * A,
    const double mineval,
    const double maxeval);

void bml_normalize_ellpack_single_complex(
    bml_matrix_ellpack_t * A,
    const double mineval,
    const double maxeval);

void bml_normalize_ellpack_double_complex(
    bml_matrix_ellpack_t * A,
    const double mineval,
    const double maxeval);

void *bml_gershgorin_ellpack(
    const bml_matrix_ellpack_t * A);

void *bml_gershgorin_ellpack_single_real(
    const bml_matrix_ellpack_t * A);

void *bml_gershgorin_ellpack_double_real(
    const bml_matrix_ellpack_t * A);

void *bml_gershgorin_ellpack_single_complex(
    const bml_matrix_ellpack_t * A);

void *bml_gershgorin_ellpack_double_complex(
    const bml_matrix_ellpack_t * A);

void *bml_gershgorin_partial_ellpack(
    const bml_matrix_ellpack_t * A,
    const int nrows);

void *bml_gershgorin_partial_ellpack_single_real(
    const bml_matrix_ellpack_t * A,
    const int nrows);

void *bml_gershgorin_partial_ellpack_double_real(
    const bml_matrix_ellpack_t * A,
    const int nrows);

void *bml_gershgorin_partial_ellpack_single_complex(
    const bml_matrix_ellpack_t * A,
    const int nrows);

void *bml_gershgorin_partial_ellpack_double_complex(
    const bml_matrix_ellpack_t * A,
    const int nrows);

#endif

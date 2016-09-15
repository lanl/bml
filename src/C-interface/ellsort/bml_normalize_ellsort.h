#ifndef __BML_NORMALIZE_ELLSORT_H
#define __BML_NORMALIZE_ELLSORT_H

#include "bml_types_ellsort.h"

void bml_normalize_ellsort(
    bml_matrix_ellsort_t * A,
    const double mineval,
    const double maxeval);

void bml_normalize_ellsort_single_real(
    bml_matrix_ellsort_t * A,
    const double mineval,
    const double maxeval);

void bml_normalize_ellsort_double_real(
    bml_matrix_ellsort_t * A,
    const double mineval,
    const double maxeval);

void bml_normalize_ellsort_single_complex(
    bml_matrix_ellsort_t * A,
    const double mineval,
    const double maxeval);

void bml_normalize_ellsort_double_complex(
    bml_matrix_ellsort_t * A,
    const double mineval,
    const double maxeval);

void *bml_gershgorin_ellsort(
    const bml_matrix_ellsort_t * A);

void *bml_gershgorin_ellsort_single_real(
    const bml_matrix_ellsort_t * A);

void *bml_gershgorin_ellsort_double_real(
    const bml_matrix_ellsort_t * A);

void *bml_gershgorin_ellsort_single_complex(
    const bml_matrix_ellsort_t * A);

void *bml_gershgorin_ellsort_double_complex(
    const bml_matrix_ellsort_t * A);

void *bml_gershgorin_partial_ellsort(
    const bml_matrix_ellsort_t * A,
    const int nrows);

void *bml_gershgorin_partial_ellsort_single_real(
    const bml_matrix_ellsort_t * A,
    const int nrows);

void *bml_gershgorin_partial_ellsort_double_real(
    const bml_matrix_ellsort_t * A,
    const int nrows);

void *bml_gershgorin_partial_ellsort_single_complex(
    const bml_matrix_ellsort_t * A,
    const int nrows);

void *bml_gershgorin_partial_ellsort_double_complex(
    const bml_matrix_ellsort_t * A,
    const int nrows);

#endif

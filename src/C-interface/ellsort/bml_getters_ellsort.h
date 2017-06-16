/** \file */

#ifndef __BML_GETTERS_ELLSORT_H
#define __BML_GETTERS_ELLSORT_H

#include "bml_types_ellsort.h"

#include <complex.h>

void *bml_get_ellsort(
    const bml_matrix_ellsort_t * A,
    const int i,
    const int j);

float *bml_get_ellsort_single_real(
    const bml_matrix_ellsort_t * A,
    const int i,
    const int j);

double *bml_get_ellsort_double_real(
    const bml_matrix_ellsort_t * A,
    const int i,
    const int j);

float complex *bml_get_ellsort_single_complex(
    const bml_matrix_ellsort_t * A,
    const int i,
    const int j);

double complex *bml_get_ellsort_double_complex(
    const bml_matrix_ellsort_t * A,
    const int i,
    const int j);

void *bml_get_row_ellsort(
    bml_matrix_ellsort_t * A,
    const int i);

void *bml_get_row_ellsort_single_real(
    bml_matrix_ellsort_t * A,
    const int i);

void *bml_get_row_ellsort_double_real(
    bml_matrix_ellsort_t * A,
    const int i);

void *bml_get_row_ellsort_single_complex(
    bml_matrix_ellsort_t * A,
    const int i);

void *bml_get_row_ellsort_double_complex(
    bml_matrix_ellsort_t * A,
    const int i);

void *bml_get_diagonal_ellsort(
    bml_matrix_ellsort_t * A);

void *bml_get_diagonal_ellsort_single_real(
    bml_matrix_ellsort_t * A);

void *bml_get_diagonal_ellsort_double_real(
    bml_matrix_ellsort_t * A);

void *bml_get_diagonal_ellsort_single_complex(
    bml_matrix_ellsort_t * A);

void *bml_get_diagonal_ellsort_double_complex(
    bml_matrix_ellsort_t * A);

#endif

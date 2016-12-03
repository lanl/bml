/** \file */

#ifndef __BML_GETTERS_ELLSORT_H
#define __BML_GETTERS_ELLSORT_H

#include "bml_types_ellsort.h"

#include <complex.h>

// Getters for diagonal

void bml_get_diagonal_ellsort(
    bml_matrix_ellsort_t * A,
    void *diagonal);

void bml_get_diagonal_ellsort_single_real(
    bml_matrix_ellsort_t * A,
    float *diagonal);

void bml_get_diagonal_ellsort_double_real(
    bml_matrix_ellsort_t * A,
    double *diagonal);

void bml_get_diagonal_ellsort_single_complex(
    bml_matrix_ellsort_t * A,
    float complex * diagonal);

void bml_get_diagonal_ellsort_double_complex(
    bml_matrix_ellsort_t * A,
    double complex * diagonal);


// Getters for row

void bml_get_row_ellsort(
    bml_matrix_ellsort_t * A,
    const int i,
    void *row);

void bml_get_row_ellsort_single_real(
    bml_matrix_ellsort_t * A,
    const int i,
    float *row);

void bml_get_row_ellsort_double_real(
    bml_matrix_ellsort_t * A,
    const int i,
    double *row);

void bml_get_row_ellsort_single_complex(
    bml_matrix_ellsort_t * A,
    const int i,
    float complex * row);

void bml_get_row_ellsort_double_complex(
    bml_matrix_ellsort_t * A,
    const int i,
    double complex * row);

#endif

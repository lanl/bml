/** \file */

#ifndef __BML_SETTERS_ELLSORT_H
#define __BML_SETTERS_ELLSORT_H

#include "bml_types_ellsort.h"

#include <complex.h>

void bml_set_element_new_ellsort(
    bml_matrix_ellsort_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_element_new_ellsort_single_real(
    bml_matrix_ellsort_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_element_new_ellsort_double_real(
    bml_matrix_ellsort_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_element_new_ellsort_single_complex(
    bml_matrix_ellsort_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_element_new_ellsort_double_complex(
    bml_matrix_ellsort_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_element_ellsort(
    bml_matrix_ellsort_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_element_ellsort_single_real(
    bml_matrix_ellsort_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_element_ellsort_double_real(
    bml_matrix_ellsort_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_element_ellsort_single_complex(
    bml_matrix_ellsort_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_element_ellsort_double_complex(
    bml_matrix_ellsort_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_row_ellsort(
    bml_matrix_ellsort_t * A,
    const int i,
    const void *row,
    const double threshold);

void bml_set_row_ellsort_single_real(
    bml_matrix_ellsort_t * A,
    const int i,
    const void *row,
    const double threshold);

void bml_set_row_ellsort_double_real(
    bml_matrix_ellsort_t * A,
    const int i,
    const void *row,
    const double threshold);

void bml_set_row_ellsort_single_complex(
    bml_matrix_ellsort_t * A,
    const int i,
    const void *row,
    const double threshold);

void bml_set_row_ellsort_double_complex(
    bml_matrix_ellsort_t * A,
    const int i,
    const void *row,
    const double threshold);

void bml_set_diagonal_ellsort(
    bml_matrix_ellsort_t * A,
    const void *diagonal,
    const double threshold);

void bml_set_diagonal_ellsort_single_real(
    bml_matrix_ellsort_t * A,
    const void *diagonal,
    const double threshold);

void bml_set_diagonal_ellsort_double_real(
    bml_matrix_ellsort_t * A,
    const void *diagonal,
    const double threshold);

void bml_set_diagonal_ellsort_single_complex(
    bml_matrix_ellsort_t * A,
    const void *diagonal,
    const double threshold);

void bml_set_diagonal_ellsort_double_complex(
    bml_matrix_ellsort_t * A,
    const void *diagonal,
    const double threshold);

#endif

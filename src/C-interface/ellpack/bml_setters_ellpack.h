/** \file */

#ifndef __BML_SETTERS_ELLPACK_H
#define __BML_SETTERS_ELLPACK_H

#include "bml_types_ellpack.h"

#include <complex.h>

void bml_set_element_new_ellpack(
  bml_matrix_ellpack_t * A,
  const int i,
  const int j,
  const void *value);

void bml_set_element_ellpack(
    bml_matrix_ellpack_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_element_ellpack_single_real(
    bml_matrix_ellpack_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_element_ellpack_double_real(
    bml_matrix_ellpack_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_element_ellpack_single_complex(
    bml_matrix_ellpack_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_element_ellpack_double_complex(
    bml_matrix_ellpack_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_row_ellpack(
    bml_matrix_ellpack_t * A,
    const int i,
    const void *row,
    const double threshold);

void bml_set_row_ellpack_single_real(
    bml_matrix_ellpack_t * A,
    const int i,
    const float *row,
    const double threshold);

void bml_set_row_ellpack_double_real(
    bml_matrix_ellpack_t * A,
    const int i,
    const double *row,
    const double threshold);

void bml_set_row_ellpack_single_complex(
    bml_matrix_ellpack_t * A,
    const int i,
    const float complex * row,
    const double threshold);

void bml_set_row_ellpack_double_complex(
    bml_matrix_ellpack_t * A,
    const int i,
    const double complex * row,
    const double threshold);

void bml_set_diagonal_ellpack(
    bml_matrix_ellpack_t * A,
    const void *diagonal,
    const double threshold);

void bml_set_diagonal_ellpack_single_real(
    bml_matrix_ellpack_t * A,
    const float *diagonal,
    const double threshold);

void bml_set_diagonal_ellpack_double_real(
    bml_matrix_ellpack_t * A,
    const double *diagonal,
    const double threshold);

void bml_set_diagonal_ellpack_single_complex(
    bml_matrix_ellpack_t * A,
    const float complex * diagonal,
    const double threshold);

void bml_set_diagonal_ellpack_double_complex(
    bml_matrix_ellpack_t * A,
    const double complex * diagonal,
    const double threshold);

#endif

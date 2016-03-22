/** \file */

#ifndef __BML_GETTERS_DENSE_H
#define __BML_GETTERS_ELLPACK_H

#include "bml_types_ellpack.h"

#include <complex.h>

// Getters for diagonal

void bml_get_diagonal_ellpack(
  bml_matrix_ellpack_t * A,
  void *diagonal);

void bml_get_diagonal_ellpack_single_real(
  bml_matrix_ellpack_t * A,
  float *diagonal);

void bml_get_diagonal_ellpack_double_real(
  bml_matrix_ellpack_t * A,
  double *diagonal);

void bml_get_diagonal_ellpack_single_complex(
  bml_matrix_ellpack_t * A,
  float complex * diagonal);

void bml_get_diagonal_ellpack_double_complex(
  bml_matrix_ellpack_t * A,
  double complex * diagonal);


// Getters for row 

void bml_get_row_ellpack(
    bml_matrix_ellpack_t * A,
    const int i,
    void *row);

void bml_get_row_ellpack_single_real(
    bml_matrix_ellpack_t * A,
    const int i,
    float *row);

void bml_get_row_ellpack_double_real(
    bml_matrix_ellpack_t * A,
    const int i,
    double *row);

void bml_get_row_ellpack_single_complex(
    bml_matrix_ellpack_t * A,
    const int i,
    float complex * row);

void bml_get_row_ellpack_double_complex(
    bml_matrix_ellpack_t * A,
    const int i,
    double complex * row);

#endif

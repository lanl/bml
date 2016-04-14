/** \file */

#ifndef __BML_GETTERS_DENSE_H
#define __BML_GETTERS_DENSE_H

#include "bml_types_dense.h"

#include <complex.h>

/*
void bml_get_dense(
    bml_matrix_dense_t * A,
    const int i,
    const int j,
    void *value);
*/



// Getters for diagonal

void bml_get_diagonal_dense(
  bml_matrix_dense_t * A,
  void *diagonal);

void bml_get_diagonal_dense_single_real(
  bml_matrix_dense_t * A,
  float *diagonal);

void bml_get_diagonal_dense_double_real(
  bml_matrix_dense_t * A,
  double *diagonal);

void bml_get_diagonal_dense_single_complex(
  bml_matrix_dense_t * A,
  float complex * diagonal);

void bml_get_diagonal_dense_double_complex(
  bml_matrix_dense_t * A,
  double complex * diagonal);


// Getters for row

void bml_get_row_dense(
    bml_matrix_dense_t * A,
    const int i,
    void *row);

void bml_get_row_dense_single_real(
    bml_matrix_dense_t * A,
    const int i,
    float *row);

void bml_get_row_dense_double_real(
    bml_matrix_dense_t * A,
    const int i,
    double *row);

void bml_get_row_dense_single_complex(
    bml_matrix_dense_t * A,
    const int i,
    float complex * row);

void bml_get_row_dense_double_complex(
    bml_matrix_dense_t * A,
    const int i,
    double complex * row);

#endif

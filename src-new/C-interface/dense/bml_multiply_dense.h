#ifndef __BML_MULTIPLY_DENSE_H
#define __BML_MULTIPLY_DENSE_H

#include "bml_types_dense.h"

// Matrix multiply - C = alpha * A * B + beta * C
void bml_multiply_dense(
    const bml_matrix_dense_t * A,
    const bml_matrix_dense_t * B,
    const bml_matrix_dense_t * C,
    const double alpha,
    const double beta);

void bml_multiply_dense_single_real(
    const bml_matrix_dense_t * A,
    const bml_matrix_dense_t * B,
    const bml_matrix_dense_t * C,
    const double alpha,
    const double beta);

void bml_multiply_dense_double_real(
    const bml_matrix_dense_t * A,
    const bml_matrix_dense_t * B,
    const bml_matrix_dense_t * C,
    const double alpha,
    const double beta);

// Matrix X^2 - X2 = X * X
void bml_multiply_x2_dense(
    const bml_matrix_dense_t * X,
    const bml_matrix_dense_t * X2);

void bml_multiply_x2_dense_single_real(
    const bml_matrix_dense_t * X,
    const bml_matrix_dense_t * X2);

void bml_multiply_x2_dense_double_real(
    const bml_matrix_dense_t * X,
    const bml_matrix_dense_t * X2);

#endif

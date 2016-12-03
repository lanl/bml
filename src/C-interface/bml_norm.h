/** \file */

#ifndef __BML_NORM_H
#define __BML_NORM_H

#include "bml_types.h"

// Calculate the sum of squares of all the elements in A
double bml_sum_squares(
    const bml_matrix_t * A);

// Calculate the sum of squares of all the elements of
// alpha * A + beta * B
double bml_sum_squares2(
    const bml_matrix_t * A,
    const bml_matrix_t * B,
    const double alpha,
    const double beta,
    const double threshold);

// Calculate the sum of squares for submatrix core elements
double bml_sum_squares_submatrix(
    const bml_matrix_t * A,
    const int core_size);

// Calculate Frobenius norm for matrix A
// sqrt(sum(A_ij*A_ij)
double bml_fnorm(
    const bml_matrix_t * A);

// Calculate Frobenius norm for 2 matrices
double bml_fnorm2(
    const bml_matrix_t * A,
    const bml_matrix_t * B);

#endif

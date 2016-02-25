/** \file */

#ifndef __BML_NORM_H
#define __BML_NORM_H

#include "bml_types.h"

// Calculate the sum of squares of all the elements in A
double bml_sum_squares(
    const bml_matrix_t * A);

// Calculate the sum of suares of all the elements of
// alpha * A + beta * B
double bml_sum_squares2(
    const bml_matrix_t * A,
    const bml_matrix_t * B,
    const double alpha,
    const double beta);

// Calculate Frobenius norm for matrix A
// sqrt(sum(A_ij*A_ij)
double bml_fnorm(
    const bml_matrix_t * A);

#endif

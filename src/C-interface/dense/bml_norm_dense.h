#ifndef __BML_NORM_DENSE_H
#define __BML_NORM_DENSE_H

#include "bml_types_dense.h"

double bml_sum_squares_dense(
    const bml_matrix_dense_t * A);

double bml_sum_squares_dense_single_real(
    const bml_matrix_dense_t * A);

double bml_sum_squares_dense_double_real(
    const bml_matrix_dense_t * A);

double bml_sum_squares_dense_single_complex(
    const bml_matrix_dense_t * A);

double bml_sum_squares_dense_double_complex(
    const bml_matrix_dense_t * A);

double bml_sum_squares2_dense(
    const bml_matrix_dense_t * A,
    const bml_matrix_dense_t * B,
    const double alpha,
    const double beta);

double bml_sum_squares2_dense_single_real(
    const bml_matrix_dense_t * A,
    const bml_matrix_dense_t * B,
    const double alpha,
    const double beta);

double bml_sum_squares2_dense_double_real(
    const bml_matrix_dense_t * A,
    const bml_matrix_dense_t * B,
    const double alpha,
    const double beta);

double bml_sum_squares2_dense_single_complex(
    const bml_matrix_dense_t * A,
    const bml_matrix_dense_t * B,
    const double alpha,
    const double beta);

double bml_sum_squares2_dense_double_complex(
    const bml_matrix_dense_t * A,
    const bml_matrix_dense_t * B,
    const double alpha,
    const double beta);

double bml_fnorm_dense(
    const bml_matrix_dense_t * A);

double bml_fnorm_dense_single_real(
    const bml_matrix_dense_t * A); 

double bml_fnorm_dense_double_real(
    const bml_matrix_dense_t * A);

double bml_fnorm_dense_single_complex(
    const bml_matrix_dense_t * A);

double bml_fnorm_dense_double_complex(
    const bml_matrix_dense_t * A);

#endif

#ifndef __BML_NORM_ELLSORT_H
#define __BML_NORM_ELLSORT_H

#include "bml_types_ellsort.h"

double bml_sum_squares_ellsort(
    const bml_matrix_ellsort_t * A);

double bml_sum_squares_ellsort_single_real(
    const bml_matrix_ellsort_t * A);

double bml_sum_squares_ellsort_double_real(
    const bml_matrix_ellsort_t * A);

double bml_sum_squares_ellsort_single_complex(
    const bml_matrix_ellsort_t * A);

double bml_sum_squares_ellsort_double_complex(
    const bml_matrix_ellsort_t * A);

double bml_sum_squares_submatrix_ellsort(
    const bml_matrix_ellsort_t * A,
    const int core_size);

double bml_sum_squares_submatrix_ellsort_single_real(
    const bml_matrix_ellsort_t * A,
    const int core_size);

double bml_sum_squares_submatrix_ellsort_double_real(
    const bml_matrix_ellsort_t * A,
    const int core_size);

double bml_sum_squares_submatrix_ellsort_single_complex(
    const bml_matrix_ellsort_t * A,
    const int core_size);

double bml_sum_squares_submatrix_ellsort_double_complex(
    const bml_matrix_ellsort_t * A,
    const int core_size);

double bml_sum_squares2_ellsort(
    const bml_matrix_ellsort_t * A,
    const bml_matrix_ellsort_t * B,
    const double alpha,
    const double beta,
    const double threshold);

double bml_sum_squares2_ellsort_single_real(
    const bml_matrix_ellsort_t * A,
    const bml_matrix_ellsort_t * B,
    const double alpha,
    const double beta,
    const double threshold);

double bml_sum_squares2_ellsort_double_real(
    const bml_matrix_ellsort_t * A,
    const bml_matrix_ellsort_t * B,
    const double alpha,
    const double beta,
    const double threshold);

double bml_sum_squares2_ellsort_single_complex(
    const bml_matrix_ellsort_t * A,
    const bml_matrix_ellsort_t * B,
    const double alpha,
    const double beta,
    const double threshold);

double bml_sum_squares2_ellsort_double_complex(
    const bml_matrix_ellsort_t * A,
    const bml_matrix_ellsort_t * B,
    const double alpha,
    const double beta,
    const double threshold);

double bml_fnorm_ellsort(
    const bml_matrix_ellsort_t * A);

double bml_fnorm_ellsort_single_real(
    const bml_matrix_ellsort_t * A);

double bml_fnorm_ellsort_double_real(
    const bml_matrix_ellsort_t * A);

double bml_fnorm_ellsort_single_complex(
    const bml_matrix_ellsort_t * A);

double bml_fnorm_ellsort_double_complex(
    const bml_matrix_ellsort_t * A);

double bml_fnorm2_ellsort(
    const bml_matrix_ellsort_t * A,
    const bml_matrix_ellsort_t * B);

double bml_fnorm2_ellsort_single_real(
    const bml_matrix_ellsort_t * A,
    const bml_matrix_ellsort_t * B);

double bml_fnorm2_ellsort_double_real(
    const bml_matrix_ellsort_t * A,
    const bml_matrix_ellsort_t * B);

double bml_fnorm2_ellsort_single_complex(
    const bml_matrix_ellsort_t * A,
    const bml_matrix_ellsort_t * B);

double bml_fnorm2_ellsort_double_complex(
    const bml_matrix_ellsort_t * A,
    const bml_matrix_ellsort_t * B);

#endif

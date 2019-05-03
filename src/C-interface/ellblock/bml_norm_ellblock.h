#ifndef __BML_NORM_ELLBLOCK_H
#define __BML_NORM_ELLBLOCK_H

#include "bml_types_ellblock.h"

double bml_sum_squares_ellblock(
    const bml_matrix_ellblock_t * A);

double bml_sum_squares_ellblock_single_real(
    const bml_matrix_ellblock_t * A);

double bml_sum_squares_ellblock_double_real(
    const bml_matrix_ellblock_t * A);

double bml_sum_squares_ellblock_single_complex(
    const bml_matrix_ellblock_t * A);

double bml_sum_squares_ellblock_double_complex(
    const bml_matrix_ellblock_t * A);

double bml_sum_squares_submatrix_ellblock(
    const bml_matrix_ellblock_t * A,
    const int core_size);

double bml_sum_squares_submatrix_ellblock_single_real(
    const bml_matrix_ellblock_t * A,
    const int core_size);

double bml_sum_squares_submatrix_ellblock_double_real(
    const bml_matrix_ellblock_t * A,
    const int core_size);

double bml_sum_squares_submatrix_ellblock_single_complex(
    const bml_matrix_ellblock_t * A,
    const int core_size);

double bml_sum_squares_submatrix_ellblock_double_complex(
    const bml_matrix_ellblock_t * A,
    const int core_size);

double bml_sum_squares2_ellblock(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B,
    const double alpha,
    const double beta,
    const double threshold);

double bml_sum_squares2_ellblock_single_real(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B,
    const double alpha,
    const double beta,
    const double threshold);

double bml_sum_squares2_ellblock_double_real(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B,
    const double alpha,
    const double beta,
    const double threshold);

double bml_sum_squares2_ellblock_single_complex(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B,
    const double alpha,
    const double beta,
    const double threshold);

double bml_sum_squares2_ellblock_double_complex(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B,
    const double alpha,
    const double beta,
    const double threshold);

double bml_fnorm_ellblock(
    const bml_matrix_ellblock_t * A);

double bml_fnorm_ellblock_single_real(
    const bml_matrix_ellblock_t * A);

double bml_fnorm_ellblock_double_real(
    const bml_matrix_ellblock_t * A);

double bml_fnorm_ellblock_single_complex(
    const bml_matrix_ellblock_t * A);

double bml_fnorm_ellblock_double_complex(
    const bml_matrix_ellblock_t * A);

double bml_fnorm2_ellblock(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B);

double bml_fnorm2_ellblock_single_real(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B);

double bml_fnorm2_ellblock_double_real(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B);

double bml_fnorm2_ellblock_single_complex(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B);

double bml_fnorm2_ellblock_double_complex(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B);

#endif

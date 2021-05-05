#ifndef __BML_NORM_DISTRIBUTED2D_H
#define __BML_NORM_DISTRIBUTED2D_H

#include "bml_types_distributed2d.h"

double bml_sum_squares_distributed2d(
    bml_matrix_distributed2d_t * A);

double bml_sum_squares_submatrix_distributed2d(
    bml_matrix_distributed2d_t * A,
    int core_size);

double bml_sum_AB_distributed2d(
    bml_matrix_distributed2d_t * A,
    bml_matrix_distributed2d_t * B,
    double alpha,
    double threshold);

double bml_sum_squares2_distributed2d(
    bml_matrix_distributed2d_t * A,
    bml_matrix_distributed2d_t * B,
    double alpha,
    double beta,
    double threshold);

double bml_fnorm_distributed2d(
    bml_matrix_distributed2d_t * A);

double bml_fnorm2_distributed2d(
    bml_matrix_distributed2d_t * A,
    bml_matrix_distributed2d_t * B);

#endif

#ifndef __BML_MULTIPLY_ELLBLOCK_H
#define __BML_MULTIPLY_ELLBLOCK_H

#include "bml_types_ellblock.h"

void bml_multiply_ellblock(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B,
    bml_matrix_ellblock_t * C,
    const double alpha,
    const double beta,
    const double threshold);

void bml_multiply_ellblock_single_real(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B,
    bml_matrix_ellblock_t * C,
    const double alpha,
    const double beta,
    const double threshold);

void bml_multiply_ellblock_double_real(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B,
    bml_matrix_ellblock_t * C,
    const double alpha,
    const double beta,
    const double threshold);

void bml_multiply_ellblock_single_complex(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B,
    bml_matrix_ellblock_t * C,
    const double alpha,
    const double beta,
    const double threshold);

void bml_multiply_ellblock_double_complex(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B,
    bml_matrix_ellblock_t * C,
    const double alpha,
    const double beta,
    const double threshold);

void *bml_multiply_x2_ellblock(
    const bml_matrix_ellblock_t * X,
    bml_matrix_ellblock_t * X2,
    const double threshold);

void *bml_multiply_x2_ellblock_single_real(
    const bml_matrix_ellblock_t * X,
    bml_matrix_ellblock_t * X2,
    const double threshold);

void *bml_multiply_x2_ellblock_double_real(
    const bml_matrix_ellblock_t * X,
    bml_matrix_ellblock_t * X2,
    const double threshold);

void *bml_multiply_x2_ellblock_single_complex(
    const bml_matrix_ellblock_t * X,
    bml_matrix_ellblock_t * X2,
    const double threshold);

void *bml_multiply_x2_ellblock_double_complex(
    const bml_matrix_ellblock_t * X,
    bml_matrix_ellblock_t * X2,
    const double threshold);

void bml_multiply_AB_ellblock(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B,
    bml_matrix_ellblock_t * C,
    const double threshold);

void bml_multiply_AB_ellblock_single_real(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B,
    bml_matrix_ellblock_t * C,
    const double threshold);

void bml_multiply_AB_ellblock_double_real(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B,
    bml_matrix_ellblock_t * C,
    const double threshold);

void bml_multiply_AB_ellblock_single_complex(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B,
    bml_matrix_ellblock_t * C,
    const double threshold);

void bml_multiply_AB_ellblock_double_complex(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B,
    bml_matrix_ellblock_t * C,
    const double threshold);

void bml_multiply_adjust_AB_ellblock(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B,
    bml_matrix_ellblock_t * C,
    const double threshold);

void bml_multiply_adjust_AB_ellblock_single_real(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B,
    bml_matrix_ellblock_t * C,
    const double threshold);

void bml_multiply_adjust_AB_ellblock_double_real(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B,
    bml_matrix_ellblock_t * C,
    const double threshold);

void bml_multiply_adjust_AB_ellblock_single_complex(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B,
    bml_matrix_ellblock_t * C,
    const double threshold);

void bml_multiply_adjust_AB_ellblock_double_complex(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B,
    bml_matrix_ellblock_t * C,
    const double threshold);

#endif

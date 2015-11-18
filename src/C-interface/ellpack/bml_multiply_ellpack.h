#ifndef __BML_MULTIPLY_ELLPACK_H
#define __BML_MULTIPLY_ELLPACK_H

#include "bml_types_ellpack.h"

void bml_multiply_ellpack(
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    const double alpha,
    const double beta,
    const double threshold);

void bml_multiply_ellpack_single_real(
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    const double alpha,
    const double beta,
    const double threshold);

void bml_multiply_ellpack_double_real(
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    const double alpha,
    const double beta,
    const double threshold);

void bml_multiply_ellpack_single_complex(
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    const double alpha,
    const double beta,
    const double threshold);

void bml_multiply_ellpack_double_complex(
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    const double alpha,
    const double beta,
    const double threshold);

void bml_multiply_x2_ellpack(
    const bml_matrix_ellpack_t * X,
    bml_matrix_ellpack_t * X2,
    const double threshold);

void bml_multiply_x2_ellpack_single_real(
    const bml_matrix_ellpack_t * X,
    bml_matrix_ellpack_t * X2,
    const double threshold);

void bml_multiply_x2_ellpack_double_real(
    const bml_matrix_ellpack_t * X,
    bml_matrix_ellpack_t * X2,
    const double threshold);

void bml_multiply_x2_ellpack_single_complex(
    const bml_matrix_ellpack_t * X,
    bml_matrix_ellpack_t * X2,
    const double threshold);

void bml_multiply_x2_ellpack_double_complex(
    const bml_matrix_ellpack_t * X,
    bml_matrix_ellpack_t * X2,
    const double threshold);

void bml_multiply_AB_ellpack(
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    const double threshold);

void bml_multiply_AB_ellpack_single_real(
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    const double threshold);

void bml_multiply_AB_ellpack_double_real(
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    const double threshold);

void bml_multiply_AB_ellpack_single_complex(
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    const double threshold);

void bml_multiply_AB_ellpack_double_complex(
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    const double threshold);

void bml_multiply_adjust_AB_ellpack(
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    const double threshold);

void bml_multiply_adjust_AB_ellpack_single_real(
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    const double threshold);

void bml_multiply_adjust_AB_ellpack_double_real(
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    const double threshold);

void bml_multiply_adjust_AB_ellpack_single_complex(
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    const double threshold);

void bml_multiply_adjust_AB_ellpack_double_complex(
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    const double threshold);

#endif

#ifndef __BML_MULTIPLY_ELLSORT_H
#define __BML_MULTIPLY_ELLSORT_H

#include "bml_types_ellsort.h"

void bml_multiply_ellsort(
    const bml_matrix_ellsort_t * A,
    const bml_matrix_ellsort_t * B,
    bml_matrix_ellsort_t * C,
    const double alpha,
    const double beta,
    const double threshold);

void bml_multiply_ellsort_single_real(
    const bml_matrix_ellsort_t * A,
    const bml_matrix_ellsort_t * B,
    bml_matrix_ellsort_t * C,
    const double alpha,
    const double beta,
    const double threshold);

void bml_multiply_ellsort_double_real(
    const bml_matrix_ellsort_t * A,
    const bml_matrix_ellsort_t * B,
    bml_matrix_ellsort_t * C,
    const double alpha,
    const double beta,
    const double threshold);

void bml_multiply_ellsort_single_complex(
    const bml_matrix_ellsort_t * A,
    const bml_matrix_ellsort_t * B,
    bml_matrix_ellsort_t * C,
    const double alpha,
    const double beta,
    const double threshold);

void bml_multiply_ellsort_double_complex(
    const bml_matrix_ellsort_t * A,
    const bml_matrix_ellsort_t * B,
    bml_matrix_ellsort_t * C,
    const double alpha,
    const double beta,
    const double threshold);

void *bml_multiply_x2_ellsort(
    const bml_matrix_ellsort_t * X,
    bml_matrix_ellsort_t * X2,
    const double threshold);

void *bml_multiply_x2_ellsort_single_real(
    const bml_matrix_ellsort_t * X,
    bml_matrix_ellsort_t * X2,
    const double threshold);

void *bml_multiply_x2_ellsort_double_real(
    const bml_matrix_ellsort_t * X,
    bml_matrix_ellsort_t * X2,
    const double threshold);

void *bml_multiply_x2_ellsort_single_complex(
    const bml_matrix_ellsort_t * X,
    bml_matrix_ellsort_t * X2,
    const double threshold);

void *bml_multiply_x2_ellsort_double_complex(
    const bml_matrix_ellsort_t * X,
    bml_matrix_ellsort_t * X2,
    const double threshold);

void bml_multiply_AB_ellsort(
    const bml_matrix_ellsort_t * A,
    const bml_matrix_ellsort_t * B,
    bml_matrix_ellsort_t * C,
    const double threshold);

void bml_multiply_AB_ellsort_single_real(
    const bml_matrix_ellsort_t * A,
    const bml_matrix_ellsort_t * B,
    bml_matrix_ellsort_t * C,
    const double threshold);

void bml_multiply_AB_ellsort_double_real(
    const bml_matrix_ellsort_t * A,
    const bml_matrix_ellsort_t * B,
    bml_matrix_ellsort_t * C,
    const double threshold);

void bml_multiply_AB_ellsort_single_complex(
    const bml_matrix_ellsort_t * A,
    const bml_matrix_ellsort_t * B,
    bml_matrix_ellsort_t * C,
    const double threshold);

void bml_multiply_AB_ellsort_double_complex(
    const bml_matrix_ellsort_t * A,
    const bml_matrix_ellsort_t * B,
    bml_matrix_ellsort_t * C,
    const double threshold);

void bml_multiply_adjust_AB_ellsort(
    const bml_matrix_ellsort_t * A,
    const bml_matrix_ellsort_t * B,
    bml_matrix_ellsort_t * C,
    const double threshold);

void bml_multiply_adjust_AB_ellsort_single_real(
    const bml_matrix_ellsort_t * A,
    const bml_matrix_ellsort_t * B,
    bml_matrix_ellsort_t * C,
    const double threshold);

void bml_multiply_adjust_AB_ellsort_double_real(
    const bml_matrix_ellsort_t * A,
    const bml_matrix_ellsort_t * B,
    bml_matrix_ellsort_t * C,
    const double threshold);

void bml_multiply_adjust_AB_ellsort_single_complex(
    const bml_matrix_ellsort_t * A,
    const bml_matrix_ellsort_t * B,
    bml_matrix_ellsort_t * C,
    const double threshold);

void bml_multiply_adjust_AB_ellsort_double_complex(
    const bml_matrix_ellsort_t * A,
    const bml_matrix_ellsort_t * B,
    bml_matrix_ellsort_t * C,
    const double threshold);

#endif

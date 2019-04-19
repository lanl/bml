#ifndef __BML_SCALE_ELLBLOCK_H
#define __BML_SCALE_ELLBLOCK_H

#include "bml_types_ellblock.h"

#include <complex.h>

bml_matrix_ellblock_t *bml_scale_ellblock_new(
    const void *scale_factor,
    const bml_matrix_ellblock_t * A);

bml_matrix_ellblock_t *bml_scale_ellblock_new_single_real(
    const float *scale_factor,
    const bml_matrix_ellblock_t * A);

bml_matrix_ellblock_t *bml_scale_ellblock_new_double_real(
    const double *scale_factor,
    const bml_matrix_ellblock_t * A);

bml_matrix_ellblock_t *bml_scale_ellblock_new_single_complex(
    const float complex * scale_factor,
    const bml_matrix_ellblock_t * A);

bml_matrix_ellblock_t *bml_scale_ellblock_new_double_complex(
    const double complex * scale_factor,
    const bml_matrix_ellblock_t * A);

void bml_scale_ellblock(
    const void *scale_factor,
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B);

void bml_scale_ellblock_single_real(
    const float *scale_factor,
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B);

void bml_scale_ellblock_double_real(
    const double *scale_factor,
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B);

void bml_scale_ellblock_single_complex(
    const float complex * scale_factor,
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B);

void bml_scale_ellblock_double_complex(
    const double complex * scale_factor,
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B);

void bml_scale_inplace_ellblock(
    const void *scale_factor,
    bml_matrix_ellblock_t * A);

void bml_scale_inplace_ellblock_single_real(
    const float *scale_factor,
    bml_matrix_ellblock_t * A);

void bml_scale_inplace_ellblock_double_real(
    const double *scale_factor,
    bml_matrix_ellblock_t * A);

void bml_scale_inplace_ellblock_single_complex(
    const float complex * scale_factor,
    bml_matrix_ellblock_t * A);

void bml_scale_inplace_ellblock_double_complex(
    const double complex * scale_factor,
    bml_matrix_ellblock_t * A);

#endif

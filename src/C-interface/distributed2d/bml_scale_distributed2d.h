#ifndef __BML_SCALE_DISTRIBUTED2D_H
#define __BML_SCALE_DISTRIBUTED2D_H

#include "bml_types_distributed2d.h"

#include <complex.h>

bml_matrix_distributed2d_t *bml_scale_distributed2d_new(
    void *scale_factor,
    bml_matrix_distributed2d_t * A);

bml_matrix_distributed2d_t *bml_scale_distributed2d_new_single_real(
    void *scale_factor,
    bml_matrix_distributed2d_t * A);

bml_matrix_distributed2d_t *bml_scale_distributed2d_new_double_real(
    void *scale_factor,
    bml_matrix_distributed2d_t * A);

bml_matrix_distributed2d_t *bml_scale_distributed2d_new_single_complex(
    void *scale_factor,
    bml_matrix_distributed2d_t * A);

bml_matrix_distributed2d_t *bml_scale_distributed2d_new_double_complex(
    void *scale_factor,
    bml_matrix_distributed2d_t * A);

void bml_scale_distributed2d(
    void *scale_factor,
    bml_matrix_distributed2d_t * A,
    bml_matrix_distributed2d_t * B);

void bml_scale_distributed2d_single_real(
    void *scale_factor,
    bml_matrix_distributed2d_t * A,
    bml_matrix_distributed2d_t * B);

void bml_scale_distributed2d_double_real(
    void *scale_factor,
    bml_matrix_distributed2d_t * A,
    bml_matrix_distributed2d_t * B);

void bml_scale_distributed2d_single_complex(
    void *scale_factor,
    bml_matrix_distributed2d_t * A,
    bml_matrix_distributed2d_t * B);

void bml_scale_distributed2d_double_complex(
    void *scale_factor,
    bml_matrix_distributed2d_t * A,
    bml_matrix_distributed2d_t * B);

void bml_scale_inplace_distributed2d(
    void *scale_factor,
    bml_matrix_distributed2d_t * A);

void bml_scale_inplace_distributed2d_single_real(
    void *scale_factor,
    bml_matrix_distributed2d_t * A);

void bml_scale_inplace_distributed2d_double_real(
    void *scale_factor,
    bml_matrix_distributed2d_t * A);

void bml_scale_inplace_distributed2d_single_complex(
    void *scale_factor,
    bml_matrix_distributed2d_t * A);

void bml_scale_inplace_distributed2d_double_complex(
    void *scale_factor,
    bml_matrix_distributed2d_t * A);

#endif

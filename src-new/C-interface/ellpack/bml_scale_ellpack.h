#ifndef __BML_SCALE_ELLPACK_H
#define __BML_SCALE_ELLPACK_H

#include "bml_types_ellpack.h"

bml_matrix_ellpack_t *bml_scale_ellpack_new(
    const double scale_factor,
    const bml_matrix_ellpack_t * A);

bml_matrix_ellpack_t *bml_scale_ellpack_new_single_real(
    const double scale_factor,
    const bml_matrix_ellpack_t * A);

bml_matrix_ellpack_t *bml_scale_ellpack_new_double_real(
    const double scale_factor,
    const bml_matrix_ellpack_t * A);

bml_matrix_ellpack_t *bml_scale_ellpack_new_single_complex(
    const double scale_factor,
    const bml_matrix_ellpack_t * A);

bml_matrix_ellpack_t *bml_scale_ellpack_new_double_complex(
    const double scale_factor,
    const bml_matrix_ellpack_t * A);

void bml_scale_ellpack(
    const double scale_factor,
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B);

void bml_scale_ellpack_single_real(
    const double scale_factor,
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B);

void bml_scale_ellpack_double_real(
    const double scale_factor,
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B);

void bml_scale_ellpack_single_complex(
    const double scale_factor,
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B);

void bml_scale_ellpack_double_complex(
    const double scale_factor,
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B);

#endif

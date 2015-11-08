#ifndef __BML_COPY_ELLPACK_H
#define __BML_COPY_ELLPACK_H

#include "bml_types_ellpack.h"

bml_matrix_ellpack_t *bml_copy_ellpack_new(
    const bml_matrix_ellpack_t * A);

bml_matrix_ellpack_t *bml_copy_ellpack_new_single_real(
    const bml_matrix_ellpack_t * A);

bml_matrix_ellpack_t *bml_copy_ellpack_new_double_real(
    const bml_matrix_ellpack_t * A);

bml_matrix_ellpack_t *bml_copy_ellpack_new_single_complex(
    const bml_matrix_ellpack_t * A);

bml_matrix_ellpack_t *bml_copy_ellpack_new_double_complex(
    const bml_matrix_ellpack_t * A);

void bml_copy_ellpack(
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B);

void bml_copy_ellpack_single_real(
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B);

void bml_copy_ellpack_double_real(
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B);

void bml_copy_ellpack_single_complex(
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B);

void bml_copy_ellpack_double_complex(
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B);

#endif

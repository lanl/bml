#ifndef __BML_ALLOCATE_ELLPACK_H
#define __BML_ALLOCATE_ELLPACK_H

#include "bml_types_ellpack.h"

void bml_deallocate_ellpack(
    bml_matrix_ellpack_t * A);

bml_matrix_ellpack_t *bml_zero_matrix_ellpack(
    const bml_matrix_precision_t matrix_precision,
    const int N,
    const int M);

bml_matrix_ellpack_t *bml_random_matrix_ellpack(
    const bml_matrix_precision_t matrix_precision,
    const int N,
    const int M);

bml_matrix_ellpack_t *bml_identity_matrix_ellpack(
    const bml_matrix_precision_t matrix_precision,
    const int N,
    const int M);

#endif

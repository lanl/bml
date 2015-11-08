#ifndef __BML_TRANSPOSE_DENSE_H
#define __BML_TRANSPOSE_DENSE_H

#include "bml_types_dense.h"

bml_matrix_dense_t *bml_transpose_new_dense(
    const bml_matrix_dense_t * A);

bml_matrix_dense_t *bml_transpose_new_dense_single_real(
    const bml_matrix_dense_t * A);

bml_matrix_dense_t *bml_transpose_new_dense_double_real(
    const bml_matrix_dense_t * A);

bml_matrix_dense_t *bml_transpose_new_dense_single_complex(
    const bml_matrix_dense_t * A);

bml_matrix_dense_t *bml_transpose_new_dense_double_complex(
    const bml_matrix_dense_t * A);

void bml_transpose_dense(
    const bml_matrix_dense_t * A);

void bml_transpose_dense_single_real(
    const bml_matrix_dense_t * A);

void bml_transpose_dense_double_real(
    const bml_matrix_dense_t * A);

void bml_transpose_dense_single_complex(
    const bml_matrix_dense_t * A);

void bml_transpose_dense_double_complex(
    const bml_matrix_dense_t * A);

#endif

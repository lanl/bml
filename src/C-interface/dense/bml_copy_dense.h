#ifndef __BML_COPY_DENSE_H
#define __BML_COPY_DENSE_H

#include "bml_types_dense.h"

bml_matrix_dense_t *bml_copy_dense_new(
    const bml_matrix_dense_t * A);

bml_matrix_dense_t *bml_copy_dense_new_single_real(
    const bml_matrix_dense_t * A);

bml_matrix_dense_t *bml_copy_dense_new_double_real(
    const bml_matrix_dense_t * A);

bml_matrix_dense_t *bml_copy_dense_new_single_complex(
    const bml_matrix_dense_t * A);

bml_matrix_dense_t *bml_copy_dense_new_double_complex(
    const bml_matrix_dense_t * A);

void bml_copy_dense(
    const bml_matrix_dense_t * A,
    bml_matrix_dense_t * B);

void bml_copy_dense_single_real(
    const bml_matrix_dense_t * A,
    bml_matrix_dense_t * B);

void bml_copy_dense_double_real(
    const bml_matrix_dense_t * A,
    bml_matrix_dense_t * B);

void bml_copy_dense_single_complex(
    const bml_matrix_dense_t * A,
    bml_matrix_dense_t * B);

void bml_copy_dense_double_complex(
    const bml_matrix_dense_t * A,
    bml_matrix_dense_t * B);

void bml_reorder_single_real(
    bml_matrix_dense_t * A,
    int * perm);

void bml_reorder_dense(
    bml_matrix_dense_t * A,
    int * perm);

void bml_reorder_dense_double_real(
    bml_matrix_dense_t * A,
    int * perm);

void bml_reorder_dense_single_complex(
    bml_matrix_dense_t * A,
    int * perm);

void bml_reorder_dense_double_complex(
    bml_matrix_dense_t * A,
    int * perm);

void bml_save_domain_dense(
     bml_matrix_dense_t * A);

void bml_restore_domain_dense(
     bml_matrix_dense_t * A);

#endif

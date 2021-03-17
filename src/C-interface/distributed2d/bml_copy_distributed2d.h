#ifndef __BML_COPY_DISTRIBUTED2D_H
#define __BML_COPY_DISTRIBUTED2D_H

#include "bml_types_distributed2d.h"

bml_matrix_distributed2d_t *bml_copy_distributed2d_new(
    bml_matrix_distributed2d_t * A);

bml_matrix_distributed2d_t *bml_copy_distributed2d_new_single_real(
    bml_matrix_distributed2d_t * A);

bml_matrix_distributed2d_t *bml_copy_distributed2d_new_double_real(
    bml_matrix_distributed2d_t * A);

bml_matrix_distributed2d_t *bml_copy_distributed2d_new_single_complex(
    bml_matrix_distributed2d_t * A);

bml_matrix_distributed2d_t *bml_copy_distributed2d_new_double_complex(
    bml_matrix_distributed2d_t * A);

void bml_copy_distributed2d(
    bml_matrix_distributed2d_t * A,
    bml_matrix_distributed2d_t * B);

void bml_copy_distributed2d_single_real(
    bml_matrix_distributed2d_t * A,
    bml_matrix_distributed2d_t * B);

void bml_copy_distributed2d_double_real(
    bml_matrix_distributed2d_t * A,
    bml_matrix_distributed2d_t * B);

void bml_copy_distributed2d_single_complex(
    bml_matrix_distributed2d_t * A,
    bml_matrix_distributed2d_t * B);

void bml_copy_distributed2d_double_complex(
    bml_matrix_distributed2d_t * A,
    bml_matrix_distributed2d_t * B);

void bml_reorder_distributed2d(
    bml_matrix_distributed2d_t * A,
    int *perm);

void bml_reorder_distributed2d_single_real(
    bml_matrix_distributed2d_t * A,
    int *perm);

void bml_reorder_distributed2d_double_real(
    bml_matrix_distributed2d_t * A,
    int *perm);

void bml_reorder_distributed2d_single_complex(
    bml_matrix_distributed2d_t * A,
    int *perm);

void bml_reorder_distributed2d_double_complex(
    bml_matrix_distributed2d_t * A,
    int *perm);

void bml_save_domain_distributed2d(
    bml_matrix_distributed2d_t * A);

void bml_restore_domain_distributed2d(
    bml_matrix_distributed2d_t * A);

#endif

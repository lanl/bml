#ifndef __BML_COPY_CSR_H
#define __BML_COPY_CSR_H

#include "bml_types_csr.h"
/*
csr_sparse_row_t * csr_copy_row_new (
    const csr_sparse_row_t * arow);
*/
csr_sparse_row_t *csr_copy_row_new_single_real(
    const csr_sparse_row_t * arow);

csr_sparse_row_t *csr_copy_row_new_double_real(
    const csr_sparse_row_t * arow);

csr_sparse_row_t *csr_copy_row_new_single_complex(
    const csr_sparse_row_t * arow);

csr_sparse_row_t *csr_copy_row_new_double_complex(
    const csr_sparse_row_t * arow);
/*
void csr_copy_row (
    const csr_sparse_row_t * arow,
    const csr_sparse_row_t * brow);
*/
void csr_copy_row_single_real(
    const csr_sparse_row_t * arow,
    csr_sparse_row_t * brow);

void csr_copy_row_double_real(
    const csr_sparse_row_t * arow,
    csr_sparse_row_t * brow);

void csr_copy_row_single_complex(
    const csr_sparse_row_t * arow,
    csr_sparse_row_t * brow);

void csr_copy_row_double_complex(
    const csr_sparse_row_t * arow,
    csr_sparse_row_t * brow);

bml_matrix_csr_t *bml_copy_csr_new(
    bml_matrix_csr_t * A);

bml_matrix_csr_t *bml_copy_csr_new_single_real(
    bml_matrix_csr_t * A);

bml_matrix_csr_t *bml_copy_csr_new_double_real(
    bml_matrix_csr_t * A);

bml_matrix_csr_t *bml_copy_csr_new_single_complex(
    bml_matrix_csr_t * A);

bml_matrix_csr_t *bml_copy_csr_new_double_complex(
    bml_matrix_csr_t * A);

void bml_copy_csr(
    bml_matrix_csr_t * A,
    bml_matrix_csr_t * B);

void bml_copy_csr_single_real(
    bml_matrix_csr_t * A,
    bml_matrix_csr_t * B);

void bml_copy_csr_double_real(
    bml_matrix_csr_t * A,
    bml_matrix_csr_t * B);

void bml_copy_csr_single_complex(
    bml_matrix_csr_t * A,
    bml_matrix_csr_t * B);

void bml_copy_csr_double_complex(
    bml_matrix_csr_t * A,
    bml_matrix_csr_t * B);

void bml_reorder_csr(
    bml_matrix_csr_t * A,
    int *perm);

void bml_reorder_csr_single_real(
    bml_matrix_csr_t * A,
    int *perm);

void bml_reorder_csr_double_real(
    bml_matrix_csr_t * A,
    int *perm);

void bml_reorder_csr_single_complex(
    bml_matrix_csr_t * A,
    int *perm);

void bml_reorder_csr_double_complex(
    bml_matrix_csr_t * A,
    int *perm);


void bml_save_domain_csr(
    bml_matrix_csr_t * A);

void bml_restore_domain_csr(
    bml_matrix_csr_t * A);

#endif

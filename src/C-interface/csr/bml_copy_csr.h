#ifndef __BML_COPY_CSR_H
#define __BML_COPY_CSR_H

#include "bml_types_csr.h"

csr_sparse_row_t * copy_csr_row_new (
    const csr_sparse_row_t * arow);
    
void copy_csr_row (
    const csr_sparse_row_t * arow,
    const csr_sparse_row_t * brow);    
    
bml_matrix_csr_t *bml_copy_csr_new(
    const bml_matrix_csr_t * A);

bml_matrix_csr_t *bml_copy_csr_new_single_real(
    const bml_matrix_csr_t * A);

bml_matrix_csr_t *bml_copy_csr_new_double_real(
    const bml_matrix_csr_t * A);

bml_matrix_csr_t *bml_copy_csr_new_single_complex(
    const bml_matrix_csr_t * A);

bml_matrix_csr_t *bml_copy_csr_new_double_complex(
    const bml_matrix_csr_t * A);

void bml_copy_csr(
    const bml_matrix_csr_t * A,
    const bml_matrix_csr_t * B);

void bml_copy_csr_single_real(
    const bml_matrix_csr_t * A,
    const bml_matrix_csr_t * B);

void bml_copy_csr_double_real(
    const bml_matrix_csr_t * A,
    const bml_matrix_csr_t * B);

void bml_copy_csr_single_complex(
    const bml_matrix_csr_t * A,
    const bml_matrix_csr_t * B);

void bml_copy_csr_double_complex(
    const bml_matrix_csr_t * A,
    const bml_matrix_csr_t * B);

/*
csr_sparse_row_t * copy_csr_new_row (
    const csr_sparse_row_t * arow);  

csr_sparse_row_t * copy_csr_new_row_single_real (
    const csr_sparse_row_t * arow);  
    
csr_sparse_row_t * copy_csr_new_row_double_real (
    const csr_sparse_row_t * arow);  
    
csr_sparse_row_t * copy_csr_new_row_single_complex (
    const csr_sparse_row_t * arow);  
    
csr_sparse_row_t * copy_csr_new_row_double_complex (
    const csr_sparse_row_t * arow);  
*/
#endif

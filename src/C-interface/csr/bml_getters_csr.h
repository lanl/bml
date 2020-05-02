/** \file */

#ifndef __BML_GETTERS_CSR_H
#define __BML_GETTERS_CSR_H

#include "bml_types_csr.h"

#include <complex.h>
/*
void *csr_get_row_element (
    csr_sparse_row_t * arow,
    int j);
*/

void *csr_get_row_element_single_real(
    csr_sparse_row_t * arow,
    int j);

void *csr_get_row_element_double_real(
    csr_sparse_row_t * arow,
    int j);

void *csr_get_row_element_single_complex(
    csr_sparse_row_t * arow,
    int j);

void *csr_get_row_element_double_complex(
    csr_sparse_row_t * arow,
    int j);
/*
int *csr_get_column_indexes(
    csr_sparse_row_t * arow);
*/
int *csr_get_column_indexes_single_real(
    csr_sparse_row_t * arow);

int *csr_get_column_indexes_double_real(
    csr_sparse_row_t * arow);

int *csr_get_column_indexes_single_complex(
    csr_sparse_row_t * arow);

int *csr_get_column_indexes_double_complex(
    csr_sparse_row_t * arow);
/*
void *csr_get_column_entries(
    csr_sparse_row_t * arow);
*/
void *csr_get_column_entries_single_real(
    csr_sparse_row_t * arow);

void *csr_get_column_entries_double_real(
    csr_sparse_row_t * arow);

void *csr_get_column_entries_single_complex(
    csr_sparse_row_t * arow);

void *csr_get_column_entries_double_complex(
    csr_sparse_row_t * arow);
/*
int csr_get_nnz(
    csr_sparse_row_t * arow);
*/
int csr_get_nnz_single_real(
    csr_sparse_row_t * arow);

int csr_get_nnz_double_real(
    csr_sparse_row_t * arow);

int csr_get_nnz_single_complex(
    csr_sparse_row_t * arow);

int csr_get_nnz_double_complex(
    csr_sparse_row_t * arow);

void *bml_get_csr(
    bml_matrix_csr_t * A,
    int i,
    int j);

void *bml_get_csr_single_real(
    bml_matrix_csr_t * A,
    int i,
    int j);

void *bml_get_csr_double_real(
    bml_matrix_csr_t * A,
    int i,
    int j);

void *bml_get_csr_single_complex(
    bml_matrix_csr_t * A,
    int i,
    int j);

void *bml_get_csr_double_complex(
    bml_matrix_csr_t * A,
    int i,
    int j);

void *bml_get_row_csr(
    bml_matrix_csr_t * A,
    int i);

void *bml_get_row_csr_single_real(
    bml_matrix_csr_t * A,
    int i);

void *bml_get_row_csr_double_real(
    bml_matrix_csr_t * A,
    int i);

void *bml_get_row_csr_single_complex(
    bml_matrix_csr_t * A,
    int i);

void *bml_get_row_csr_double_complex(
    bml_matrix_csr_t * A,
    int i);

void *bml_get_diagonal_csr(
    bml_matrix_csr_t * A);

void *bml_get_diagonal_csr_single_real(
    bml_matrix_csr_t * A);

void *bml_get_diagonal_csr_double_real(
    bml_matrix_csr_t * A);

void *bml_get_diagonal_csr_single_complex(
    bml_matrix_csr_t * A);

void *bml_get_diagonal_csr_double_complex(
    bml_matrix_csr_t * A);

void bml_get_sparse_row_csr(
    bml_matrix_csr_t * A,
    int i,
    int **cols,
    void **vals,
    int *nnz);

void bml_get_sparse_row_csr_single_real(
    bml_matrix_csr_t * A,
    int i,
    int **cols,
    void **vals,
    int *nnz);

void bml_get_sparse_row_csr_double_real(
    bml_matrix_csr_t * A,
    int i,
    int **cols,
    void **vals,
    int *nnz);

void bml_get_sparse_row_csr_single_complex(
    bml_matrix_csr_t * A,
    int i,
    int **cols,
    void **vals,
    int *nnz);

void bml_get_sparse_row_csr_double_complex(
    bml_matrix_csr_t * A,
    int i,
    int **cols,
    void **vals,
    int *nnz);
#endif

#ifndef __BML_SUBMATRIX_CSR_H
#define __BML_SUBMATRIX_CSR_H

#include "bml_types_csr.h"

bml_matrix_csr_t *bml_extract_submatrix_csr(
    bml_matrix_csr_t * A,
    int irow,
    int icol,
    int B_N,
    int B_M);

bml_matrix_csr_t
    * bml_extract_submatrix_csr_single_real(bml_matrix_csr_t * A, int irow,
                                            int icol,
                                            int B_N,
                                            int B_M);

bml_matrix_csr_t
    * bml_extract_submatrix_csr_double_real(bml_matrix_csr_t * A, int irow,
                                            int icol,
                                            int B_N,
                                            int B_M);

bml_matrix_csr_t
    * bml_extract_submatrix_csr_single_complex(bml_matrix_csr_t * A, int irow,
                                               int icol,
                                               int B_N,
                                               int B_M);

bml_matrix_csr_t
    * bml_extract_submatrix_csr_double_complex(bml_matrix_csr_t * A, int irow,
                                               int icol,
                                               int B_N,
                                               int B_M);


void bml_assign_submatrix_csr(
    bml_matrix_csr_t * A,
    bml_matrix_csr_t * B,
    int irow,
    int icol);

void bml_assign_submatrix_csr_single_real(
    bml_matrix_csr_t * A,
    bml_matrix_csr_t * B,
    int irow,
    int icol);

void bml_assign_submatrix_csr_double_real(
    bml_matrix_csr_t * A,
    bml_matrix_csr_t * B,
    int irow,
    int icol);

void bml_assign_submatrix_csr_single_complex(
    bml_matrix_csr_t * A,
    bml_matrix_csr_t * B,
    int irow,
    int icol);

void bml_assign_submatrix_csr_double_complex(
    bml_matrix_csr_t * A,
    bml_matrix_csr_t * B,
    int irow,
    int icol);

#endif

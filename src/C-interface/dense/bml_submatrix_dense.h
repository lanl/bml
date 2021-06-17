#ifndef __BML_SUBMATRIX_DENSE_H
#define __BML_SUBMATRIX_DENSE_H

#include "bml_types_dense.h"

bml_matrix_dense_t *bml_extract_submatrix_dense(
    bml_matrix_dense_t * A,
    int irow,
    int icol,
    int B_N,
    int B_M);

bml_matrix_dense_t
    * bml_extract_submatrix_dense_single_real(bml_matrix_dense_t * A,
                                              int irow,
                                              int icol,
                                              int B_N,
                                              int B_M);

bml_matrix_dense_t
    * bml_extract_submatrix_dense_double_real(bml_matrix_dense_t * A,
                                              int irow,
                                              int icol,
                                              int B_N,
                                              int B_M);

bml_matrix_dense_t
    * bml_extract_submatrix_dense_single_complex(bml_matrix_dense_t * A,
                                                 int irow,
                                                 int icol,
                                                 int B_N,
                                                 int B_M);

bml_matrix_dense_t
    * bml_extract_submatrix_dense_double_complex(bml_matrix_dense_t * A,
                                                 int irow,
                                                 int icol,
                                                 int B_N,
                                                 int B_M);

void bml_assign_submatrix_dense(
    bml_matrix_dense_t * A,
    bml_matrix_dense_t * B,
    int irow,
    int icol);

void bml_assign_submatrix_dense_single_real(
    bml_matrix_dense_t * A,
    bml_matrix_dense_t * B,
    int irow,
    int icol);

void bml_assign_submatrix_dense_double_real(
    bml_matrix_dense_t * A,
    bml_matrix_dense_t * B,
    int irow,
    int icol);

void bml_assign_submatrix_dense_single_complex(
    bml_matrix_dense_t * A,
    bml_matrix_dense_t * B,
    int irow,
    int icol);

void bml_assign_submatrix_dense_double_complex(
    bml_matrix_dense_t * A,
    bml_matrix_dense_t * B,
    int irow,
    int icol);

#endif

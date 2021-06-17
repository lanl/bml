#ifndef __BML_SUBMATRIX_ELLBLOCK_H
#define __BML_SUBMATRIX_ELLBLOCK_H

#include "bml_types_ellblock.h"

bml_matrix_ellblock_t *bml_extract_submatrix_ellblock(
    bml_matrix_ellblock_t * A,
    int irow,
    int icol,
    int B_N,
    int B_M);

bml_matrix_ellblock_t
    * bml_extract_submatrix_ellblock_single_real(bml_matrix_ellblock_t * A,
                                                 int irow,
                                                 int icol,
                                                 int B_N,
                                                 int B_M);

bml_matrix_ellblock_t
    * bml_extract_submatrix_ellblock_double_real(bml_matrix_ellblock_t * A,
                                                 int irow,
                                                 int icol,
                                                 int B_N,
                                                 int B_M);

bml_matrix_ellblock_t
    * bml_extract_submatrix_ellblock_single_complex(bml_matrix_ellblock_t * A,
                                                    int irow,
                                                    int icol,
                                                    int B_N,
                                                    int B_M);

bml_matrix_ellblock_t
    * bml_extract_submatrix_ellblock_double_complex(bml_matrix_ellblock_t * A,
                                                    int irow,
                                                    int icol,
                                                    int B_N,
                                                    int B_M);

void bml_assign_submatrix_ellblock(
    bml_matrix_ellblock_t * A,
    bml_matrix_ellblock_t * B,
    int irow,
    int icol);

void bml_assign_submatrix_ellblock_single_real(
    bml_matrix_ellblock_t * A,
    bml_matrix_ellblock_t * B,
    int irow,
    int icol);

void bml_assign_submatrix_ellblock_double_real(
    bml_matrix_ellblock_t * A,
    bml_matrix_ellblock_t * B,
    int irow,
    int icol);

void bml_assign_submatrix_ellblock_single_complex(
    bml_matrix_ellblock_t * A,
    bml_matrix_ellblock_t * B,
    int irow,
    int icol);

void bml_assign_submatrix_ellblock_double_complex(
    bml_matrix_ellblock_t * A,
    bml_matrix_ellblock_t * B,
    int irow,
    int icol);

#endif

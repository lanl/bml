#ifndef __BML_TRANSPOSE_BELLBLOCK_H
#define __BML_TRANSPOSE_BELLBLOCK_H

#include "bml_types_ellblock.h"

void ellblock_swap_block_row_entries_single_real(
    bml_matrix_ellblock_t * A,
    const int block_row,
    const int iposb,
    const int jposb);

void ellblock_swap_block_row_entries_double_real(
    bml_matrix_ellblock_t * A,
    const int block_row,
    const int iposb,
    const int jposb);

void ellblock_swap_block_row_entries_single_complex(
    bml_matrix_ellblock_t * A,
    const int block_row,
    const int iposb,
    const int jposb);

void ellblock_swap_block_row_entries_double_complex(
    bml_matrix_ellblock_t * A,
    const int block_row,
    const int iposb,
    const int jposb);

bml_matrix_ellblock_t *bml_transpose_new_ellblock(
    bml_matrix_ellblock_t * A);

bml_matrix_ellblock_t
    * bml_transpose_new_ellblock_single_real(bml_matrix_ellblock_t * A);

bml_matrix_ellblock_t
    * bml_transpose_new_ellblock_double_real(bml_matrix_ellblock_t * A);

bml_matrix_ellblock_t
    * bml_transpose_new_ellblock_single_complex(bml_matrix_ellblock_t * A);

bml_matrix_ellblock_t
    * bml_transpose_new_ellblock_double_complex(bml_matrix_ellblock_t * A);

void bml_transpose_ellblock(
    bml_matrix_ellblock_t * A);

void bml_transpose_ellblock_single_real(
    bml_matrix_ellblock_t * A);

void bml_transpose_ellblock_double_real(
    bml_matrix_ellblock_t * A);

void bml_transpose_ellblock_single_complex(
    bml_matrix_ellblock_t * A);

void bml_transpose_ellblock_double_complex(
    bml_matrix_ellblock_t * A);

#endif

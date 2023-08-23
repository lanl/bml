#ifndef __BML_TRANSPOSE_ELLPACK_H
#define __BML_TRANSPOSE_ELLPACK_H

#include "bml_types_ellpack.h"

void ellpack_swap_row_entries_single_real (
    bml_matrix_ellpack_t * A,
    const int row,
    const int ipos,
    const int jpos);

void ellpack_swap_row_entries_double_real (
    bml_matrix_ellpack_t * A,
    const int row,
    const int ipos,
    const int jpos);

void ellpack_swap_row_entries_single_complex (
    bml_matrix_ellpack_t * A,
    const int row,
    const int ipos,
    const int jpos);

void ellpack_swap_row_entries_double_complex (
    bml_matrix_ellpack_t * A,
    const int row,
    const int ipos,
    const int jpos);

void ellpack_transpose_left_looking_single_real (
    bml_matrix_ellpack_t * A);

void ellpack_transpose_left_looking_double_real (
    bml_matrix_ellpack_t * A);

void ellpack_transpose_left_looking_single_complex (
    bml_matrix_ellpack_t * A);

void ellpack_transpose_left_looking_double_complex (
    bml_matrix_ellpack_t * A);

void ellpack_transpose_right_looking_single_real (
    bml_matrix_ellpack_t * A);

void ellpack_transpose_right_looking_double_real (
    bml_matrix_ellpack_t * A);

void ellpack_transpose_right_looking_single_complex (
    bml_matrix_ellpack_t * A);

void ellpack_transpose_right_looking_double_complex (
    bml_matrix_ellpack_t * A);

bml_matrix_ellpack_t *bml_transpose_new_ellpack(
    bml_matrix_ellpack_t * A);

bml_matrix_ellpack_t
    * bml_transpose_new_ellpack_single_real(bml_matrix_ellpack_t * A);

bml_matrix_ellpack_t
    * bml_transpose_new_ellpack_double_real(bml_matrix_ellpack_t * A);

bml_matrix_ellpack_t
    * bml_transpose_new_ellpack_single_complex(bml_matrix_ellpack_t * A);

bml_matrix_ellpack_t
    * bml_transpose_new_ellpack_double_complex(bml_matrix_ellpack_t * A);

void bml_transpose_ellpack(
    bml_matrix_ellpack_t * A);

void bml_transpose_ellpack_single_real(
    bml_matrix_ellpack_t * A);

void bml_transpose_ellpack_double_real(
    bml_matrix_ellpack_t * A);

void bml_transpose_ellpack_single_complex(
    bml_matrix_ellpack_t * A);

void bml_transpose_ellpack_double_complex(
    bml_matrix_ellpack_t * A);

#if defined(BML_USE_CUSPARSE)
void bml_transpose_cusparse_ellpack(
    bml_matrix_ellpack_t * A);

void bml_transpose_cusparse_ellpack_single_real(
    bml_matrix_ellpack_t * A);

void bml_transpose_cusparse_ellpack_double_real(
    bml_matrix_ellpack_t * A);

void bml_transpose_cusparse_ellpack_single_complex(
    bml_matrix_ellpack_t * A);

void bml_transpose_cusparse_ellpack_double_complex(
    bml_matrix_ellpack_t * A);
#endif
#endif

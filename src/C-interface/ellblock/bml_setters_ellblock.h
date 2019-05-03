/** \file */

#ifndef __BML_SETTERS_ELLBLOCK_H
#define __BML_SETTERS_ELLBLOCK_H

#include "bml_types_ellblock.h"

#include <complex.h>

void bml_set_element_new_ellblock(
    bml_matrix_ellblock_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_element_new_ellblock_single_real(
    bml_matrix_ellblock_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_element_new_ellblock_double_real(
    bml_matrix_ellblock_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_element_new_ellblock_single_complex(
    bml_matrix_ellblock_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_element_new_ellblock_double_complex(
    bml_matrix_ellblock_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_element_ellblock(
    bml_matrix_ellblock_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_element_ellblock_single_real(
    bml_matrix_ellblock_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_element_ellblock_double_real(
    bml_matrix_ellblock_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_element_ellblock_single_complex(
    bml_matrix_ellblock_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_element_ellblock_double_complex(
    bml_matrix_ellblock_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_row_ellblock(
    bml_matrix_ellblock_t * A,
    const int i,
    const void *row,
    const double threshold);

void bml_set_row_ellblock_single_real(
    bml_matrix_ellblock_t * A,
    const int i,
    const float *row,
    const double threshold);

void bml_set_row_ellblock_double_real(
    bml_matrix_ellblock_t * A,
    const int i,
    const double *row,
    const double threshold);

void bml_set_row_ellblock_single_complex(
    bml_matrix_ellblock_t * A,
    const int i,
    const float complex * row,
    const double threshold);

void bml_set_row_ellblock_double_complex(
    bml_matrix_ellblock_t * A,
    const int i,
    const double complex * row,
    const double threshold);

void bml_set_diagonal_ellblock(
    bml_matrix_ellblock_t * A,
    const void *diagonal,
    const double threshold);

void bml_set_diagonal_ellblock_single_real(
    bml_matrix_ellblock_t * A,
    const float *diagonal,
    const double threshold);

void bml_set_diagonal_ellblock_double_real(
    bml_matrix_ellblock_t * A,
    const double *diagonal,
    const double threshold);

void bml_set_diagonal_ellblock_single_complex(
    bml_matrix_ellblock_t * A,
    const float complex * diagonal,
    const double threshold);

void bml_set_diagonal_ellblock_double_complex(
    bml_matrix_ellblock_t * A,
    const double complex * diagonal,
    const double threshold);

void bml_set_block_ellblock_single_real(
    bml_matrix_ellblock_t * A,
    const int ib,
    const int jb,
    const float *values);

void bml_set_block_ellblock_double_real(
    bml_matrix_ellblock_t * A,
    const int ib,
    const int jb,
    const double *values);

void bml_set_block_ellblock_single_complex(
    bml_matrix_ellblock_t * A,
    const int ib,
    const int jb,
    const float complex * values);

void bml_set_block_ellblock_double_complex(
    bml_matrix_ellblock_t * A,
    const int ib,
    const int jb,
    const double complex * values);



#endif

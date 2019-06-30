/** \file */

#ifndef __BML_SETTERS_DENSE_H
#define __BML_SETTERS_DENSE_H

#include "bml_types_dense.h"

#include <complex.h>

void bml_set_element_dense(
    bml_matrix_dense_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_element_dense_single_real(
    bml_matrix_dense_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_element_dense_double_real(
    bml_matrix_dense_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_element_dense_single_complex(
    bml_matrix_dense_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_element_dense_double_complex(
    bml_matrix_dense_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_row_dense(
    bml_matrix_dense_t * A,
    const int i,
    const void *row);

void bml_set_row_dense_single_real(
    bml_matrix_dense_t * A,
    const int i,
    const void *row);

void bml_set_row_dense_double_real(
    bml_matrix_dense_t * A,
    const int i,
    const void *row);

void bml_set_row_dense_single_complex(
    bml_matrix_dense_t * A,
    const int i,
    const void *row);

void bml_set_row_dense_double_complex(
    bml_matrix_dense_t * A,
    const int i,
    const void *row);

void bml_set_diagonal_dense(
    bml_matrix_dense_t * A,
    const void *diagonal);

void bml_set_diagonal_dense_single_real(
    bml_matrix_dense_t * A,
    const void *diagonal);

void bml_set_diagonal_dense_double_real(
    bml_matrix_dense_t * A,
    const void *diagonal);

void bml_set_diagonal_dense_single_complex(
    bml_matrix_dense_t * A,
    const void *diagonal);

void bml_set_diagonal_dense_double_complex(
    bml_matrix_dense_t * A,
    const void *diagonal);

#endif

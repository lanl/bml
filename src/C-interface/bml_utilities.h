/** \file */

#ifndef __BML_UTILITIES_H
#define __BML_UTILITIES_H

#include "bml_types.h"

void bml_print_dense_matrix(
    const int N,
    const bml_matrix_precision_t matrix_precision,
    const bml_dense_order_t order,
    const void *A,
    const int i_l,
    const int i_u,
    const int j_l,
    const int j_u);

void bml_print_dense_vector(
    const int N,
    bml_matrix_precision_t matrix_precision,
    const void *v,
    const int i_l,
    const int i_u);

void bml_print_bml_vector(
    const bml_vector_t * v,
    const int i_l,
    const int i_u);

void bml_print_bml_matrix(
    const bml_matrix_t * A,
    const int i_l,
    const int i_u,
    const int j_l,
    const int j_u);

#endif

#ifndef __BML_ELEMNET_MULTIPLY_DENSE_H
#define __BML_ELEMNET_MULTIPLY_DENSE_H

#include "bml_types_dense.h"

// Element-wise Matrix multiply (Hadamard product)
void bml_element_multiply_AB_dense(
    bml_matrix_dense_t * A,
    bml_matrix_dense_t * B,
    bml_matrix_dense_t * C);

void bml_element_multiply_AB_dense_single_real(
    bml_matrix_dense_t * A,
    bml_matrix_dense_t * B,
    bml_matrix_dense_t * C);

void bml_element_multiply_AB_dense_double_real(
    bml_matrix_dense_t * A,
    bml_matrix_dense_t * B,
    bml_matrix_dense_t * C);

void bml_element_multiply_AB_dense_single_complex(
    bml_matrix_dense_t * A,
    bml_matrix_dense_t * B,
    bml_matrix_dense_t * C);

void bml_element_multiply_AB_dense_double_complex(
    bml_matrix_dense_t * A,
    bml_matrix_dense_t * B,
    bml_matrix_dense_t * C);

#endif

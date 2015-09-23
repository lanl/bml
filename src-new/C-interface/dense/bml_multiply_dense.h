#ifndef __BML_MULTIPLY_DENSE_H
#define __BML_MULTIPLY_DENSE_H

#include "bml_types_dense.h"

// Matrix multiply - C = alpha * A * B + beta * C
void bml_multiply_dense(const bml_matrix_dense_t *A, const bml_matrix_dense_t *B, const bml_matrix_dense_t *C, const double alpha, const double beta);

#endif

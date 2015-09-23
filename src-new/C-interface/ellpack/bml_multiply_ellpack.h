#ifndef __BML_MULTIPLY_ELLPACK_H
#define __BML_MULTIPLY_ELLPACK_H

#include "bml_types_ellpack.h"

// Matrix multiply - C = alpha *A * B + beta * C
void bml_multiply_ellpack(const bml_matrix_ellpack_t *A, const bml_matrix_ellpack_t *B, const bml_matrix_ellpack_t *C, const double alpha, const double beta, const double threshold);

// Sparse X^2
void bml_multiplyX2_ellpack_double(const bml_matrix_ellpack_t *xmatrix, const bml_matrix_ellpack_t *x2matrix, double trX, double trX2, const double threshold);

#endif

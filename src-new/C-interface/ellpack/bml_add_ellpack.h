#ifndef __BML_ADD_ELLPACK_H
#define __BML_ADD_ELLPACK_H

#include "bml_types_ellpack.h"

void bml_add_ellpack(const bml_matrix_ellpack_t *A, const bml_matrix_ellpack_t *B, const double alpha, const double beta, const double threshold);

void bml_add_identity_ellpack(const bml_matrix_ellpack_t *A, const double beta, const double threshold);

// Sparse matrix add
void bml_add_ellpack_double(const bml_matrix_ellpack_t *xmatrix, const bml_matrix_ellpack_t *x2matrix, const double alpha, const double beta, const double threshold);

#endif

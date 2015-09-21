#ifndef __BML_COPY_DENSE_H
#define __BML_COPY_DENSE_H

#include "bml_types_dense.h"

bml_matrix_dense_t *bml_copy_dense_new(const bml_matrix_dense_t *A);
void bml_copy_dense(const bml_matrix_dense_t *A, bml_matrix_dense_t *B);

#endif

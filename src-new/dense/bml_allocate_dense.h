#ifndef __BML_ALLOCATE_DENSE_H
#define __BML_ALLOCATE_DENSE_H

#include "bml_types_dense.h"

void bml_allocate_dense(const bml_matrix_precision_t matrix_precision,
                        bml_matrix_dense_t *A,
                        const int N);

void bml_deallocate_dense(bml_matrix_dense_t *A);

#endif

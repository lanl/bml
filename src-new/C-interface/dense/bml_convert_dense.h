#ifndef __BML_CONVERT_DENSE_H
#define __BML_CONVERT_DENSE_H

#include "../bml_types.h"
#include "bml_types_dense.h"

bml_matrix_dense_t *bml_convert_from_dense_dense(const bml_matrix_precision_t matrix_precision,
                                                 const int N,
                                                 const void *A,
                                                 const double threshold);

void *bml_convert_to_dense_dense(const bml_matrix_dense_t *A);

#endif

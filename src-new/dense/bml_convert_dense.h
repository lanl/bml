#ifndef __BML_CONVERT_DENSE_H
#define __BML_CONVERT_DENSE_H

#include "bml_types_dense.h"

bml_matrix_dense_t *bml_convert_from_dense_dense(const int N,
                                                 const double *A_dense,
                                                 const double threshold);

double *bml_convert_to_dense_dense(const bml_matrix_dense_t *A);

#endif

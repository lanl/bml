#ifndef __BML_CONVERT_ELLPACK_H
#define __BML_CONVERT_ELLPACK_H

#include "../bml_types.h"
#include "bml_types_ellpack.h"

bml_matrix_ellpack_t *bml_convert_from_dense_ellpack(const bml_matrix_precision_t matrix_precision,
                                                 const int N,
                                                 const void *A,
                                                 const double threshold,
                                                 const int M);

void *bml_convert_to_dense_ellpack(const bml_matrix_ellpack_t *A);

#endif

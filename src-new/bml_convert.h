#ifndef __BML_CONVERT_H
#define __BML_CONVERT_H

#include "bml_types.h"

bml_matrix_t *bml_convert_from_dense(const bml_matrix_type_t matrix_type,
                                     const double *A,
                                     const double threshold);

double *bml_convert_to_dense(const bml_matrix_t *A);

#endif

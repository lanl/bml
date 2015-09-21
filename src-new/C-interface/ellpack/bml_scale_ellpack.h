#ifndef __BML_SCALE_ELLPACK_H
#define __BML_SCALE_ELLPACK_H

#include "bml_types_ellpack.h"

bml_matrix_ellpack_t *bml_scale_ellpack_new(const double scale_factor, const bml_matrix_ellpack_t *A);
void bml_scale_ellpack(const double scale_factor, const bml_matrix_ellpack_t *A, bml_matrix_ellpack_t *B);

#endif

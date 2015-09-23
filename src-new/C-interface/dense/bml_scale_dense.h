#ifndef __BML_SCALE_DENSE_H
#define __BML_SCALE_DENSE_H

#include "bml_types_dense.h"

bml_matrix_dense_t *bml_scale_dense_new(
    const double scale_factor,
    const bml_matrix_dense_t * A);

void bml_scale_dense(
    const double scale_factor,
    const bml_matrix_dense_t * A,
    const bml_matrix_dense_t * B);

#endif

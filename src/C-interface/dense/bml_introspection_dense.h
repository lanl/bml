#ifndef __BML_INTROSPECTION_DENSE_H
#define __BML_INTROSPECTION_DENSE_H

#include "bml_types_dense.h"

bml_matrix_precision_t bml_get_precision_dense(
    const bml_matrix_dense_t * A);

int bml_get_N_dense(
    const bml_matrix_dense_t * A);

int bml_get_M_dense(
    const bml_matrix_dense_t * A);

#endif

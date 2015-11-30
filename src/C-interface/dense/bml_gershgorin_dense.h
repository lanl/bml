#ifndef __BML_GERSHGORIN_DENSE_H
#define __BML_GERSHGORIN_DENSE_H

#include "bml_types_dense.h"

void *bml_gershgorin_dense(
    const bml_matrix_dense_t * A);

void *bml_gershgorin_dense_single_real(
    const bml_matrix_dense_t * A);

void *bml_gershgorin_dense_double_real(
    const bml_matrix_dense_t * A);

void *bml_gershgorin_dense_single_complex(
    const bml_matrix_dense_t * A);

void *bml_gershgorin_dense_double_complex(
    const bml_matrix_dense_t * A);

#endif

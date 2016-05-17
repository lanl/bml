#ifndef __BML_PARALLEL_DENSE_H
#define __BML_PARALLEL_DENSE_H

#include "bml_types_dense.h"

void bml_allGatherVParallel_dense(
    bml_matrix_dense_t * A);

void bml_allGatherVParallel_dense_single_real(
    bml_matrix_dense_t * A);

void bml_allGatherVParallel_dense_double_real(
    bml_matrix_dense_t * A);

void bml_allGatherVParallel_dense_single_complex(
    bml_matrix_dense_t * A);

void bml_allGatherVParallel_dense_double_complex(
    bml_matrix_dense_t * A);

#endif

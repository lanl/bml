#ifndef __BML_THRESHOLD_DENSE_H
#define __BML_THRESHOLD_DENSE_H

#include "bml_types_dense.h"

bml_matrix_dense_t *bml_threshold_new_dense(
    const bml_matrix_dense_t * A, const double threshold);

bml_matrix_dense_t *bml_threshold_new_dense_single_real(
    const bml_matrix_dense_t * A, const double threshold);

bml_matrix_dense_t *bml_threshold_new_dense_double_real(
    const bml_matrix_dense_t * A, const double threshold);

bml_matrix_dense_t *bml_threshold_new_dense_single_complex(
    const bml_matrix_dense_t * A, const double threshold);

bml_matrix_dense_t *bml_threshold_new_dense_double_complex(
    const bml_matrix_dense_t * A, const double threshold);

void bml_threshold_dense(
    const bml_matrix_dense_t * A, const double threshold);

void bml_threshold_dense_single_real(
    const bml_matrix_dense_t * A, const double threshold);

void bml_threshold_dense_double_real(
    const bml_matrix_dense_t * A, const double threshold);

void bml_threshold_dense_single_complex(
    const bml_matrix_dense_t * A, const double threshold);

void bml_threshold_dense_double_complex(
    const bml_matrix_dense_t * A, const double threshold);

#endif
